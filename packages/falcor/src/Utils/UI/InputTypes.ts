/**
 * Mirrors Utils/UI/InputTypes.h: the mouse and keyboard events SampleApp hands to its
 * callbacks, plus adapters from DOM events. Keys use native's Input::Key names
 * ("A".."Z", "Key0".."Key9", "Space", "Escape", "Enter", "Left", "F1", ...).
 */

import { Logger } from "../Logger.js";

export enum MouseEventType {
    ButtonDown,
    ButtonUp,
    Move,
    Wheel,
}

export enum MouseButton {
    Left,
    Middle,
    Right,
    Unknown,
}

export enum KeyboardEventType {
    KeyPressed,
    KeyReleased,
    KeyRepeated,
    Input,
}

/** Mirrors Input::ModifierFlags. */
export enum ModifierFlags {
    None = 0,
    Shift = 1,
    Ctrl = 2,
    Alt = 4,
}

export interface MouseEvent {
    type: MouseEventType;
    /** Normalized [0, 1] coordinates, (0, 0) top-left. */
    pos: [number, number];
    /** Pixel coordinates in [0, clientSize]. */
    screenPos: [number, number];
    wheelDelta: [number, number];
    mods: ModifierFlags;
    button: MouseButton;
}

export interface KeyboardEvent {
    type: KeyboardEventType;
    key: string;
    mods: ModifierFlags;
    /** UTF-32 codepoint for Input events. */
    codepoint: number;
}

const mods = (e: { shiftKey: boolean; ctrlKey: boolean; altKey: boolean }): ModifierFlags =>
    (e.shiftKey ? ModifierFlags.Shift : 0) | (e.ctrlKey ? ModifierFlags.Ctrl : 0) | (e.altKey ? ModifierFlags.Alt : 0);

/** DOM KeyboardEvent.code -> native Input::Key name. */
export function keyFromCode(code: string): string {
    if (code.startsWith("Key")) return code.slice(3);
    if (code.startsWith("Digit")) return `Key${code.slice(5)}`;
    if (code.startsWith("Arrow")) return code.slice(5);
    const map: Record<string, string> = { ShiftLeft: "LeftShift", ShiftRight: "RightShift", ControlLeft: "LeftControl", ControlRight: "RightControl", AltLeft: "LeftAlt", AltRight: "RightAlt" };
    return map[code] ?? code;
}

const kAsciiKeys: Record<string, string> = { Space: " ", Quote: "'", Comma: ",", Minus: "-", Period: ".", Slash: "/", Semicolon: ";", Equal: "=", BracketLeft: "[", Backslash: "\\", BracketRight: "]", Backquote: "`" };
/** Input::Key's special keys, in enum order from 256. */
const kSpecialKeys = ["Escape", "Tab", "Enter", "Backspace", "Insert", "Delete", "ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "PageUp", "PageDown", "Home", "End", "CapsLock", "ScrollLock", "NumLock", "PrintScreen", "Pause",
    ...Array.from({ length: 12 }, (_, i) => `F${i + 1}`), ...Array.from({ length: 10 }, (_, i) => `Numpad${i}`), "NumpadDecimal", "NumpadDivide", "NumpadMultiply", "NumpadSubtract", "NumpadAdd", "NumpadEnter", "NumpadEqual",
    "ShiftLeft", "ControlLeft", "AltLeft", "MetaLeft", "ShiftRight", "ControlRight", "AltRight", "MetaRight", "ContextMenu"];

/** DOM KeyboardEvent.code -> native Input::Key value (ASCII below 256, Unknown if unmapped). */
export function nativeKeyCode(code: string): number {
    if (/^Key[A-Z]$/.test(code)) return code.charCodeAt(3);
    if (/^Digit[0-9]$/.test(code)) return code.charCodeAt(5);
    if (kAsciiKeys[code]) return kAsciiKeys[code]!.charCodeAt(0);
    const i = kSpecialKeys.indexOf(code);
    return i >= 0 ? 256 + i : 256 + kSpecialKeys.length;
}

export function toKeyboardEvent(e: globalThis.KeyboardEvent, type: KeyboardEventType.KeyPressed | KeyboardEventType.KeyReleased): KeyboardEvent {
    return { type: type === KeyboardEventType.KeyPressed && e.repeat ? KeyboardEventType.KeyRepeated : type, key: keyFromCode(e.code), mods: mods(e), codepoint: 0 };
}

export function toMouseEvent(e: globalThis.MouseEvent | WheelEvent, type: MouseEventType, element: Element): MouseEvent {
    const rect = element.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const button = e.button === 0 ? MouseButton.Left : e.button === 1 ? MouseButton.Middle : e.button === 2 ? MouseButton.Right : MouseButton.Unknown;
    const wheel = type === MouseEventType.Wheel ? [-(e as WheelEvent).deltaX / 100, -(e as WheelEvent).deltaY / 100] : [0, 0];
    return { type, pos: [x / rect.width, y / rect.height], screenPos: [x, y], wheelDelta: wheel as [number, number], mods: mods(e), button };
}

/** Mirrors GamepadButton. */
export enum GamepadButton {
    A,
    B,
    X,
    Y,
    LeftBumper,
    RightBumper,
    Back,
    Start,
    Guide,
    LeftThumb,
    RightThumb,
    Up,
    Right,
    Down,
    Left,
    Count,
}

export enum GamepadEventType {
    ButtonDown,
    ButtonUp,
    Connected,
    Disconnected,
}

export interface GamepadEvent {
    type: GamepadEventType;
    button?: GamepadButton;
}

/** Mirrors GamepadState: GLFW axes in [-1, 1], y down; triggers too (-1 = released). */
export interface GamepadState {
    leftX: number;
    leftY: number;
    rightX: number;
    rightY: number;
    leftTrigger: number;
    rightTrigger: number;
    buttons: boolean[];
}

export interface GamepadCallbacks {
    handleGamepadEvent(event: GamepadEvent): void;
    handleGamepadState(state: GamepadState): void;
}

/** W3C "standard" mapping button index of each GamepadButton. */
const kStandardButtons = [0, 1, 2, 3, 4, 5, 8, 9, 16, 10, 11, 12, 15, 13, 14];

/** Mirrors Window::handleGamepadInput on the Gamepad API: call once per frame. */
export class GamepadInput {
    private activeIndex = -1;
    private previous: boolean[] = [];

    poll(callbacks: GamepadCallbacks): void {
        const pads = typeof navigator !== "undefined" && typeof navigator.getGamepads === "function" ? navigator.getGamepads() : [];
        const isGamepad = (p: Gamepad | null | undefined): p is Gamepad => !!p && p.connected && p.mapping === "standard";
        if (this.activeIndex < 0) {
            const pad = pads.find(isGamepad);
            if (pad) {
                Logger.info(`Gamepad '${pad.id}' connected.`);
                this.activeIndex = pad.index;
                this.previous = [];
                callbacks.handleGamepadEvent({ type: GamepadEventType.Connected });
            }
        }
        const pad = this.activeIndex >= 0 ? pads[this.activeIndex] : null;
        if (this.activeIndex >= 0 && !isGamepad(pad)) {
            Logger.info("Gamepad disconnected.");
            this.activeIndex = -1;
            callbacks.handleGamepadEvent({ type: GamepadEventType.Disconnected });
        }
        if (!isGamepad(pad)) return;

        const axis = (i: number) => pad.axes[i] ?? 0;
        const trigger = (i: number) => (pad.buttons[i]?.value ?? 0) * 2 - 1;
        const buttons = kStandardButtons.map((i) => pad.buttons[i]?.pressed ?? false);
        const state: GamepadState = { leftX: axis(0), leftY: axis(1), rightX: axis(2), rightY: axis(3), leftTrigger: trigger(6), rightTrigger: trigger(7), buttons };
        // Synthesize gamepad button events.
        for (let b = 0; b < GamepadButton.Count; ++b) {
            if (buttons[b] && !this.previous[b]) callbacks.handleGamepadEvent({ type: GamepadEventType.ButtonDown, button: b });
            if (!buttons[b] && this.previous[b]) callbacks.handleGamepadEvent({ type: GamepadEventType.ButtonUp, button: b });
        }
        this.previous = buttons;
        callbacks.handleGamepadState(state);
    }
}
