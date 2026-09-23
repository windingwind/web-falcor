/**
 * Mirrors Utils/UI/InputTypes.h: the mouse and keyboard events SampleApp hands to its
 * callbacks, plus adapters from DOM events. Keys use native's Input::Key names
 * ("A".."Z", "Key0".."Key9", "Space", "Escape", "Enter", "Left", "F1", ...).
 */

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
