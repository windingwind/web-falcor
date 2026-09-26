// Viewer camera input: forwards DOM mouse/keyboard events and gamepad state to the
// scene's own camera controller (Scene::onMouseEvent/onKeyEvent/onGamepadState).
// update() returns true when the camera moved so the caller resets accumulation.
import {
    CameraControllerType as NativeControllerType,
    KeyboardEventType,
    MouseEventType,
    UpDirection,
    add3,
    mul3,
    normalize3,
    sub3,
    toKeyboardEvent,
    toMouseEvent,
    type GamepadState,
    type Scene,
} from "@web-falcor/falcor";

/** Mirrors Scene::CameraControllerType (dropdown spellings from Scene.cpp). */
export const kCameraControllerTypes = ["First Person", "Orbiter", "6-DOF"] as const;
export type CameraControllerType = (typeof kCameraControllerTypes)[number];
export const kUpDirectionNames = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"] as const;

export class CameraController {
    private dollyAccum = 0; // wheel dolly for the first-person controllers (web extra; native ignores the wheel there)
    /** Called after the scene handled F3 (Scene::addViewpoint) so the viewpoint list refreshes. */
    onViewpointAdded: () => void = () => {};

    constructor(
        private readonly canvas: HTMLCanvasElement,
        private readonly getScene: () => Scene | null,
    ) {
        canvas.addEventListener("mousedown", (e) => this.onMouse(e, MouseEventType.ButtonDown));
        window.addEventListener("mouseup", (e) => this.onMouse(e, MouseEventType.ButtonUp));
        window.addEventListener("mousemove", (e) => this.onMouse(e, MouseEventType.Move));
        canvas.addEventListener("wheel", this.onWheel, { passive: false });
        canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    }

    getControllerType(): CameraControllerType {
        return kCameraControllerTypes[this.getScene()?.getCameraControllerType() ?? NativeControllerType.FirstPerson];
    }
    /** Mirrors Scene::setCameraController. */
    setControllerType(type: CameraControllerType): void {
        this.getScene()?.setCameraController(kCameraControllerTypes.indexOf(type) as NativeControllerType);
    }
    getUpDirection(): UpDirection {
        return this.getScene()?.getUpDirection() ?? UpDirection.YPos;
    }
    setUpDirection(up: UpDirection): void {
        this.getScene()?.setUpDirection(up);
    }
    /** Movement speed in world units/second (Scene::setCameraSpeed). */
    setSpeed(s: number): void {
        const scene = this.getScene();
        if (scene) scene.cameraSpeed = s;
    }
    getSpeed(): number {
        return this.getScene()?.cameraSpeed ?? 1;
    }

    private isTypingTarget(t: EventTarget | null): boolean {
        const el = t as HTMLElement | null;
        return !!el && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(el.tagName);
    }
    private onMouse(e: MouseEvent, type: MouseEventType): void {
        this.getScene()?.onMouseEvent(toMouseEvent(e, type, this.canvas));
    }
    private onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const scene = this.getScene();
        if (!scene?.cameraControlsEnabled) return;
        if (!scene.onMouseEvent(toMouseEvent(e, MouseEventType.Wheel, this.canvas))) this.dollyAccum += -Math.sign(e.deltaY) * scene.cameraSpeed * 0.5;
    };
    /** Scene::onKeyEvent for a DOM key event; returns whether the scene consumed it. */
    handleKey(e: KeyboardEvent): boolean {
        if (this.isTypingTarget(e.target)) return false;
        const handled = this.getScene()?.onKeyEvent(toKeyboardEvent(e, e.type === "keydown" ? KeyboardEventType.KeyPressed : KeyboardEventType.KeyReleased)) ?? false;
        if (handled && e.type === "keydown" && /^F\d+$/.test(e.key)) e.preventDefault();
        if (handled && e.type === "keydown" && e.key === "F3") this.onViewpointAdded();
        return handled;
    }

    /** Mirrors Scene::onGamepadState. */
    onGamepadState(state: GamepadState): boolean {
        return this.getScene()?.onGamepadState(state) ?? false;
    }

    /** Scene::updateSelectedCamera's controller step, plus the wheel dolly. `now` is the rAF timestamp (ms). */
    update(scene: Scene, now: number): boolean {
        let changed = scene.updateCamera(now / 1000);
        if (this.dollyAccum !== 0) {
            const camera = scene.camera;
            const viewDir = normalize3(sub3(camera.getTarget(), camera.getPosition()));
            camera.setPosition(add3(camera.getPosition(), mul3(viewDir, this.dollyAccum)));
            camera.setTarget(add3(camera.getPosition(), viewDir));
            this.dollyAccum = 0;
            changed = true;
        }
        return changed;
    }
}
