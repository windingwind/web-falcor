// Viewer camera input: translates DOM mouse/keyboard/gamepad events into the
// native controller events and drives the ported Scene/Camera controllers
// (First Person / Orbiter / 6-DoF, mirrors Scene::setCameraController).
// update() returns true when the camera moved so the caller resets accumulation.
import {
    FirstPersonCameraController,
    OrbiterCameraController,
    SixDoFCameraController,
    UpDirection,
    float2,
    add3,
    mul3,
    normalize3,
    sub3,
    length3,
    type Camera,
    type CameraController as NativeController,
    type ControllerMouseEvent,
} from "@web-falcor/falcor";

/** Mirrors Scene::CameraControllerType (dropdown spellings from Scene.cpp). */
export const kCameraControllerTypes = ["First Person", "Orbiter", "6-DOF"] as const;
export type CameraControllerType = (typeof kCameraControllerTypes)[number];
export const kUpDirectionNames = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"] as const;

export class CameraController {
    private camera: Camera | null = null;
    private controller: NativeController | null = null;
    private type: CameraControllerType = "First Person";
    private upDirection = UpDirection.YPos;
    private speed = 1; // native Scene::mCameraSpeed default
    private dollyAccum = 0; // wheel dolly for the first-person controllers (web extra; native ignores the wheel there)
    /** False while the scene's camera controls are disabled (Scene::setCameraControlsEnabled). */
    inputEnabled: () => boolean = () => true;

    constructor(private readonly canvas: HTMLCanvasElement) {
        canvas.addEventListener("mousedown", this.onMouseDown);
        window.addEventListener("mouseup", this.onMouseUp);
        window.addEventListener("mousemove", this.onMouseMove);
        canvas.addEventListener("wheel", this.onWheel, { passive: false });
        canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        window.addEventListener("keydown", this.onKey);
        window.addEventListener("keyup", this.onKey);
    }

    getControllerType(): CameraControllerType {
        return this.type;
    }
    /** Mirrors Scene::setCameraController; Orbiter keeps the current view (no scene AABB on the web). */
    setControllerType(type: CameraControllerType): void {
        this.type = type;
        this.controller = null;
        if (this.camera) this.createController(this.camera);
    }
    getUpDirection(): UpDirection {
        return this.upDirection;
    }
    setUpDirection(up: UpDirection): void {
        this.upDirection = up;
        this.controller?.setUpDirection(up);
    }
    /** Movement speed in world units/second (Scene::setCameraSpeed). */
    setSpeed(s: number): void {
        this.speed = Math.max(0.01, s);
        this.controller?.setCameraSpeed(this.speed);
    }
    getSpeed(): number {
        return this.speed;
    }

    private createController(camera: Camera): void {
        this.camera = camera;
        switch (this.type) {
            case "Orbiter": {
                const c = new OrbiterCameraController(camera);
                // Native: scene AABB center/radius with distance 3.5 radii; here the current target/distance so the view is kept.
                const center = camera.getTarget();
                const distance = Math.max(1e-3, length3(sub3(camera.getPosition(), center)));
                c.setModelParams(center, distance / 3.5, 3.5);
                this.controller = c;
                break;
            }
            case "6-DOF":
                this.controller = new SixDoFCameraController(camera);
                break;
            default:
                this.controller = new FirstPersonCameraController(camera);
        }
        this.controller.setUpDirection(this.upDirection);
        this.controller.setCameraSpeed(this.speed);
    }

    private pos(e: MouseEvent): float2 {
        const r = this.canvas.getBoundingClientRect();
        return new float2((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    }
    private button(e: MouseEvent): ControllerMouseEvent["button"] {
        return e.button === 0 ? "left" : e.button === 2 ? "right" : e.button === 1 ? "middle" : undefined;
    }
    private isTypingTarget(t: EventTarget | null): boolean {
        const el = t as HTMLElement | null;
        return !!el && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(el.tagName);
    }
    private onMouseDown = (e: MouseEvent) => {
        if (!this.inputEnabled()) return;
        const button = this.button(e);
        if (button) this.controller?.onMouseEvent({ type: "buttonDown", button, pos: this.pos(e) });
    };
    private onMouseUp = (e: MouseEvent) => {
        const button = this.button(e);
        if (button) this.controller?.onMouseEvent({ type: "buttonUp", button, pos: this.pos(e) });
    };
    private onMouseMove = (e: MouseEvent) => {
        if (!this.inputEnabled()) return;
        this.controller?.onMouseEvent({ type: "move", pos: this.pos(e) });
    };
    private onWheel = (e: WheelEvent) => {
        e.preventDefault();
        if (!this.inputEnabled()) return;
        const up = -Math.sign(e.deltaY); // native wheelDelta.y: +1 = scroll up
        const handled = this.controller?.onMouseEvent({ type: "wheel", pos: this.pos(e), wheelDelta: new float2(0, up) }) ?? false;
        if (!handled) this.dollyAccum += up * this.speed * 0.5;
    };
    private onKey = (e: KeyboardEvent) => {
        if (this.isTypingTarget(e.target)) return;
        // Releases still go through, so no key stays held while controls are off.
        if (e.type === "keydown" && !this.inputEnabled()) return;
        this.controller?.onKeyEvent({ type: e.type === "keydown" ? "keyPressed" : "keyReleased", key: e.key.toLowerCase(), shift: e.shiftKey, ctrl: e.ctrlKey });
    };
    private pollGamepad(): void {
        const pad = typeof navigator.getGamepads === "function" ? navigator.getGamepads().find((g) => g && g.connected) : null;
        if (!pad) return;
        const a = pad.axes;
        this.controller?.onGamepadState({
            leftX: a[0] ?? 0,
            leftY: -(a[1] ?? 0),
            rightX: a[2] ?? 0,
            rightY: -(a[3] ?? 0),
            leftTrigger: pad.buttons[6]?.value ?? 0,
            rightTrigger: pad.buttons[7]?.value ?? 0,
        });
    }

    /** Applies pending input to `camera`. `now` is the rAF timestamp (ms). */
    update(camera: Camera, now: number): boolean {
        if (camera !== this.camera || !this.controller) this.createController(camera);
        this.pollGamepad();
        let changed = this.controller!.update(now / 1000);
        if (this.dollyAccum !== 0) {
            const viewDir = normalize3(sub3(camera.getTarget(), camera.getPosition()));
            camera.setPosition(add3(camera.getPosition(), mul3(viewDir, this.dollyAccum)));
            camera.setTarget(add3(camera.getPosition(), viewDir));
            this.dollyAccum = 0;
            changed = true;
        }
        return changed;
    }
}
