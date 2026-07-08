// First-person controller (mirrors Falcor's FirstPersonCameraController): left-drag
// looks, WASD/QE move, Shift/Ctrl scale speed, wheel dollies. update() returns true
// when the camera moved so the caller can reset path-tracer accumulation.
import { type Camera, float3, add3, sub3, mul3, cross, dot3, normalize3, length3, quatFromAngleAxis, rotateVector } from "@web-falcor/falcor";

const WORLD_UP = new float3(0, 1, 0);
const LOOK_SENSITIVITY = 0.0042; // radians per pixel
const MAX_UP_DOT = 0.99; // keep the view direction ~8° off the poles (no gimbal flip)

export class CameraController {
    private readonly keys = new Set<string>();
    private dragging = false;
    private lastX = 0;
    private lastY = 0;
    private yawAccum = 0; // pending mouse look, radians
    private pitchAccum = 0;
    private dollyAccum = 0; // pending wheel dolly, world units
    private speed = 2; // movement speed, world units/second
    private lastTime = -1;

    constructor(private readonly canvas: HTMLCanvasElement) {
        canvas.addEventListener("mousedown", this.onMouseDown);
        window.addEventListener("mouseup", this.onMouseUp);
        window.addEventListener("mousemove", this.onMouseMove);
        canvas.addEventListener("wheel", this.onWheel, { passive: false });
        canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        window.addEventListener("keydown", this.onKeyDown);
        window.addEventListener("keyup", this.onKeyUp);
    }

    /** Movement speed in world units/second. */
    setSpeed(s: number): void {
        this.speed = Math.max(0.01, s);
    }
    getSpeed(): number {
        return this.speed;
    }

    private isTypingTarget(t: EventTarget | null): boolean {
        const el = t as HTMLElement | null;
        return !!el && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(el.tagName);
    }

    private onMouseDown = (e: MouseEvent) => {
        if (e.button !== 0) return;
        this.dragging = true;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
    };
    private onMouseUp = () => {
        this.dragging = false;
    };
    private onMouseMove = (e: MouseEvent) => {
        if (!this.dragging) return;
        this.yawAccum -= (e.clientX - this.lastX) * LOOK_SENSITIVITY;
        this.pitchAccum -= (e.clientY - this.lastY) * LOOK_SENSITIVITY;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
    };
    private onWheel = (e: WheelEvent) => {
        e.preventDefault();
        this.dollyAccum += -Math.sign(e.deltaY) * this.speed * 0.5;
    };
    private onKeyDown = (e: KeyboardEvent) => {
        if (this.isTypingTarget(e.target)) return;
        this.keys.add(e.code);
    };
    private onKeyUp = (e: KeyboardEvent) => {
        this.keys.delete(e.code);
    };

    /** Applies pending input to `camera`. `now` is the rAF timestamp (ms). */
    update(camera: Camera, now: number): boolean {
        const dt = this.lastTime < 0 ? 0 : Math.min(0.1, (now - this.lastTime) / 1000);
        this.lastTime = now;

        let pos = camera.getPosition();
        const target = camera.getTarget();
        let viewDir = normalize3(sub3(target, pos));
        let changed = false;

        // Mouse look.
        if (this.yawAccum !== 0 || this.pitchAccum !== 0) {
            const right = normalize3(cross(viewDir, WORLD_UP));
            if (this.yawAccum !== 0) viewDir = rotateVector(quatFromAngleAxis(this.yawAccum, WORLD_UP), viewDir);
            if (this.pitchAccum !== 0) {
                const pitched = normalize3(rotateVector(quatFromAngleAxis(this.pitchAccum, right), viewDir));
                if (Math.abs(dot3(pitched, WORLD_UP)) < MAX_UP_DOT) viewDir = pitched;
            }
            viewDir = normalize3(viewDir);
            this.yawAccum = 0;
            this.pitchAccum = 0;
            changed = true;
        }

        // Keyboard movement, frame-rate independent.
        const right = normalize3(cross(viewDir, WORLD_UP));
        let move = new float3(0, 0, 0);
        const k = this.keys;
        if (k.has("KeyW")) move = add3(move, viewDir);
        if (k.has("KeyS")) move = sub3(move, viewDir);
        if (k.has("KeyD")) move = add3(move, right);
        if (k.has("KeyA")) move = sub3(move, right);
        if (k.has("KeyE")) move = add3(move, WORLD_UP);
        if (k.has("KeyQ")) move = sub3(move, WORLD_UP);
        const mag = length3(move);
        if (mag > 0 && dt > 0) {
            const fast = k.has("ShiftLeft") || k.has("ShiftRight");
            const slow = k.has("ControlLeft") || k.has("ControlRight");
            const speedMod = fast ? 5 : slow ? 0.2 : 1;
            pos = add3(pos, mul3(move, (this.speed * speedMod * dt) / mag));
            changed = true;
        }

        // Wheel dolly.
        if (this.dollyAccum !== 0) {
            pos = add3(pos, mul3(viewDir, this.dollyAccum));
            this.dollyAccum = 0;
            changed = true;
        }

        if (changed) {
            camera.setPosition(pos);
            camera.setTarget(add3(pos, viewDir));
            camera.setUpVector(WORLD_UP);
        }
        return changed;
    }
}
