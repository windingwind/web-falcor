/**
 * Camera controllers mirroring Scene/Camera/CameraController: Orbiter
 * (arcball around a point) and FirstPerson/SixDoF (WASD + mouse-look).
 * Web divergences (docs §9): normalized input events replace the native
 * Input types, gamepad state is not wired, and update() takes the current
 * time in seconds (native reads an internal CpuTimer).
 */

import type { Camera } from "./Camera.js";
import { float2, float3, add3, sub3, mul3, cross, normalize3, min3, max3 } from "../../Utils/Math/Vector.js";
import { float4x4 } from "../../Utils/Math/Matrix.js";
import { matrixFromQuat, quatFromAngleAxis, quatFromRotationBetweenVectors } from "../../Utils/Math/Quaternion.js";

export enum UpDirection { XPos, XNeg, YPos, YNeg, ZPos, ZNeg }

/** Normalized mouse event: pos in [0,1]² (y down), like native MouseEvent. */
export interface ControllerMouseEvent {
    type: "buttonDown" | "buttonUp" | "move" | "wheel";
    button?: "left" | "middle" | "right";
    pos: float2;
    wheelDelta?: float2;
}

/** Mirrors GamepadState (axes in [-1,1], triggers in [0,1]). */
export interface ControllerGamepadState {
    leftX: number;
    leftY: number;
    rightX: number;
    rightY: number;
    leftTrigger: number;
    rightTrigger: number;
}

export interface ControllerKeyEvent {
    type: "keyPressed" | "keyReleased";
    /** Lower-case key letter (w/a/s/d/q/e). */
    key: string;
    shift?: boolean;
    ctrl?: boolean;
}

/** mul(v, M): native row-vector transform (3x3 part). */
function mulVecMat(v: float3, m: float4x4): float3 {
    return new float3(
        v.x * m.get(0, 0) + v.y * m.get(1, 0) + v.z * m.get(2, 0),
        v.x * m.get(0, 1) + v.y * m.get(1, 1) + v.z * m.get(2, 1),
        v.x * m.get(0, 2) + v.y * m.get(1, 2) + v.z * m.get(2, 2),
    );
}

/** FalcorMath.h project2DCrdToUnitSphere (xy in [-1,1]²). */
function project2DCrdToUnitSphere(xy: float2): float3 {
    const d = xy.x * xy.x + xy.y * xy.y;
    if (d < 1) return new float3(xy.x, xy.y, Math.sqrt(1 - d));
    const len = Math.sqrt(d);
    return new float3(xy.x / len, xy.y / len, 0);
}

const kGamepadDeadZone = 0.1;
const kGamepadPowerCurve = 1.2;
const kGamepadRotationSpeed = 2.5;

function applyDeadZone1(v: number, deadZone: number): number {
    return (v * Math.max(v - deadZone, 0)) / (1 - deadZone);
}

function applyDeadZone2(v: float2, deadZone: number): float2 {
    const scale = Math.max(Math.hypot(v.x, v.y) - deadZone, 0) / (1 - deadZone);
    return new float2(v.x * scale, v.y * scale);
}

function applyPowerCurve1(v: number, power: number): number {
    return Math.pow(Math.abs(v), power) * (v < 0 ? -1 : 1);
}

function applyPowerCurve2(v: float2, power: number): float2 {
    return new float2(applyPowerCurve1(v.x, power), applyPowerCurve1(v.y, power));
}

/** [0,1] screen position to [-1,1] with y flipped (native convertCamPosRange). */
function convertCamPosRange(pos: float2): float2 {
    return new float2(pos.x * 2 - 1, pos.y * -2 + 1);
}

export abstract class CameraController {
    protected upDirection = UpDirection.YPos;
    protected speed = 1;
    protected boundsMin: float3 | null = null;
    protected boundsMax: float3 | null = null;

    constructor(protected readonly camera: Camera) {}

    onMouseEvent(_ev: ControllerMouseEvent): boolean { return false; }
    onKeyEvent(_ev: ControllerKeyEvent): boolean { return false; }
    onGamepadState(_state: ControllerGamepadState): boolean { return false; }

    /** Applies pending input to the camera; returns whether it changed. */
    abstract update(nowSeconds?: number): boolean;

    setUpDirection(up: UpDirection): void { this.upDirection = up; }
    getUpDirection(): UpDirection { return this.upDirection; }
    setCameraSpeed(speed: number): void { this.speed = speed; }
    getCameraSpeed(): number { return this.speed; }
    resetInputState(): void {}

    /** Mirrors setCameraBounds (position clamped to the box interior). */
    setCameraBounds(minPoint: float3, maxPoint: float3): void {
        this.boundsMin = minPoint.clone();
        this.boundsMax = maxPoint.clone();
    }

    protected getUpVector(): float3 {
        const index = this.upDirection as number;
        const up = new float3(0, 0, 0);
        const v = index % 2 === 0 ? 1 : -1;
        if (index < 2) up.x = v;
        else if (index < 4) up.y = v;
        else up.z = v;
        return up;
    }
}

/** Mirrors OrbiterCameraController: arcball orbit around a model center. */
export class OrbiterCameraController extends CameraController {
    private modelCenter = new float3(0, 0, 0);
    private modelRadius = 1;
    private cameraDistance = 1;
    private rotation = float4x4.identity();
    private lastVector = new float3(0, 0, 0);
    private isLeftButtonDown = false;
    private dirty = false;

    /** Mirrors setModelParams(center, radius, distanceInRadius). */
    setModelParams(center: float3, radius: number, distanceInRadius: number): void {
        this.modelCenter = center.clone();
        this.modelRadius = radius;
        this.cameraDistance = distanceInRadius;
        this.rotation = float4x4.identity();
        this.dirty = true;
    }

    override onMouseEvent(ev: ControllerMouseEvent): boolean {
        switch (ev.type) {
            case "wheel":
                this.cameraDistance -= (ev.wheelDelta?.y ?? 0) * 0.2;
                this.dirty = true;
                return true;
            case "buttonDown":
                if (ev.button === "left") {
                    this.lastVector = project2DCrdToUnitSphere(convertCamPosRange(ev.pos));
                    this.isLeftButtonDown = true;
                    return true;
                }
                return false;
            case "buttonUp":
                if (ev.button === "left") {
                    const handled = this.isLeftButtonDown;
                    this.isLeftButtonDown = false;
                    return handled;
                }
                return false;
            case "move":
                if (this.isLeftButtonDown) {
                    const curVec = project2DCrdToUnitSphere(convertCamPosRange(ev.pos));
                    const q = quatFromRotationBetweenVectors(this.lastVector, curVec);
                    this.rotation = mulMat3(matrixFromQuat(q), this.rotation);
                    this.dirty = true;
                    this.lastVector = curVec;
                    return true;
                }
                return false;
        }
        return false;
    }

    override update(): boolean {
        if (!this.dirty) return false;
        this.dirty = false;
        this.camera.setTarget(this.modelCenter);
        const offset = mul3(mulVecMat(new float3(0, 0, 1), this.rotation), this.modelRadius * this.cameraDistance);
        this.camera.setPosition(add3(this.modelCenter, offset));
        this.camera.setUpVector(mulVecMat(new float3(0, 1, 0), this.rotation));
        return true;
    }

    override resetInputState(): void {
        this.isLeftButtonDown = false;
        this.dirty = false;
    }
}

/** 4x4 product restricted to the rotation block (native float3x3 mul). */
function mulMat3(a: float4x4, b: float4x4): float4x4 {
    const out = float4x4.identity();
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            let sum = 0;
            for (let k = 0; k < 3; k++) sum += a.get(r, k) * b.get(k, c);
            out.set(r, c, sum);
        }
    }
    return out;
}

enum Direction { Forward, Backward, Right, Left, Up, Down }

/** Mirrors FirstPersonCameraControllerCommon<b6DoF>. */
abstract class FirstPersonCameraControllerCommon extends CameraController {
    private isLeftButtonDown = false;
    private isRightButtonDown = false;
    private shouldRotate = false;
    private lastMousePos = new float2(0, 0);
    private mouseDelta = new float2(0, 0);
    private movement = new Set<Direction>();
    private speedModifier = 1;
    private lastTime: number | null = null;
    private gamepadPresent = false;
    private gamepadLeftStick = new float2(0, 0);
    private gamepadRightStick = new float2(0, 0);
    private gamepadLeftTrigger = 0;
    private gamepadRightTrigger = 0;

    protected constructor(camera: Camera, private readonly sixDoF: boolean) {
        super(camera);
    }

    override onKeyEvent(ev: ControllerKeyEvent): boolean {
        const pressed = ev.type === "keyPressed";
        const dir = ({ w: Direction.Forward, s: Direction.Backward, a: Direction.Right, d: Direction.Left, q: Direction.Down, e: Direction.Up } as Record<string, Direction>)[ev.key];
        this.speedModifier = ev.ctrl ? 0.25 : ev.shift ? 10 : 1;
        if (dir === undefined) return false;
        if (pressed) this.movement.add(dir);
        else this.movement.delete(dir);
        return true;
    }

    override onGamepadState(state: ControllerGamepadState): boolean {
        this.gamepadPresent = true;
        this.gamepadLeftStick = applyPowerCurve2(applyDeadZone2(new float2(state.leftX, state.leftY), kGamepadDeadZone), kGamepadPowerCurve);
        this.gamepadRightStick = applyPowerCurve2(applyDeadZone2(new float2(state.rightX, state.rightY), kGamepadDeadZone), kGamepadPowerCurve);
        this.gamepadLeftTrigger = applyPowerCurve1(applyDeadZone1(state.leftTrigger, kGamepadDeadZone), kGamepadPowerCurve);
        this.gamepadRightTrigger = applyPowerCurve1(applyDeadZone1(state.rightTrigger, kGamepadDeadZone), kGamepadPowerCurve);
        return (
            Math.hypot(this.gamepadLeftStick.x, this.gamepadLeftStick.y) > 0 ||
            Math.hypot(this.gamepadRightStick.x, this.gamepadRightStick.y) > 0 ||
            this.gamepadLeftTrigger > 0 ||
            this.gamepadRightTrigger > 0
        );
    }

    override onMouseEvent(ev: ControllerMouseEvent): boolean {
        switch (ev.type) {
            case "buttonDown":
                if (ev.button === "left" || ev.button === "right") {
                    this.lastMousePos = new float2(ev.pos.x, ev.pos.y);
                    if (ev.button === "left") this.isLeftButtonDown = true;
                    else this.isRightButtonDown = true;
                    return true;
                }
                return false;
            case "buttonUp":
                if (ev.button === "left") {
                    const handled = this.isLeftButtonDown;
                    this.isLeftButtonDown = false;
                    return handled;
                }
                if (ev.button === "right") {
                    const handled = this.isRightButtonDown;
                    this.isRightButtonDown = false;
                    return handled;
                }
                return false;
            case "move":
                if (this.isLeftButtonDown || this.isRightButtonDown) {
                    this.mouseDelta = new float2(ev.pos.x - this.lastMousePos.x, ev.pos.y - this.lastMousePos.y);
                    this.lastMousePos = new float2(ev.pos.x, ev.pos.y);
                    this.shouldRotate = true;
                    return true;
                }
                return false;
        }
        return false;
    }

    override update(nowSeconds = performance.now() / 1000): boolean {
        const elapsed = Math.min(0.1, this.lastTime === null ? 0 : nowSeconds - this.lastTime);
        this.lastTime = nowSeconds;

        let dirty = false;
        const anyGamepadMovement =
            this.gamepadPresent && (Math.hypot(this.gamepadLeftStick.x, this.gamepadLeftStick.y) > 0 || this.gamepadLeftTrigger > 0 || this.gamepadRightTrigger > 0);
        const anyGamepadRotation = this.gamepadPresent && Math.hypot(this.gamepadRightStick.x, this.gamepadRightStick.y) > 0;

        if (this.shouldRotate || anyGamepadRotation) {
            const camPos = this.camera.getPosition();
            let camUp = this.sixDoF ? this.camera.getUpVector() : this.getUpVector();
            let viewDir = normalize3(sub3(this.camera.getTarget(), camPos));

            if (this.isLeftButtonDown || anyGamepadRotation) {
                const sideway = cross(viewDir, normalize3(camUp));
                const mouseRotation = this.isLeftButtonDown
                    ? new float2(this.mouseDelta.x * this.speedModifier, this.mouseDelta.y * this.speedModifier)
                    : new float2(0, 0);
                const gamepadRotation = anyGamepadRotation
                    ? new float2(this.gamepadRightStick.x * kGamepadRotationSpeed * elapsed, this.gamepadRightStick.y * kGamepadRotationSpeed * elapsed)
                    : new float2(0, 0);
                const rotation = new float2(mouseRotation.x + gamepadRotation.x, mouseRotation.y + gamepadRotation.y);

                const rotY = matrixFromQuat(quatFromAngleAxis(rotation.y, sideway));
                viewDir = mulVecMat(viewDir, rotY);
                camUp = mulVecMat(camUp, rotY);

                const rotX = matrixFromQuat(quatFromAngleAxis(rotation.x, camUp));
                viewDir = mulVecMat(viewDir, rotX);

                this.camera.setTarget(add3(camPos, viewDir));
                this.camera.setUpVector(camUp);
                dirty = true;
            }

            if (this.sixDoF && this.isRightButtonDown) {
                const rot = matrixFromQuat(quatFromAngleAxis(this.mouseDelta.x * this.speedModifier, viewDir));
                this.camera.setUpVector(mulVecMat(camUp, rot));
                dirty = true;
            }

            this.shouldRotate = false;
        }

        if ((this.movement.size > 0 || anyGamepadMovement) && elapsed > 0) {
            const movement = new float3(0, 0, 0);
            if (this.movement.has(Direction.Forward)) movement.z += 1;
            if (this.movement.has(Direction.Backward)) movement.z -= 1;
            if (this.movement.has(Direction.Left)) movement.x += 1;
            if (this.movement.has(Direction.Right)) movement.x -= 1;
            if (this.movement.has(Direction.Up)) movement.y += 1;
            if (this.movement.has(Direction.Down)) movement.y -= 1;

            if (anyGamepadMovement) {
                movement.x += this.gamepadLeftStick.x;
                movement.z -= this.gamepadLeftStick.y;
                movement.y -= this.gamepadLeftTrigger;
                movement.y += this.gamepadRightTrigger;
            }

            let camPos = this.camera.getPosition();
            const camUp = this.camera.getUpVector();
            const viewDir = normalize3(sub3(this.camera.getTarget(), camPos));
            const sideway = cross(viewDir, normalize3(camUp));

            const curMove = this.speedModifier * this.speed * elapsed;
            camPos = add3(camPos, mul3(viewDir, movement.z * curMove));
            camPos = add3(camPos, mul3(sideway, movement.x * curMove));
            camPos = add3(camPos, mul3(camUp, movement.y * curMove));
            if (this.boundsMin && this.boundsMax) camPos = min3(max3(camPos, this.boundsMin), this.boundsMax);

            this.camera.setPosition(camPos);
            this.camera.setTarget(add3(camPos, viewDir));
            dirty = true;
        }

        // Will be set true in the next onGamepadState call (native pattern).
        this.gamepadPresent = false;

        return dirty;
    }

    override resetInputState(): void {
        this.isLeftButtonDown = false;
        this.isRightButtonDown = false;
        this.shouldRotate = false;
        this.movement.clear();
        this.gamepadLeftStick = new float2(0, 0);
        this.gamepadRightStick = new float2(0, 0);
        this.gamepadLeftTrigger = 0;
        this.gamepadRightTrigger = 0;
    }
}

/** Y-locked WASD + mouse-look controller. */
export class FirstPersonCameraController extends FirstPersonCameraControllerCommon {
    constructor(camera: Camera) { super(camera, false); }
}

/** Free controller: camera-relative up, right-drag rolls around the view axis. */
export class SixDoFCameraController extends FirstPersonCameraControllerCommon {
    constructor(camera: Camera) { super(camera, true); }
}
