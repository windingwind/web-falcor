/**
 * Camera mirroring Falcor/Scene/Camera/Camera.h.
 *
 * Focal length is in millimeters against a 35mm-film 24mm frame height
 * (Falcor convention); depth range is [0,1], right-handed view space.
 */

import { ScriptWriter } from "../../Utils/Scripting/ScriptWriter.js";
import { float2, float3 } from "../../Utils/Math/Vector.js";
import {
    float4x4,
    inverse,
    matrixFromLookAt,
    mulMat,
    perspective,
} from "../../Utils/Math/Matrix.js";

/** Mirrors focalLengthToFovY (Utils/Math/FalcorMath.h). */
export function focalLengthToFovY(focalLength: number, frameHeight: number): number {
    return 2 * Math.atan(0.5 * frameHeight / focalLength);
}

export function fovYToFocalLength(fovY: number, frameHeight: number): number {
    return (0.5 * frameHeight) / Math.tan(0.5 * fovY);
}

/** GPU-facing camera parameters (mirrors Scene/Camera/CameraData.slang). */
export interface CameraData {
    viewMat: float4x4;
    projMat: float4x4;
    viewProjMat: float4x4;
    viewProjMatNoJitter: float4x4;
    prevViewProjMatNoJitter: float4x4;
    invViewProj: float4x4;
    posW: float3;
    focalLength: number;
    up: float3;
    aspectRatio: number;
    target: float3;
    nearZ: number;
    cameraU: float3;
    farZ: number;
    cameraV: float3;
    jitterX: number;
    cameraW: float3;
    jitterY: number;
    frameHeight: number;
    frameWidth: number;
    focalDistance: number;
    apertureRadius: number;
    shutterSpeed: number;
    ISOSpeed: number;
}

/** A float3 from a JS vector or a Python-side proxy with x/y/z. */
function asFloat3(v: { x: number; y: number; z: number }): float3 {
    return new float3(Number(v.x), Number(v.y), Number(v.z));
}

export class Camera {
    name: string;
    private _position = new float3(0, 0, 5);
    private _target = new float3(0, 0, 0);
    private _up = new float3(0, 1, 0);
    private _focalLength = 21.0; // Falcor default
    private _focalDistance = 10000.0;
    private _apertureRadius = 0.0;
    private _shutterSpeed = 0.004;
    private _ISOSpeed = 100.0;
    private _frameHeight = 24.0;
    private _frameWidth = 0;
    /** Native mPreserveHeight: which film dimension stays fixed when the aspect ratio changes. */
    private preserveHeight = true;
    private _aspectRatio = 1.7777;
    private nearZ = 0.1;
    private farZ = 1000;
    private jitter = new float2(0, 0);
    private dirty = true;
    private data: CameraData | null = null;
    private jitterPattern: { generator: import("../../Utils/SampleGenerators/CPUSampleGenerator.js").CPUSampleGenerator | null; scale: float2 } = {
        generator: null,
        scale: new float2(0, 0),
    };
    private prevViewProjMatNoJitter: float4x4 | null = null;
    private lastFrameViewProjMatNoJitter: float4x4 | null = null;

    constructor(name = "Camera") {
        this.name = name;
    }

    // Python properties of native's Camera binding (they mark the camera dirty like the setters).
    get position(): float3 { return this.getPosition(); }
    set position(v: float3) { this.setPosition(asFloat3(v)); }
    get target(): float3 { return this.getTarget(); }
    set target(v: float3) { this.setTarget(asFloat3(v)); }
    get up(): float3 { return this.getUpVector(); }
    set up(v: float3) { this.setUpVector(asFloat3(v)); }
    get focalLength(): number { return this._focalLength; }
    set focalLength(v: number) { this.setFocalLength(v); }
    get frameHeight(): number { return this.getFrameHeight(); }
    set frameHeight(v: number) { this.setFrameHeight(v); }
    get frameWidth(): number { return this.getFrameWidth(); }
    set frameWidth(v: number) { this.setFrameWidth(v); }
    get focalDistance(): number { return this._focalDistance; }
    set focalDistance(v: number) { this.setFocalDistance(v); }
    get apertureRadius(): number { return this._apertureRadius; }
    set apertureRadius(v: number) { this.setApertureRadius(v); }
    get shutterSpeed(): number { return this._shutterSpeed; }
    set shutterSpeed(v: number) { this.setShutterSpeed(v); }
    get ISOSpeed(): number { return this._ISOSpeed; }
    set ISOSpeed(v: number) { this.setISOSpeed(v); }
    get aspectRatio(): number { return this._aspectRatio; }
    set aspectRatio(v: number) { this.setAspectRatio(v); }
    get nearPlane(): number { return this.nearZ; }
    set nearPlane(v: number) { this.setDepthRange(v, this.farZ); }
    get farPlane(): number { return this.farZ; }
    set farPlane(v: number) { this.setDepthRange(this.nearZ, v); }
    /** Mirrors Animatable::setIsAnimated: whether the scene's animation drives this camera. */
    animated = true;
    /** Mirrors Animatable::hasAnimation (set by the scene for the node-bound camera). */
    hasAnimation = false;

    /** Mirrors Camera::setPatternGenerator (jitter applied each beginFrame). */
    setPatternGenerator(generator: typeof this.jitterPattern.generator, scale: float2): void {
        this.jitterPattern = { generator, scale };
        if (!generator) this.setJitter(0, 0);
    }

    /** Mirrors Camera::beginFrame: jitter pattern advance + prev-matrix roll.
     *  prev must be the matrix USED last frame (native mPrevData), not a
     *  recompute — position/target may have changed since the last frame. */
    beginFrame(): void {
        if (this.jitterPattern.generator) {
            const j = this.jitterPattern.generator.next();
            this.setJitter(Math.fround(j.x * this.jitterPattern.scale.x), Math.fround(j.y * this.jitterPattern.scale.y));
        }
        const cur = this.getData().viewProjMatNoJitter.clone();
        this.prevViewProjMatNoJitter = this.lastFrameViewProjMatNoJitter ?? cur;
        this.lastFrameViewProjMatNoJitter = cur;
        this.dirty = true; // data.prevViewProjMatNoJitter must be rebuilt
    }

    setPosition(p: float3): void { this._position = p.clone(); this.dirty = true; }
    getPosition(): float3 { return this._position.clone(); }
    setTarget(t: float3): void { this._target = t.clone(); this.dirty = true; }
    getTarget(): float3 { return this._target.clone(); }
    setUpVector(u: float3): void { this._up = u.clone(); this.dirty = true; }
    getUpVector(): float3 { return this._up.clone(); }
    /** Mirrors Camera::getScript: the pose as script lines on `cameraVar`. */
    getScript(cameraVar: string): string {
        return (
            ScriptWriter.makeSetProperty(cameraVar, "position", this._position) +
            ScriptWriter.makeSetProperty(cameraVar, "target", this._target) +
            ScriptWriter.makeSetProperty(cameraVar, "up", this._up)
        );
    }

    setFocalLength(mm: number): void { this._focalLength = mm; this.dirty = true; }
    getFocalLength(): number { return this._focalLength; }
    /** Mirrors Camera::setFrameHeight (film-back height in mm; USD cameras author it). */
    setFrameHeight(mm: number): void { this._frameHeight = mm; this.preserveHeight = true; this.dirty = true; }
    getFrameHeight(): number { return this.preserveHeight ? this._frameHeight : this._frameWidth / this._aspectRatio; }
    /** Mirrors Camera::setFrameWidth: the width stays fixed and the height follows the aspect ratio. */
    setFrameWidth(mm: number): void { this._frameWidth = mm; this.preserveHeight = false; this.dirty = true; }
    getFrameWidth(): number { return this.preserveHeight ? this._frameHeight * this._aspectRatio : this._frameWidth; }
    setFocalDistance(d: number): void { this._focalDistance = d; this.dirty = true; }
    getFocalDistance(): number { return this._focalDistance; }
    setApertureRadius(r: number): void { this._apertureRadius = r; this.dirty = true; }
    getApertureRadius(): number { return this._apertureRadius; }
    /** Mirrors Camera::setShutterSpeed (seconds; physical-exposure metadata). */
    setShutterSpeed(s: number): void { this._shutterSpeed = s; this.dirty = true; }
    getShutterSpeed(): number { return this._shutterSpeed; }
    /** Mirrors Camera::setISOSpeed. */
    setISOSpeed(iso: number): void { this._ISOSpeed = iso; this.dirty = true; }
    getISOSpeed(): number { return this._ISOSpeed; }
    setAspectRatio(ratio: number): void { this._aspectRatio = ratio; this.dirty = true; }
    getAspectRatio(): number { return this._aspectRatio; }
    setDepthRange(nearZ: number, farZ: number): void { this.nearZ = nearZ; this.farZ = farZ; this.dirty = true; }
    getNearPlane(): number { return this.nearZ; }
    getFarPlane(): number { return this.farZ; }
    setJitter(x: number, y: number): void { this.jitter = new float2(x, y); this.dirty = true; }

    getFovY(): number {
        return focalLengthToFovY(this._focalLength, this.getFrameHeight());
    }

    /** Mirrors Camera::calculateCameraParameters + getData. */
    getData(): CameraData {
        if (this.dirty || !this.data) {
            const viewMat = matrixFromLookAt(this._position, this._target, this._up);
            const projMatNoJitter = perspective(this.getFovY(), this._aspectRatio, this.nearZ, this.farZ);
            let projMat = projMatNoJitter;
            // Camera jitter offsets clip-space positions (mirrors Camera::calculateCameraParameters).
            if (this.jitter.x !== 0 || this.jitter.y !== 0) {
                projMat = projMatNoJitter.clone();
                projMat.set(0, 2, projMat.get(0, 2) + 2 * this.jitter.x);
                projMat.set(1, 2, projMat.get(1, 2) - 2 * this.jitter.y);
            }
            const viewProjMat = mulMat(projMat, viewMat);
            const viewProjMatNoJitter = mulMat(projMatNoJitter, viewMat);

            // Ray-gen basis (mirrors upstream cameraU/V/W computation).
            const invView = inverse(viewMat);
            const right = new float3(invView.get(0, 0), invView.get(1, 0), invView.get(2, 0));
            const upV = new float3(invView.get(0, 1), invView.get(1, 1), invView.get(2, 1));
            const fwd = new float3(-invView.get(0, 2), -invView.get(1, 2), -invView.get(2, 2));
            // U/V/W lengths carry the focal distance (upstream convention: the
            // thin-lens focal plane sits at |cameraW|).
            const tanHalfFovY = Math.tan(0.5 * this.getFovY());
            const ulen = this._focalDistance * tanHalfFovY * this._aspectRatio;
            const vlen = this._focalDistance * tanHalfFovY;

            this.data = {
                viewMat,
                projMat,
                viewProjMat,
                viewProjMatNoJitter,
                prevViewProjMatNoJitter: this.prevViewProjMatNoJitter ?? viewProjMatNoJitter,
                invViewProj: inverse(viewProjMat),
                posW: this._position.clone(),
                focalLength: this._focalLength,
                up: this._up.clone(),
                aspectRatio: this._aspectRatio,
                target: this._target.clone(),
                nearZ: this.nearZ,
                cameraU: new float3(right.x * ulen, right.y * ulen, right.z * ulen),
                farZ: this.farZ,
                cameraV: new float3(upV.x * vlen, upV.y * vlen, upV.z * vlen),
                jitterX: this.jitter.x,
                cameraW: new float3(fwd.x * this._focalDistance, fwd.y * this._focalDistance, fwd.z * this._focalDistance),
                jitterY: this.jitter.y,
                frameHeight: this.getFrameHeight(),
                frameWidth: this.getFrameWidth(),
                focalDistance: this._focalDistance,
                apertureRadius: this._apertureRadius,
                shutterSpeed: this._shutterSpeed,
                ISOSpeed: this._ISOSpeed,
            };
            this.dirty = false;
        }
        return this.data;
    }

    getViewMatrix(): float4x4 {
        return this.getData().viewMat;
    }
    getProjMatrix(): float4x4 {
        return this.getData().projMat;
    }
    getViewProjMatrix(): float4x4 {
        return this.getData().viewProjMat;
    }
}
