/**
 * Camera mirroring Falcor/Scene/Camera/Camera.h.
 *
 * Focal length is in millimeters against a 35mm-film 24mm frame height
 * (Falcor convention); depth range is [0,1], right-handed view space.
 */

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
}

export class Camera {
    name: string;
    private position = new float3(0, 0, 5);
    private target = new float3(0, 0, 0);
    private up = new float3(0, 1, 0);
    private focalLength = 21.0; // Falcor default
    private frameHeight = 24.0;
    private aspectRatio = 1.7777;
    private nearZ = 0.1;
    private farZ = 1000;
    private jitter = new float2(0, 0);
    private dirty = true;
    private data: CameraData | null = null;
    private jitterPattern: { generator: import("../../Utils/SampleGenerators/CPUSampleGenerator.js").CPUSampleGenerator | null; scale: float2 } = {
        generator: null,
        scale: new float2(0, 0),
    };

    constructor(name = "Camera") {
        this.name = name;
    }

    /** Mirrors Camera::setPatternGenerator (jitter applied each beginFrame). */
    setPatternGenerator(generator: typeof this.jitterPattern.generator, scale: float2): void {
        this.jitterPattern = { generator, scale };
        if (!generator) this.setJitter(0, 0);
    }

    /** Mirrors the jitter part of Camera::beginFrame (called once per frame). */
    beginFrame(): void {
        if (this.jitterPattern.generator) {
            const j = this.jitterPattern.generator.next();
            this.setJitter(Math.fround(j.x * this.jitterPattern.scale.x), Math.fround(j.y * this.jitterPattern.scale.y));
        }
    }

    setPosition(p: float3): void { this.position = p.clone(); this.dirty = true; }
    getPosition(): float3 { return this.position.clone(); }
    setTarget(t: float3): void { this.target = t.clone(); this.dirty = true; }
    getTarget(): float3 { return this.target.clone(); }
    setUpVector(u: float3): void { this.up = u.clone(); this.dirty = true; }
    setFocalLength(mm: number): void { this.focalLength = mm; this.dirty = true; }
    getFocalLength(): number { return this.focalLength; }
    setAspectRatio(ratio: number): void { this.aspectRatio = ratio; this.dirty = true; }
    getAspectRatio(): number { return this.aspectRatio; }
    setDepthRange(nearZ: number, farZ: number): void { this.nearZ = nearZ; this.farZ = farZ; this.dirty = true; }
    setJitter(x: number, y: number): void { this.jitter = new float2(x, y); this.dirty = true; }

    getFovY(): number {
        return focalLengthToFovY(this.focalLength, this.frameHeight);
    }

    /** Mirrors Camera::calculateCameraParameters + getData. */
    getData(): CameraData {
        if (this.dirty || !this.data) {
            const viewMat = matrixFromLookAt(this.position, this.target, this.up);
            let projMat = perspective(this.getFovY(), this.aspectRatio, this.nearZ, this.farZ);
            // Camera jitter offsets clip-space positions (mirrors Camera::calculateCameraParameters).
            if (this.jitter.x !== 0 || this.jitter.y !== 0) {
                projMat = projMat.clone();
                projMat.set(0, 2, projMat.get(0, 2) + 2 * this.jitter.x);
                projMat.set(1, 2, projMat.get(1, 2) - 2 * this.jitter.y);
            }
            const viewProjMat = mulMat(projMat, viewMat);

            // Ray-gen basis (mirrors upstream cameraU/V/W computation).
            const invView = inverse(viewMat);
            const right = new float3(invView.get(0, 0), invView.get(1, 0), invView.get(2, 0));
            const upV = new float3(invView.get(0, 1), invView.get(1, 1), invView.get(2, 1));
            const fwd = new float3(-invView.get(0, 2), -invView.get(1, 2), -invView.get(2, 2));
            const tanHalfFovY = Math.tan(0.5 * this.getFovY());

            this.data = {
                viewMat,
                projMat,
                viewProjMat,
                invViewProj: inverse(viewProjMat),
                posW: this.position.clone(),
                focalLength: this.focalLength,
                up: this.up.clone(),
                aspectRatio: this.aspectRatio,
                target: this.target.clone(),
                nearZ: this.nearZ,
                cameraU: new float3(right.x * tanHalfFovY * this.aspectRatio, right.y * tanHalfFovY * this.aspectRatio, right.z * tanHalfFovY * this.aspectRatio),
                farZ: this.farZ,
                cameraV: new float3(upV.x * tanHalfFovY, upV.y * tanHalfFovY, upV.z * tanHalfFovY),
                jitterX: this.jitter.x,
                cameraW: fwd,
                jitterY: this.jitter.y,
                frameHeight: this.frameHeight,
                frameWidth: this.frameHeight * this.aspectRatio,
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
