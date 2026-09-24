/**
 * Quaternion math mirroring Falcor/Utils/Math/Quaternion.h (x, y, z, w order,
 * w = scalar part).
 */

import { float3 } from "./Vector.js";
import { float4x4 } from "./Matrix.js";

export class quatf {
    constructor(
        public x = 0,
        public y = 0,
        public z = 0,
        public w = 1,
    ) {}

    static identity(): quatf {
        return new quatf(0, 0, 0, 1);
    }
}

/** quatFromAngleAxis (angle in radians, axis need not be normalized upstream — but is expected normalized). */
export function quatFromAngleAxis(angle: number, axis: float3): quatf {
    const s = Math.sin(angle * 0.5);
    return new quatf(axis.x * s, axis.y * s, axis.z * s, Math.cos(angle * 0.5));
}

export function mulQuat(a: quatf, b: quatf): quatf {
    return new quatf(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

export function normalizeQuat(q: quatf): quatf {
    const l = Math.hypot(q.x, q.y, q.z, q.w);
    return l > 0 ? new quatf(q.x / l, q.y / l, q.z / l, q.w / l) : quatf.identity();
}

/** Rotates a vector by the quaternion. */
export function rotateVector(q: quatf, v: float3): float3 {
    // v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    const qv = new float3(q.x, q.y, q.z);
    const t = new float3(
        2 * (qv.y * v.z - qv.z * v.y),
        2 * (qv.z * v.x - qv.x * v.z),
        2 * (qv.x * v.y - qv.y * v.x),
    );
    return new float3(
        v.x + q.w * t.x + (qv.y * t.z - qv.z * t.y),
        v.y + q.w * t.y + (qv.z * t.x - qv.x * t.z),
        v.z + q.w * t.z + (qv.x * t.y - qv.y * t.x),
    );
}

/** quatFromRotationBetweenVectors (QuaternionMath.h; inputs normalized). */
export function quatFromRotationBetweenVectors(orig: float3, dest: float3): quatf {
    const cosTheta = orig.x * dest.x + orig.y * dest.y + orig.z * dest.z;
    if (cosTheta >= 1 - 1e-7) return new quatf(0, 0, 0, 1);
    if (cosTheta < -1 + 1e-7) {
        // Opposite directions: any perpendicular axis works (favor Y-ish).
        let axis = new float3(-orig.y, orig.x, 0); // cross((0,0,1), orig)
        if (axis.x * axis.x + axis.y * axis.y < 1e-7) axis = new float3(0, -orig.z, orig.y); // cross((1,0,0), orig)
        const len = Math.hypot(axis.x, axis.y, axis.z);
        return quatFromAngleAxis(Math.PI, new float3(axis.x / len, axis.y / len, axis.z / len));
    }
    const axis = new float3(
        orig.y * dest.z - orig.z * dest.y,
        orig.z * dest.x - orig.x * dest.z,
        orig.x * dest.y - orig.y * dest.x,
    );
    const s = Math.sqrt((1 + cosTheta) * 2);
    return new quatf(axis.x / s, axis.y / s, axis.z / s, s * 0.5);
}

/** matrixFromQuat (mirrors upstream 3x3 expansion, embedded in a 4x4). */
export function matrixFromQuat(q: quatf): float4x4 {
    const qxx = q.x * q.x, qyy = q.y * q.y, qzz = q.z * q.z;
    const qxz = q.x * q.z, qxy = q.x * q.y, qyz = q.y * q.z;
    const qwx = q.w * q.x, qwy = q.w * q.y, qwz = q.w * q.z;
    const m = float4x4.identity();
    m.set(0, 0, 1 - 2 * (qyy + qzz)); m.set(0, 1, 2 * (qxy - qwz)); m.set(0, 2, 2 * (qxz + qwy));
    m.set(1, 0, 2 * (qxy + qwz)); m.set(1, 1, 1 - 2 * (qxx + qzz)); m.set(1, 2, 2 * (qyz - qwx));
    m.set(2, 0, 2 * (qxz - qwy)); m.set(2, 1, 2 * (qyz + qwx)); m.set(2, 2, 1 - 2 * (qxx + qyy));
    return m;
}

export function slerp(a: quatf, b: quatf, t: number): quatf {
    let cosTheta = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    let bx = b.x, by = b.y, bz = b.z, bw = b.w;
    if (cosTheta < 0) {
        cosTheta = -cosTheta;
        bx = -bx; by = -by; bz = -bz; bw = -bw;
    }
    if (cosTheta > 0.9995) {
        return normalizeQuat(new quatf(
            a.x + (bx - a.x) * t,
            a.y + (by - a.y) * t,
            a.z + (bz - a.z) * t,
            a.w + (bw - a.w) * t,
        ));
    }
    const theta = Math.acos(cosTheta);
    const sinTheta = Math.sin(theta);
    const wa = Math.sin((1 - t) * theta) / sinTheta;
    const wb = Math.sin(t * theta) / sinTheta;
    return new quatf(wa * a.x + wb * bx, wa * a.y + wb * by, wa * a.z + wb * bz, wa * a.w + wb * bw);
}

/** Mirrors dot(quat, quat). */
export function dotQuat(a: quatf, b: quatf): number {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** Mirrors cross(quat, quat): the Hamilton product, as native defines it. */
export const crossQuat = mulQuat;

export function lengthQuat(q: quatf): number {
    return Math.sqrt(dotQuat(q, q));
}

export function conjugateQuat(q: quatf): quatf {
    return new quatf(-q.x, -q.y, -q.z, q.w);
}

export function inverseQuat(q: quatf): quatf {
    const d = dotQuat(q, q);
    return new quatf(-q.x / d, -q.y / d, -q.z / d, q.w / d);
}

/** Mirrors lerp(quat, quat, t): componentwise, not normalized. */
export function lerpQuat(a: quatf, b: quatf, t: number): quatf {
    return new quatf(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

/** Mirrors isfinite/isinf/isnan(quat): per component, x y z w. */
export const isfiniteQuat = (q: quatf): boolean[] => [q.x, q.y, q.z, q.w].map(Number.isFinite);
export const isinfQuat = (q: quatf): boolean[] => [q.x, q.y, q.z, q.w].map((v) => v === Infinity || v === -Infinity);
export const isnanQuat = (q: quatf): boolean[] => [q.x, q.y, q.z, q.w].map(Number.isNaN);

const kFloatEpsilon = 2 ** -23;

/** Mirrors pitch(quat), in radians. */
export function pitch(q: quatf): number {
    const y = 2 * (q.y * q.z + q.w * q.x);
    const x = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;
    // Singularity: avoid atan2(0, 0).
    if (Math.abs(x) < kFloatEpsilon && Math.abs(y) < kFloatEpsilon) return 2 * Math.atan2(q.x, q.w);
    return Math.atan2(y, x);
}

/** Mirrors yaw(quat), in radians. */
export function yaw(q: quatf): number {
    return Math.asin(Math.min(Math.max(-2 * (q.x * q.z - q.w * q.y), -1), 1));
}

/** Mirrors roll(quat), in radians. */
export function roll(q: quatf): number {
    return Math.atan2(2 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z);
}

/** Mirrors eulerAngles(quat): (pitch, yaw, roll) in radians. */
export function eulerAngles(q: quatf): float3 {
    return new float3(pitch(q), yaw(q), roll(q));
}

/** Mirrors math::quatFromEulerAngles (pitch, yaw, roll in radians). */
export function quatFromEulerAngles(e: float3): quatf {
    const [cx, cy, cz] = [Math.cos(e.x * 0.5), Math.cos(e.y * 0.5), Math.cos(e.z * 0.5)];
    const [sx, sy, sz] = [Math.sin(e.x * 0.5), Math.sin(e.y * 0.5), Math.sin(e.z * 0.5)];
    return new quatf(sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz);
}

/** Mirrors math::quatFromMatrix, from the rotation in the upper-left 3x3. */
export function quatFromMatrix(m: float4x4): quatf {
    const g = (r: number, c: number) => m.get(r, c);
    const candidates = [g(0, 0) + g(1, 1) + g(2, 2), g(0, 0) - g(1, 1) - g(2, 2), g(1, 1) - g(0, 0) - g(2, 2), g(2, 2) - g(0, 0) - g(1, 1)];
    let biggestIndex = 0;
    for (let i = 1; i < 4; i++) if (candidates[i]! > candidates[biggestIndex]!) biggestIndex = i;
    const biggestVal = Math.sqrt(candidates[biggestIndex]! + 1) * 0.5;
    const mult = 0.25 / biggestVal;
    switch (biggestIndex) {
        case 0: return new quatf((g(2, 1) - g(1, 2)) * mult, (g(0, 2) - g(2, 0)) * mult, (g(1, 0) - g(0, 1)) * mult, biggestVal);
        case 1: return new quatf(biggestVal, (g(1, 0) + g(0, 1)) * mult, (g(0, 2) + g(2, 0)) * mult, (g(2, 1) - g(1, 2)) * mult);
        case 2: return new quatf((g(1, 0) + g(0, 1)) * mult, biggestVal, (g(2, 1) + g(1, 2)) * mult, (g(0, 2) - g(2, 0)) * mult);
        default: return new quatf((g(0, 2) + g(2, 0)) * mult, (g(2, 1) + g(1, 2)) * mult, biggestVal, (g(1, 0) - g(0, 1)) * mult);
    }
}

/** Mirrors math::quatFromLookAt (dir and up normalized); right-handed maps forward onto -Z. */
export function quatFromLookAt(dir: float3, up: float3, rightHanded = true): quatf {
    const cross = (a: float3, b: float3) => new float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    const c2 = rightHanded ? new float3(-dir.x, -dir.y, -dir.z) : dir;
    const r = cross(up, c2);
    const l = Math.hypot(r.x, r.y, r.z);
    const c0 = new float3(r.x / l, r.y / l, r.z / l);
    const c1 = cross(c2, c0);
    const m = float4x4.identity();
    [c0, c1, c2].forEach((c, col) => { m.set(0, col, c.x); m.set(1, col, c.y); m.set(2, col, c.z); });
    return quatFromMatrix(m);
}
