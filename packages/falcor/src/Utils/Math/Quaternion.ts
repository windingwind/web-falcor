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
