/**
 * Matrix math mirroring Falcor/Utils/Math/Matrix.h + MatrixMath.h.
 *
 * Conventions (verified against upstream):
 * - Row-major storage: element (row, col) at data[row * 4 + col].
 * - Column vectors: transform is mul(M, v); translation lives in column 3.
 * - Right-handed; perspective/ortho map depth to [0, 1] (D3D clip space).
 */

import { float3, float4, cross, dot3, normalize3, sub3 } from "./Vector.js";

export class float4x4 {
    /** Row-major, data[row * 4 + col]. */
    constructor(public readonly data = new Float32Array(16)) {}

    static identity(): float4x4 {
        const m = new float4x4();
        m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1;
        return m;
    }
    static zeros(): float4x4 {
        return new float4x4();
    }
    static fromRows(rows: number[][]): float4x4 {
        const m = new float4x4();
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m.data[r * 4 + c] = rows[r]![c]!;
        return m;
    }

    get(row: number, col: number): number { return this.data[row * 4 + col]!; }
    set(row: number, col: number, v: number): void { this.data[row * 4 + col] = v; }
    clone(): float4x4 { return new float4x4(new Float32Array(this.data)); }

    /** Row-major float array (matches Falcor's memory layout). */
    toArray(): Float32Array { return this.data; }
    /** Column-major floats (WGSL mat4x4 default layout). */
    toColumnMajorArray(): Float32Array {
        const out = new Float32Array(16);
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[c * 4 + r] = this.data[r * 4 + c]!;
        return out;
    }
}

/** mul(A, B): matrix product (A then applied after B for column vectors). */
export function mulMat(a: float4x4, b: float4x4): float4x4 {
    const out = new float4x4();
    for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a.get(r, k) * b.get(k, c);
            out.set(r, c, sum);
        }
    }
    return out;
}

/** mul(M, v): column-vector transform. */
export function mulMatVec(m: float4x4, v: float4): float4 {
    return new float4(
        m.get(0, 0) * v.x + m.get(0, 1) * v.y + m.get(0, 2) * v.z + m.get(0, 3) * v.w,
        m.get(1, 0) * v.x + m.get(1, 1) * v.y + m.get(1, 2) * v.z + m.get(1, 3) * v.w,
        m.get(2, 0) * v.x + m.get(2, 1) * v.y + m.get(2, 2) * v.z + m.get(2, 3) * v.w,
        m.get(3, 0) * v.x + m.get(3, 1) * v.y + m.get(3, 2) * v.z + m.get(3, 3) * v.w,
    );
}

/** transformPoint: w=1, perspective divide NOT applied (mirrors Falcor transformPoint). */
export function transformPoint(m: float4x4, p: float3): float3 {
    const v = mulMatVec(m, new float4(p.x, p.y, p.z, 1));
    return new float3(v.x, v.y, v.z);
}

/** transformVector: w=0. */
export function transformVector(m: float4x4, d: float3): float3 {
    const v = mulMatVec(m, new float4(d.x, d.y, d.z, 0));
    return new float3(v.x, v.y, v.z);
}

export function transpose(m: float4x4): float4x4 {
    const out = new float4x4();
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out.set(r, c, m.get(c, r));
    return out;
}

export function determinant(m: float4x4): number {
    const a = m.data;
    const b00 = a[0]! * a[5]! - a[1]! * a[4]!;
    const b01 = a[0]! * a[6]! - a[2]! * a[4]!;
    const b02 = a[0]! * a[7]! - a[3]! * a[4]!;
    const b03 = a[1]! * a[6]! - a[2]! * a[5]!;
    const b04 = a[1]! * a[7]! - a[3]! * a[5]!;
    const b05 = a[2]! * a[7]! - a[3]! * a[6]!;
    const b06 = a[8]! * a[13]! - a[9]! * a[12]!;
    const b07 = a[8]! * a[14]! - a[10]! * a[12]!;
    const b08 = a[8]! * a[15]! - a[11]! * a[12]!;
    const b09 = a[9]! * a[14]! - a[10]! * a[13]!;
    const b10 = a[9]! * a[15]! - a[11]! * a[13]!;
    const b11 = a[10]! * a[15]! - a[11]! * a[14]!;
    return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
}

/** General 4x4 inverse (throws on singular). */
export function inverse(m: float4x4): float4x4 {
    const a = m.data;
    const b00 = a[0]! * a[5]! - a[1]! * a[4]!;
    const b01 = a[0]! * a[6]! - a[2]! * a[4]!;
    const b02 = a[0]! * a[7]! - a[3]! * a[4]!;
    const b03 = a[1]! * a[6]! - a[2]! * a[5]!;
    const b04 = a[1]! * a[7]! - a[3]! * a[5]!;
    const b05 = a[2]! * a[7]! - a[3]! * a[6]!;
    const b06 = a[8]! * a[13]! - a[9]! * a[12]!;
    const b07 = a[8]! * a[14]! - a[10]! * a[12]!;
    const b08 = a[8]! * a[15]! - a[11]! * a[12]!;
    const b09 = a[9]! * a[14]! - a[10]! * a[13]!;
    const b10 = a[9]! * a[15]! - a[11]! * a[13]!;
    const b11 = a[10]! * a[15]! - a[11]! * a[14]!;
    const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (Math.abs(det) < 1e-12) throw new Error("inverse(): singular matrix");
    const invDet = 1 / det;
    const out = new float4x4();
    const o = out.data;
    o[0] = (a[5]! * b11 - a[6]! * b10 + a[7]! * b09) * invDet;
    o[1] = (-a[1]! * b11 + a[2]! * b10 - a[3]! * b09) * invDet;
    o[2] = (a[13]! * b05 - a[14]! * b04 + a[15]! * b03) * invDet;
    o[3] = (-a[9]! * b05 + a[10]! * b04 - a[11]! * b03) * invDet;
    o[4] = (-a[4]! * b11 + a[6]! * b08 - a[7]! * b07) * invDet;
    o[5] = (a[0]! * b11 - a[2]! * b08 + a[3]! * b07) * invDet;
    o[6] = (-a[12]! * b05 + a[14]! * b02 - a[15]! * b01) * invDet;
    o[7] = (a[8]! * b05 - a[10]! * b02 + a[11]! * b01) * invDet;
    o[8] = (a[4]! * b10 - a[5]! * b08 + a[7]! * b06) * invDet;
    o[9] = (-a[0]! * b10 + a[1]! * b08 - a[3]! * b06) * invDet;
    o[10] = (a[12]! * b04 - a[13]! * b02 + a[15]! * b00) * invDet;
    o[11] = (-a[8]! * b04 + a[9]! * b02 - a[11]! * b00) * invDet;
    o[12] = (-a[4]! * b09 + a[5]! * b07 - a[6]! * b06) * invDet;
    o[13] = (a[0]! * b09 - a[1]! * b07 + a[2]! * b06) * invDet;
    o[14] = (-a[12]! * b03 + a[13]! * b01 - a[14]! * b00) * invDet;
    o[15] = (a[8]! * b03 - a[9]! * b01 + a[10]! * b00) * invDet;
    return out;
}

/** matrixFromTranslation (translation in column 3). */
export function matrixFromTranslation(t: float3): float4x4 {
    const m = float4x4.identity();
    m.set(0, 3, t.x);
    m.set(1, 3, t.y);
    m.set(2, 3, t.z);
    return m;
}

export function matrixFromScaling(s: float3): float4x4 {
    const m = float4x4.identity();
    m.set(0, 0, s.x);
    m.set(1, 1, s.y);
    m.set(2, 2, s.z);
    return m;
}

/** matrixFromLookAt, right-handed (mirrors upstream exactly). */
export function matrixFromLookAt(eye: float3, center: float3, up: float3): float4x4 {
    const f = normalize3(sub3(eye, center));
    const r = normalize3(cross(up, f));
    const u = cross(f, r);
    const m = float4x4.identity();
    m.set(0, 0, r.x); m.set(0, 1, r.y); m.set(0, 2, r.z);
    m.set(1, 0, u.x); m.set(1, 1, u.y); m.set(1, 2, u.z);
    m.set(2, 0, f.x); m.set(2, 1, f.y); m.set(2, 2, f.z);
    m.set(0, 3, -dot3(r, eye));
    m.set(1, 3, -dot3(u, eye));
    m.set(2, 3, -dot3(f, eye));
    return m;
}

/** perspective (RH, depth [0,1]; fovy in radians; mirrors upstream exactly). */
export function perspective(fovy: number, aspect: number, zNear: number, zFar: number): float4x4 {
    const tanHalfFovy = Math.tan(fovy / 2);
    const m = float4x4.zeros();
    m.set(0, 0, 1 / (aspect * tanHalfFovy));
    m.set(1, 1, 1 / tanHalfFovy);
    m.set(2, 2, zFar / (zNear - zFar));
    m.set(3, 2, -1);
    m.set(2, 3, -(zFar * zNear) / (zFar - zNear));
    return m;
}

/** ortho (RH, depth [0,1]). */
export function ortho(left: number, right: number, bottom: number, top: number, zNear: number, zFar: number): float4x4 {
    const m = float4x4.identity();
    m.set(0, 0, 2 / (right - left));
    m.set(1, 1, 2 / (top - bottom));
    m.set(2, 2, 1 / (zNear - zFar));
    m.set(0, 3, -(right + left) / (right - left));
    m.set(1, 3, -(top + bottom) / (top - bottom));
    m.set(2, 3, zNear / (zNear - zFar));
    return m;
}
