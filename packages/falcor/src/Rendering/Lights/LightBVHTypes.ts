/**
 * LightBVH packed node types mirroring Rendering/Lights/LightBVHTypes.slang
 * (the default compressed 32-byte layout) plus the host-side conversion
 * helpers they depend on (Utils/Math/{FormatConversion.h,PackedFormats.h} semantics).
 */

import { float16ToFloat32, float32ToFloat16 } from "../../Utils/Math/Float16.js";

export const kInvalidCosConeAngle = -1;
export const kTriangleCountBits = 4;
export const kTriangleOffsetBits = 31 - kTriangleCountBits;
export const kMaxLeafTriangleCount = 1 << kTriangleCountBits;
export const kMaxLeafTriangleOffset = 1 << kTriangleOffsetBits;
export const kMaxBVHDepth = 64;

export type Vec3 = [number, number, number];

/** Native float16_t cast (Float16.cpp): round to nearest, ties up. */
export const f32tof16RoundHalfUp = float32ToFloat16;
export const f16tof32 = float16ToFloat32;

/** FormatConversion.h floatToSnorm16: NaN -> 0, clamp, trunc(v*32767 +- 0.5). */
export function floatToSnorm16(v: number): number {
    v = Number.isNaN(v) ? 0 : Math.min(Math.max(v, -1), 1);
    return Math.trunc(v * 32767 + (v >= 0 ? 0.5 : -0.5));
}

function snorm16ToFloat(bits16: number): number {
    const signed = (bits16 << 16) >> 16;
    return Math.max(Math.fround(signed / 32767), -1);
}

function octWrap(x: number, y: number): [number, number] {
    return [(1 - Math.abs(y)) * (x >= 0 ? 1 : -1), (1 - Math.abs(x)) * (y >= 0 ? 1 : -1)];
}

/** PackedFormats.h encodeNormal2x16 (octahedral snorm). */
export function encodeNormal2x16Host(n: Vec3): number {
    const invL1 = Math.fround(1 / (Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2])));
    let px = Math.fround(n[0] * invL1);
    let py = Math.fround(n[1] * invL1);
    if (n[2] < 0) [px, py] = octWrap(px, py);
    return ((floatToSnorm16(px) & 0xffff) | (floatToSnorm16(py) << 16)) >>> 0;
}

/** PackedFormats.h decodeNormal2x16 + Falcor normalize (v * (1/sqrt(dot))). */
export function decodeNormal2x16Host(packed: number): Vec3 {
    const px = snorm16ToFloat(packed & 0xffff);
    const py = snorm16ToFloat((packed >>> 16) & 0xffff);
    let nx = px;
    let ny = py;
    const nz = Math.fround(1 - Math.abs(px) - Math.abs(py));
    if (nz < 0) [nx, ny] = octWrap(nx, ny);
    const invLen = Math.fround(1 / Math.sqrt(Math.fround(nx * nx + ny * ny + nz * nz)));
    return [Math.fround(nx * invLen), Math.fround(ny * invLen), Math.fround(nz * invLen)];
}

export interface SharedNodeAttributes {
    origin: Vec3;
    extent: Vec3;
    flux: number;
    cosConeAngle: number;
    coneDirection: Vec3;
}

/** One compressed PackedNode = 8 uints (32 bytes). */
export class PackedNode {
    data = new Uint32Array(8);

    isLeaf(): boolean {
        return this.data[0]! >>> 31 !== 0;
    }

    setInternalNode(rightChildIdx: number, attribs: SharedNodeAttributes): void {
        this.data[0] = rightChildIdx >>> 0;
        this.setNodeAttributes(attribs);
    }

    setLeafNode(triangleCount: number, triangleOffset: number, attribs: SharedNodeAttributes): void {
        this.data[0] = ((1 << 31) | (triangleCount << kTriangleOffsetBits) | triangleOffset) >>> 0;
        this.setNodeAttributes(attribs);
    }

    getRightChildIdx(): number {
        return this.data[0]!;
    }

    getLeafTriangleCount(): number {
        return (this.data[0]! >>> kTriangleOffsetBits) & ((1 << kTriangleCountBits) - 1);
    }

    getLeafTriangleOffset(): number {
        return this.data[0]! & ((1 << kTriangleOffsetBits) - 1);
    }

    setNodeAttributes(a: SharedNodeAttributes): void {
        const f32 = new Float32Array(1);
        const u32 = new Uint32Array(f32.buffer);
        const asuint = (v: number) => {
            f32[0] = v;
            return u32[0]!;
        };
        this.data[1] = asuint(a.origin[0]);
        this.data[2] = asuint(a.origin[1]);
        this.data[3] = asuint(a.origin[2]);
        // Note: (uint)((cos+1)*32767) truncates toward zero (native comment: round
        // toward -inf so the quantized cone angle is equal or larger).
        const packedAngle = Math.trunc(Math.fround((a.cosConeAngle + 1) * 32767)) >>> 0;
        this.data[4] = (f32tof16RoundHalfUp(a.extent[0]) | (f32tof16RoundHalfUp(a.extent[1]) << 16)) >>> 0;
        this.data[5] = (f32tof16RoundHalfUp(a.extent[2]) | (packedAngle << 16)) >>> 0;
        this.data[6] = encodeNormal2x16Host(a.coneDirection);
        this.data[7] = asuint(a.flux);
    }

    getNodeAttributes(): SharedNodeAttributes {
        const u32 = new Uint32Array(1);
        const f32 = new Float32Array(u32.buffer);
        const asfloat = (v: number) => {
            u32[0] = v;
            return f32[0]!;
        };
        return {
            origin: [asfloat(this.data[1]!), asfloat(this.data[2]!), asfloat(this.data[3]!)],
            extent: [f16tof32(this.data[4]! & 0xffff), f16tof32(this.data[4]! >>> 16), f16tof32(this.data[5]! & 0xffff)],
            cosConeAngle: Math.fround((this.data[5]! >>> 16) * (1 / 32767) - 1),
            coneDirection: decodeNormal2x16Host(this.data[6]!),
            flux: asfloat(this.data[7]!),
        };
    }
}
