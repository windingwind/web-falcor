/**
 * Host-side packers for GPU scene structures (Scene/SceneTypes.slang layouts
 * transcribed from upstream).
 */

import { float2, float3, float4 } from "../Utils/Math/Vector.js";
import { f32tof16 } from "./Material/MaterialData.js";

export interface StaticVertex {
    position: float3;
    normal: float3;
    /** xyz = tangent, w = sign (0 if invalid). */
    tangent: float4;
    texCrd: float2;
    curveRadius?: number;
}

/** Octahedral snorm2x16 encode (Utils/Math/PackedFormats.slang encodeNormal2x16). */
function encodeNormal2x16(n: float3): number {
    const l1 = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z) || 1;
    let ox = n.x / l1;
    let oy = n.y / l1;
    if (n.z < 0) {
        const tx = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1);
        const ty = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
        ox = tx;
        oy = ty;
    }
    const sx = Math.round(Math.min(Math.max(ox, -1), 1) * 32767) & 0xffff;
    const sy = Math.round(Math.min(Math.max(oy, -1), 1) * 32767) & 0xffff;
    return (sy << 16) | sx;
}

/**
 * WGSL address-space layout: vec3 members align to 16 bytes, so the packed
 * vertex struct is 48 bytes on the web (native Falcor packs it into 32).
 * Element offsets: position@0, packedNormalTangentCurveRadius@16, texCrd@32.
 */
export const kPackedStaticVertexSize = 48;

/** Packs vertices into PackedStaticVertexData layout (SceneTypes.slang pack()). */
export function packStaticVertices(vertices: StaticVertex[]): ArrayBuffer {
    const buffer = new ArrayBuffer(vertices.length * kPackedStaticVertexSize);
    const dv = new DataView(buffer);
    for (let i = 0; i < vertices.length; i++) {
        const v = vertices[i]!;
        const base = i * kPackedStaticVertexSize;
        dv.setFloat32(base + 0, v.position.x, true);
        dv.setFloat32(base + 4, v.position.y, true);
        dv.setFloat32(base + 8, v.position.z, true);

        const nx = f32tof16(v.normal.x);
        const ny = f32tof16(v.normal.y);
        const nz = f32tof16(v.normal.z);
        let packedTangentSign = v.tangent.w;
        if ((v.curveRadius ?? 0) > 0) packedTangentSign *= v.curveRadius!;
        const tw = f32tof16(packedTangentSign);

        dv.setUint32(base + 16, ((ny << 16) | nx) >>> 0, true);
        dv.setUint32(base + 20, ((tw << 16) | nz) >>> 0, true);
        dv.setUint32(base + 24, encodeNormal2x16(new float3(v.tangent.x, v.tangent.y, v.tangent.z)) >>> 0, true);

        dv.setFloat32(base + 32, v.texCrd.x, true);
        dv.setFloat32(base + 36, v.texCrd.y, true);
    }
    return buffer;
}

/** Mirrors GeometryType (SceneDefines.slangh GEOMETRY_TYPE_* values). */
export enum GeometryType {
    None = 0,
    TriangleMesh = 1,
    DisplacedTriangleMesh = 2,
    Curve = 3,
    SDFGrid = 5,
    Custom = 6,
}

export const kGeometryInstanceSize = 32; // 8 uints

export interface GeometryInstance {
    type: GeometryType;
    globalMatrixID: number;
    materialID: number;
    geometryID: number;
    vbOffset: number;
    ibOffset: number;
    instanceIndex: number;
    geometryIndex: number;
    flags?: number;
}

export function packGeometryInstances(instances: GeometryInstance[]): Uint32Array {
    const out = new Uint32Array(instances.length * 8);
    for (let i = 0; i < instances.length; i++) {
        const inst = instances[i]!;
        // Upper 3 bits of flags store the geometry type (kTypeOffset = 29).
        out[i * 8 + 0] = (((inst.type & 0x7) << 29) | ((inst.flags ?? 0) & 0x1fffffff)) >>> 0;
        out[i * 8 + 1] = inst.globalMatrixID;
        out[i * 8 + 2] = inst.materialID;
        out[i * 8 + 3] = inst.geometryID;
        out[i * 8 + 4] = inst.vbOffset;
        out[i * 8 + 5] = inst.ibOffset;
        out[i * 8 + 6] = inst.instanceIndex;
        out[i * 8 + 7] = inst.geometryIndex;
    }
    return out;
}

export const kMeshDescSize = 32; // 8 uints

export interface MeshDescData {
    vbOffset: number;
    ibOffset: number;
    vertexCount: number;
    indexCount: number;
    skinningVbOffset?: number;
    prevVbOffset?: number;
    materialID: number;
    flags?: number; // MeshFlags: bit0 = Use16BitIndices, bit1 = IsSkinned, ...
}

export function packMeshDescs(meshes: MeshDescData[]): Uint32Array {
    const out = new Uint32Array(meshes.length * 8);
    for (let i = 0; i < meshes.length; i++) {
        const m = meshes[i]!;
        out.set(
            [m.vbOffset, m.ibOffset, m.vertexCount, m.indexCount, m.skinningVbOffset ?? 0, m.prevVbOffset ?? 0, m.materialID, m.flags ?? 0],
            i * 8,
        );
    }
    return out;
}

/** Mirrors LightType (LightData.slang). */
export enum LightType {
    Point = 0,
    Directional = 1,
    Distant = 2,
    Rect = 3,
    Disc = 4,
    Sphere = 5,
}

export interface AnalyticLight {
    type: LightType;
    /** Position (point lights) in world space. */
    posW?: float3;
    /** Direction (directional/spot), normalized. */
    dirW?: float3;
    /** Emitted radiance/intensity. */
    intensity: float3;
}

export const kLightDataSize = 224; // 6x16B rows + 2x 64B float4x4 (rows at 96/160)

/** Packs LightData (LightData.slang layout; 16-byte rows, matrices identity). */
export function packLights(lights: AnalyticLight[]): ArrayBuffer {
    const buffer = new ArrayBuffer(Math.max(lights.length, 1) * kLightDataSize);
    const dv = new DataView(buffer);
    lights.forEach((light, i) => {
        const base = i * kLightDataSize;
        const posW = light.posW ?? new float3(0, 0, 0);
        const dirW = light.dirW ?? new float3(0, -1, 0);
        dv.setFloat32(base + 0, posW.x, true);
        dv.setFloat32(base + 4, posW.y, true);
        dv.setFloat32(base + 8, posW.z, true);
        dv.setUint32(base + 12, light.type, true);
        dv.setFloat32(base + 16, dirW.x, true);
        dv.setFloat32(base + 20, dirW.y, true);
        dv.setFloat32(base + 24, dirW.z, true);
        dv.setFloat32(base + 28, Math.PI, true); // openingAngle (full sphere)
        dv.setFloat32(base + 32, light.intensity.x, true);
        dv.setFloat32(base + 36, light.intensity.y, true);
        dv.setFloat32(base + 40, light.intensity.z, true);
        dv.setFloat32(base + 44, -1, true); // cosOpeningAngle
        dv.setFloat32(base + 48, 0.9999893, true); // cosSubtendedAngle
        dv.setFloat32(base + 52, 0, true); // penumbraAngle
        // rows 4-5 (tangent/surfaceArea/bitangent) zeroed; matrices identity at 96/160.
        for (let m = 0; m < 2; m++) {
            const mBase = base + 96 + m * 64;
            for (let d = 0; d < 4; d++) dv.setFloat32(mBase + d * 20, 1, true); // diagonal (row stride 16 + 4)
        }
    });
    return buffer;
}
