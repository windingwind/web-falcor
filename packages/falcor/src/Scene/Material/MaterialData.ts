/**
 * Host-side material data packing mirroring Scene/Material/MaterialData.slang,
 * TextureHandle.slang and BasicMaterialData.slang (bit layouts transcribed from
 * upstream; blob is 128 bytes = 16B header + 112B payload).
 */

import { float3, float4 } from "../../Utils/Math/Vector.js";

/** Mirrors MaterialType (MaterialTypes.slang). */
export enum MaterialType {
    Standard = 1,
    Cloth = 2,
    Hair = 3,
    MERL = 4,
    MERLMix = 5,
    PBRTDiffuse = 6,
    PBRTDiffuseTransmission = 7,
    PBRTConductor = 8,
    PBRTDielectric = 9,
    PBRTCoatedConductor = 10,
    PBRTCoatedDiffuse = 11,
    RGL = 12,
}

export enum AlphaMode {
    Opaque = 0,
    Mask = 1,
}

/** float32 -> float16 bit pattern. */
export function f32tof16(value: number): number {
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    f32[0] = value;
    const x = u32[0]!;
    const sign = (x >> 16) & 0x8000;
    let exp = ((x >> 23) & 0xff) - 127 + 15;
    let frac = (x >> 13) & 0x3ff;
    if (exp <= 0) return sign; // flush denormals/underflow to signed zero
    if (exp >= 31) return sign | 0x7c00; // overflow -> inf
    return sign | (exp << 10) | frac;
}

/** Mirrors TextureHandle packing (29-bit ID, 2-bit mode, 1-bit udim). */
export enum TextureHandleMode {
    Uniform = 0,
    Texture = 1,
}

export function packTextureHandle(mode: TextureHandleMode, textureID = 0, udim = false): number {
    return (textureID & 0x1fffffff) | ((mode & 0x3) << 29) | ((udim ? 1 : 0) << 31);
}

export interface MaterialHeaderDesc {
    materialType: MaterialType;
    nestedPriority?: number;
    activeLobes?: number; // LobeType mask; 0xff = all
    doubleSided?: boolean;
    thinSurface?: boolean;
    emissive?: boolean;
    isBasicMaterial?: boolean;
    alphaThreshold?: number;
    alphaMode?: AlphaMode;
    defaultTextureSamplerID?: number;
    lightProfileEnabled?: boolean;
    deltaSpecular?: boolean;
    ior?: number;
}

/** Packs MaterialHeader's uint4 (bit layout from MaterialData.slang). */
export function packMaterialHeader(desc: MaterialHeaderDesc): Uint32Array {
    // packedData.x: type[16] | nestedPriority[4] | lobes[8] | doubleSided | thin | emissive | isBasic
    let x = 0;
    x |= (desc.materialType & 0xffff) << 0;
    x |= ((desc.nestedPriority ?? 0) & 0xf) << 16;
    x |= ((desc.activeLobes ?? 0xff) & 0xff) << 20;
    x |= (desc.doubleSided ? 1 : 0) << 28;
    x |= (desc.thinSurface ? 1 : 0) << 29;
    x |= (desc.emissive ? 1 : 0) << 30;
    x |= (desc.isBasicMaterial ?? true ? 1 : 0) << 31;

    // packedData.y: alphaThreshold f16[16] | alphaMode[1] | samplerID[8] | lightProfile | deltaSpecular
    let y = 0;
    y |= f32tof16(desc.alphaThreshold ?? 0.5) << 0;
    y |= ((desc.alphaMode ?? AlphaMode.Opaque) & 0x1) << 16;
    y |= ((desc.defaultTextureSamplerID ?? 0) & 0xff) << 17;
    y |= (desc.lightProfileEnabled ? 1 : 0) << 25;
    y |= (desc.deltaSpecular ? 1 : 0) << 26;

    // packedData.z: IoR f16[16]
    const z = f32tof16(desc.ior ?? 1.5);
    // packedData.w: alpha texture handle
    const w = 0;
    return new Uint32Array([x >>> 0, y >>> 0, z >>> 0, w >>> 0]);
}

export interface BasicMaterialDesc {
    baseColor?: float4;
    /** occlusion (R), roughness (G), metallic (B) in MetalRough mode. */
    specular?: float4;
    /** Transmission color (PBRTConductor reads the conductor k from here). */
    transmission?: float3;
    diffuseTransmission?: number;
    emissive?: float3;
    emissiveFactor?: number;
    specularTransmission?: number;
    volumeAbsorption?: float3;
    volumeScattering?: float3;
    displacementScale?: number;
    displacementOffset?: number;
    texBaseColor?: number; // packed TextureHandle
    texSpecular?: number;
    texEmissive?: number;
    texNormalMap?: number;
}

/**
 * Packs a full 128-byte MaterialDataBlob for a basic (standard) material.
 * Field order transcribed from BasicMaterialData.slang.
 */
export function packBasicMaterialBlob(header: MaterialHeaderDesc, mat: BasicMaterialDesc): Uint8Array {
    const blob = new ArrayBuffer(128);
    const u32 = new Uint32Array(blob);
    const dv = new DataView(blob);

    u32.set(packMaterialHeader({ ...header, isBasicMaterial: true }), 0);

    // Payload starts at byte 16 (BasicMaterialData layout).
    let off = 16;
    // flags: bit 0 shading model (MetalRough=0), bits 1-2 normal map type
    // (None=0, RGB=1 - standard 8-bit normal maps; native detects RG/BC5,
    // which the web texture pipeline does not produce).
    dv.setUint32(off, mat.texNormalMap !== undefined ? 1 << 1 : 0, true); off += 4;
    dv.setFloat32(off, mat.emissiveFactor ?? 1, true); off += 4;

    const bc = mat.baseColor ?? new float4(1, 1, 1, 1);
    dv.setUint16(off, f32tof16(bc.x), true); dv.setUint16(off + 2, f32tof16(bc.y), true);
    dv.setUint16(off + 4, f32tof16(bc.z), true); dv.setUint16(off + 6, f32tof16(bc.w), true);
    off += 8;
    const sp = mat.specular ?? new float4(0, 0.5, 0, 0);
    dv.setUint16(off, f32tof16(sp.x), true); dv.setUint16(off + 2, f32tof16(sp.y), true);
    dv.setUint16(off + 4, f32tof16(sp.z), true); dv.setUint16(off + 6, f32tof16(sp.w), true);
    off += 8;

    const em = mat.emissive ?? new float3(0, 0, 0);
    dv.setFloat32(off, em.x, true); dv.setFloat32(off + 4, em.y, true); dv.setFloat32(off + 8, em.z, true);
    off += 12;
    dv.setUint16(off, f32tof16(mat.specularTransmission ?? 0), true); off += 2;
    const tr = mat.transmission ?? new float3(1, 1, 1);
    dv.setUint16(off, f32tof16(tr.x), true); dv.setUint16(off + 2, f32tof16(tr.y), true); dv.setUint16(off + 4, f32tof16(tr.z), true); off += 6;
    dv.setUint16(off, f32tof16(mat.diffuseTransmission ?? 0), true); off += 2; // diffuseTransmission
    // volumeScattering f16x3 + pad (4 halves)
    const vs = mat.volumeScattering ?? new float3(0, 0, 0);
    dv.setUint16(off, f32tof16(vs.x), true); dv.setUint16(off + 2, f32tof16(vs.y), true); dv.setUint16(off + 4, f32tof16(vs.z), true); dv.setUint16(off + 6, 0, true);
    off += 8;
    const va = mat.volumeAbsorption ?? new float3(0, 0, 0);
    dv.setUint16(off, f32tof16(va.x), true); dv.setUint16(off + 2, f32tof16(va.y), true); dv.setUint16(off + 4, f32tof16(va.z), true);
    off += 6;
    dv.setUint16(off, 0, true); off += 2; // volumeAnisotropy
    off += 2; // trailing pad: displacementScale is 4-byte aligned (payload offset 64)
    dv.setFloat32(off, mat.displacementScale ?? 0, true); off += 4; // displacementScale
    dv.setFloat32(off, mat.displacementOffset ?? 0, true); off += 4; // displacementOffset

    dv.setUint32(off, mat.texBaseColor ?? 0, true); off += 4;
    dv.setUint32(off, mat.texSpecular ?? 0, true); off += 4;
    dv.setUint32(off, mat.texEmissive ?? 0, true); off += 4;
    dv.setUint32(off, mat.texNormalMap ?? 0, true); off += 4;
    dv.setUint32(off, 0, true); off += 4; // texTransmission
    dv.setUint32(off, 0, true); off += 4; // texDisplacementMap

    return new Uint8Array(blob);
}
