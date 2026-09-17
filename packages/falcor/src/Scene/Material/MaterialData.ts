/**
 * Host-side material data packing mirroring Scene/Material/MaterialData.slang,
 * TextureHandle.slang and BasicMaterialData.slang (bit layouts transcribed from
 * upstream; blob is 128 bytes = 16B header + 112B payload).
 */

import { float3, float4 } from "../../Utils/Math/Vector.js";
import { float32ToFloat16 } from "../../Utils/Math/Float16.js";

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

/** Mirrors Scene/Material/MaterialTypes.slang NormalMapType. */
export enum NormalMapType {
    None = 0,
    /** Normal encoded in RGB channels in [0,1]. */
    RGB = 1,
    /** Tangent-space encoding in RG channels in [0,1]. */
    RG = 2,
}

export enum AlphaMode {
    Opaque = 0,
    Mask = 1,
}

/** float32 -> float16 bit pattern (native float16_t cast: round to nearest, ties up). */
export function f32tof16(value: number): number {
    return float32ToFloat16(value);
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
    /** Packed TextureHandle sampled by the alpha test (native: the base color texture's handle). */
    alphaTextureHandle?: number;
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
    // packedData.w: alpha texture handle (Uniform mode = 0 -> alpha test reads 1, like native)
    const w = desc.alphaTextureHandle ?? 0;
    return new Uint32Array([x >>> 0, y >>> 0, z >>> 0, w >>> 0]);
}

/** Mirrors DiffuseSpecularData.slang: a best-fit diffuse/GGX approximation of a measured BRDF. */
export interface DiffuseSpecularData {
    /** Base color in linear space. */
    baseColor: [number, number, number];
    /** Linearly perceptual roughness (the specular lobe squares it). */
    roughness: number;
    specular: number;
    metallic: number;
    /** Loss when fitted to the target BRDF; not used for rendering. */
    lossValue: number;
}

/**
 * Mirrors MERLMaterialData.slang. Web divergence (docs §9): `bufferID` and
 * `texAlbedoLUT` carry *byte offsets* into the single shared material buffer
 * rather than a buffer index and a texture handle — WGSL has no binding arrays,
 * and the albedo LUT is float data the packed RGBA8 texture array cannot hold.
 */
export interface MERLMaterialDesc {
    /** Byte offset of the interleaved RGB BRDF table. */
    dataOffset: number;
    /** Byte offset of the 256-entry float4 albedo LUT. */
    albedoLUTOffset: number;
    extraData: DiffuseSpecularData;
}

/** Packs a 128-byte MaterialDataBlob for a MERL material (MERLMaterialData layout). */
export function packMERLMaterialBlob(header: MaterialHeaderDesc, merl: MERLMaterialDesc): Uint8Array {
    const blob = new ArrayBuffer(128);
    const u32 = new Uint32Array(blob);
    const dv = new DataView(blob);
    u32.set(packMaterialHeader({ ...header, materialType: MaterialType.MERL, isBasicMaterial: false }), 0);

    let off = 16;
    dv.setUint32(off, merl.dataOffset, true); off += 4; // bufferID -> byte offset
    dv.setUint32(off, 0, true); off += 4; // samplerID
    off = writeDiffuseSpecularData(dv, off, merl.extraData); // DiffuseSpecularData extraData
    dv.setUint32(off, merl.albedoLUTOffset, true); off += 4; // texAlbedoLUT -> byte offset
    return new Uint8Array(blob);
}

/**
 * Mirrors MERLMixMaterialData.slang. Web divergence (docs §9): `bufferID`,
 * `texIndexMap` and `texAlbedoLUT` all carry *byte offsets* into the single
 * shared material buffer. The index map lives there rather than in the packed
 * texture array because that array shares one linear sampler and BRDF indices
 * must be point-sampled.
 */
export interface MERLMixMaterialDesc {
    brdfCount: number;
    /** Stride in bytes between consecutive BRDF tables. */
    byteStride: number;
    /** Byte offset of BRDF 0's table. */
    dataOffset: number;
    /** Byte offset of the per-BRDF DiffuseSpecularData array. */
    extraDataOffset: number;
    /** Byte offset of the `[width, height, one byte per texel]` index map block. */
    indexMapOffset: number;
    /** Byte offset of the 256 x brdfCount float4 albedo LUT. */
    albedoLUTOffset: number;
    /** Packed TextureHandle of the normal map, if any. */
    texNormalMap?: number;
    normalMapType?: NormalMapType;
}

/** Size of DiffuseSpecularData in bytes (baseColor.rgb + 4 scalars). */
export const kDiffuseSpecularDataSize = 28;

/** Writes a DiffuseSpecularData at `off`; returns the offset just past it. */
export function writeDiffuseSpecularData(dv: DataView, off: number, d: DiffuseSpecularData): number {
    for (const c of d.baseColor) { dv.setFloat32(off, c, true); off += 4; }
    dv.setFloat32(off, d.roughness, true); off += 4;
    dv.setFloat32(off, d.specular, true); off += 4;
    dv.setFloat32(off, d.metallic, true); off += 4;
    dv.setFloat32(off, d.lossValue, true); off += 4;
    return off;
}

/** Packs a 128-byte MaterialDataBlob for a MERLMix material (MERLMixMaterialData layout). */
export function packMERLMixMaterialBlob(header: MaterialHeaderDesc, mix: MERLMixMaterialDesc): Uint8Array {
    const blob = new ArrayBuffer(128);
    const u32 = new Uint32Array(blob);
    const dv = new DataView(blob);
    u32.set(packMaterialHeader({ ...header, materialType: MaterialType.MERLMix, isBasicMaterial: false }), 0);

    let off = 16;
    const put = (v: number) => {
        dv.setUint32(off, v, true);
        off += 4;
    };
    // flags: normal map type in bits 0-1; both sampler IDs stay 0 (single sampler, §6.2).
    put(mix.normalMapType ?? (mix.texNormalMap !== undefined ? NormalMapType.RGB : NormalMapType.None));
    put(mix.brdfCount);
    put(mix.byteStride);
    put(mix.dataOffset); // bufferID -> byte offset of BRDF 0
    put(mix.extraDataOffset);
    put(kDiffuseSpecularDataSize);
    put(mix.texNormalMap ?? 0);
    put(mix.indexMapOffset); // texIndexMap -> byte offset
    put(mix.albedoLUTOffset); // texAlbedoLUT -> byte offset
    return new Uint8Array(blob);
}

/**
 * Mirrors RGLMaterialData.slang. Web divergence (docs §9): every `*BufID` is an
 * *element* offset (float index) into the single shared material buffer rather
 * than a buffer index, and `texAlbedoLUT` likewise addresses the LUT inside it.
 */
export interface RGLMaterialDesc {
    phiSize: number;
    thetaSize: number;
    sigmaSize: [number, number];
    ndfSize: [number, number];
    vndfSize: [number, number];
    lumiSize: [number, number];
    /** Element offsets, in the order RGLMaterialData declares the buffers. */
    offsets: {
        theta: number;
        phi: number;
        sigma: number;
        ndf: number;
        vndf: number;
        lumi: number;
        rgb: number;
        vndfMarginal: number;
        lumiMarginal: number;
        vndfConditional: number;
        lumiConditional: number;
        albedoLUT: number;
    };
}

/** Packs a 128-byte MaterialDataBlob for an RGL material (RGLMaterialData layout). */
export function packRGLMaterialBlob(header: MaterialHeaderDesc, rgl: RGLMaterialDesc): Uint8Array {
    const blob = new ArrayBuffer(128);
    const u32 = new Uint32Array(blob);
    const dv = new DataView(blob);
    u32.set(packMaterialHeader({ ...header, materialType: MaterialType.RGL, isBasicMaterial: false }), 0);

    let off = 16;
    const put = (v: number) => {
        dv.setUint32(off, v, true);
        off += 4;
    };
    put(rgl.phiSize);
    put(rgl.thetaSize);
    for (const size of [rgl.sigmaSize, rgl.ndfSize, rgl.vndfSize, rgl.lumiSize]) {
        put(size[0]);
        put(size[1]);
    }
    const o = rgl.offsets;
    for (const v of [o.theta, o.phi, o.sigma, o.ndf, o.vndf, o.lumi, o.rgb, o.vndfMarginal, o.lumiMarginal, o.vndfConditional, o.lumiConditional]) put(v);
    put(0); // samplerID
    put(o.albedoLUT);
    return new Uint8Array(blob);
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
    texDisplacement?: number;
}

/**
 * Packs a full 128-byte MaterialDataBlob for a basic (standard) material.
 * Field order transcribed from BasicMaterialData.slang.
 */
export function packBasicMaterialBlob(header: MaterialHeaderDesc, mat: BasicMaterialDesc): Uint8Array {
    const blob = new ArrayBuffer(128);
    const u32 = new Uint32Array(blob);
    const dv = new DataView(blob);

    // Mirrors Material::updateTextureHandle: the base color handle doubles as the alpha texture handle.
    u32.set(packMaterialHeader({ ...header, isBasicMaterial: true, alphaTextureHandle: header.alphaTextureHandle ?? mat.texBaseColor ?? 0 }), 0);

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
    dv.setUint32(off, mat.texDisplacement ?? 0, true); off += 4; // texDisplacementMap

    return new Uint8Array(blob);
}
