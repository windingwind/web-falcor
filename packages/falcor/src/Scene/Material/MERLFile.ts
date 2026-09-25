/**
 * MERL BRDF database loader mirroring Scene/Material/MERLFile.{h,cpp}.
 *
 * A `.binary` file stores three int32 dimensions (90 x 90 x 180 bins over the
 * half/difference angles) followed by the R, G and B planes of that many
 * doubles each, pre-divided by fixed per-channel scales. The samples are
 * converted to fp32 and interleaved, which is exactly the layout
 * `MERLCommon::eval` indexes on the GPU.
 */

import { Logger } from "../../Utils/Logger.js";
import { RuntimeError } from "../../Core/Error.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import { ImageIO } from "../../Utils/Image/ImageIO.js";
import type { DiffuseSpecularData } from "./MaterialData.js";

/** Angular sampling resolution of the measured data (MERLFile.cpp). */
export const kBRDFSamplingResThetaH = 90;
export const kBRDFSamplingResThetaD = 90;
export const kBRDFSamplingResPhiD = 360;
export const kMERLSampleCount = (kBRDFSamplingResThetaH * kBRDFSamplingResThetaD * kBRDFSamplingResPhiD) / 2;

/** Albedo LUT resolution (MERLMaterialData::kAlbedoLUTSize). */
export const kMERLAlbedoLUTSize = 256;

/** Per-channel scales the database is stored with. */
const kScale = [1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0];

/**
 * Mirrors DiffuseSpecularUtils::loadJSONData's defaults: a mix of diffuse and
 * specular at medium roughness, used when no `.json` fit sits next to the data.
 */
export const kDefaultDiffuseSpecularData: DiffuseSpecularData = {
    baseColor: [0.5, 0.5, 0.5],
    roughness: 0.5,
    specular: 0,
    metallic: 0.5,
    lossValue: 0,
};

export interface MERLBRDF {
    /** File basename without extension (mirrors MERLFile::mDesc.name). */
    name: string;
    /** RGB triples per bin, interleaved — 3 * kMERLSampleCount floats. */
    data: Float32Array;
    /** Best-fit analytic approximation used for sampling (from the `.json` sidecar). */
    extraData: DiffuseSpecularData;
    /** Precomputed albedo LUT (kMERLAlbedoLUTSize float4) from the `.dds` beside the file, if present. */
    albedoLUT?: Float32Array;
}

/** Per-texel BRDF selector (MERLMixMaterial's index map: 8-bit unorm red channel). */
export interface MERLIndexMap {
    width: number;
    height: number;
    /** One index per texel, row-major from the top-left; wraps modulo the BRDF count. */
    indices: Uint8Array;
}

/** Host-side data of a MERLMix material: N BRDFs plus the map that selects them. */
export interface MERLMixData {
    brdfs: MERLBRDF[];
    indexMap: MERLIndexMap;
    /** Packed TextureHandle of the normal map, if any. */
    texNormalMap?: number;
}

/** sRGB -> linear (matches DiffuseSpecularUtils::loadJSONData's conversion). */
function srgbToLinear(c: number): number {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Mirrors DiffuseSpecularUtils::loadJSONData over an already-fetched sidecar. */
export function parseDiffuseSpecularJSON(text: string): DiffuseSpecularData {
    const doc = JSON.parse(text) as Record<string, number[] | number>;
    const srgb = (doc["base_color_srgb"] as number[]) ?? [0.5, 0.5, 0.5];
    return {
        baseColor: [srgbToLinear(srgb[0]!), srgbToLinear(srgb[1]!), srgbToLinear(srgb[2]!)],
        roughness: (doc["roughness"] as number) ?? 0.5,
        specular: (doc["specular"] as number) ?? 0,
        metallic: (doc["metallic"] as number) ?? 0.5,
        lossValue: (doc["loss_value"] as number) ?? 0,
    };
}

/**
 * Mirrors MERLFile::loadBRDF + prepareData: reads the samples, applies the
 * per-channel scales and sanitizes them (inf/NaN become zero, negatives are
 * clamped) exactly as native does.
 */
export function parseMERLBinary(buffer: ArrayBuffer, name: string, extraData: DiffuseSpecularData = kDefaultDiffuseSpecularData): MERLBRDF {
    const view = new DataView(buffer);
    const dims = [view.getInt32(0, true), view.getInt32(4, true), view.getInt32(8, true)];
    const n = dims[0]! * dims[1]! * dims[2]!;
    if (n !== kMERLSampleCount) throw new RuntimeError(`MERLFile: dimensions don't match in '${name}' (${dims.join("x")})`);
    if (buffer.byteLength < 12 + 3 * n * 8) throw new RuntimeError(`MERLFile: failed to load BRDF data from '${name}' (file too short)`);

    const data = new Float32Array(n * 3);
    let negCount = 0;
    let infCount = 0;
    let nanCount = 0;
    for (let i = 0; i < n; i++) {
        // The file stores whole R, G and B planes; the GPU layout is interleaved.
        const rgb = [
            view.getFloat64(12 + i * 8, true) * kScale[0]!,
            view.getFloat64(12 + (i + n) * 8, true) * kScale[1]!,
            view.getFloat64(12 + (i + 2 * n) * 8, true) * kScale[2]!,
        ];
        const isNeg = rgb.some((v) => v! < 0);
        const isInf = rgb.some((v) => v === Infinity || v === -Infinity);
        const isNaN_ = rgb.some((v) => Number.isNaN(v));
        if (isNeg) negCount++;
        if (isInf) infCount++;
        if (isNaN_) nanCount++;
        for (let c = 0; c < 3; c++) {
            data[i * 3 + c] = isInf || isNaN_ ? 0 : Math.max(rgb[c]!, 0);
        }
    }
    if (negCount > 0) Logger.warning(`MERL BRDF ${name} has ${negCount} samples with negative values. Clamped to zero.`);
    if (infCount > 0) Logger.warning(`MERL BRDF ${name} has ${infCount} samples with inf values. Sample set to zero.`);
    if (nanCount > 0) Logger.warning(`MERL BRDF ${name} has ${nanCount} samples with NaN values. Sample set to zero.`);

    return { name, data, extraData };
}

/**
 * Loads a `.binary` and its optional `.json` fit sidecar (web divergence,
 * docs §9: native reads both from disk in the constructor).
 */
export async function loadMERLBinary(url: string): Promise<MERLBRDF> {
    const res = await fetch(url);
    if (!res.ok) throw new RuntimeError(`MERLFile: failed to fetch '${url}' (${res.status})`);
    const name = url.split("/").pop()!.replace(/\.[^.]*$/, "");

    let extraData = kDefaultDiffuseSpecularData;
    const jsonUrl = url.replace(/\.[^.]*$/, ".json");
    try {
        const sidecar = await fetch(jsonUrl);
        if (sidecar.ok) extraData = parseDiffuseSpecularJSON(await sidecar.text());
        else Logger.warning(`MERLFile: Failed to load associated JSON data for BRDF '${name}'.`);
    } catch {
        Logger.warning(`MERLFile: Failed to load associated JSON data for BRDF '${name}'.`);
    }
    const brdf = parseMERLBinary(await res.arrayBuffer(), name, extraData);
    // MERLFile::prepareAlbedoLUT: a cached RGBA32Float 256x1 table beside the BRDF is used as is.
    try {
        const lut = await fetch(url.replace(/\.[^.]*$/, ".dds"));
        if (lut.ok) {
            const bitmap = ImageIO.loadBitmapFromDDS(new Uint8Array(await lut.arrayBuffer()));
            if (bitmap.format === ResourceFormat.RGBA32Float && bitmap.width === kMERLAlbedoLUTSize && bitmap.height === 1) {
                brdf.albedoLUT = new Float32Array(bitmap.data.buffer.slice(bitmap.data.byteOffset, bitmap.data.byteOffset + kMERLAlbedoLUTSize * 16));
                Logger.info(`Loaded albedo LUT from '${url.replace(/\.[^.]*$/, ".dds")}'.`);
            }
        }
    } catch {
        /* no cached table: computed with the scene */
    }
    return brdf;
}
