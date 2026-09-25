/**
 * RGL measured-BSDF loader mirroring Scene/Material/RGLFile.{h,cpp} and the
 * data preparation in RGLMaterial::loadBRDF / RGLCommon.cpp.
 *
 * The files (rgl.epfl.ch/materials, Dupuy & Jakob 2018) are "tensor_file"
 * containers: a small header followed by named N-dimensional arrays. Rendering
 * needs theta_i, phi_i, sigma, ndf, vndf, luminance and rgb; the VNDF and
 * luminance tables are additionally turned into samplable marginal/conditional
 * CDFs, exactly as native does on load.
 */

import { RuntimeError } from "../../Core/Error.js";

/** Field types Falcor reads; all others are skipped. */
export enum RGLFieldType {
    UInt8 = 1,
    UInt32 = 5,
    Float32 = 10,
}

export interface RGLField {
    name: string;
    type: RGLFieldType;
    shape: number[];
    /** Elements in file order (row-major). */
    data: Float32Array | Uint32Array | Uint8Array;
}

/** Mirrors RGLFile::MeasurementData plus the sizes RGLMaterialData records. */
export interface RGLMeasurement {
    name: string;
    description: string;
    isotropic: boolean;
    thetaI: Float32Array;
    phiI: Float32Array;
    /** [width, height] of the 2D tables, as RGLMaterialData stores them. */
    sigmaSize: [number, number];
    sigma: Float32Array;
    ndfSize: [number, number];
    ndf: Float32Array;
    /** [width, height] of each 2D slice of the 4D tables. */
    vndfSize: [number, number];
    vndf: Float32Array;
    lumiSize: [number, number];
    luminance: Float32Array;
    rgb: Float32Array;
    /** Marginal/conditional CDFs of the (normalized) VNDF and luminance tables. */
    vndfMarginal: Float32Array;
    vndfConditional: Float32Array;
    lumiMarginal: Float32Array;
    lumiConditional: Float32Array;
    /** Precomputed albedo LUT (kRGLAlbedoLUTSize float4) from the `.dds` beside the file, if present. */
    albedoLUT?: Float32Array;
}

/** Maximum table resolution native accepts (RGLMaterialData::kMaxResolution). */
export const kRGLMaxResolution = 0xffff;
/** Albedo LUT resolution (RGLMaterialData::kAlbedoLUTSize). */
export const kRGLAlbedoLUTSize = 256;

function fieldSize(type: RGLFieldType): number {
    switch (type) {
        case RGLFieldType.UInt8:
            return 1;
        case RGLFieldType.UInt32:
        case RGLFieldType.Float32:
            return 4;
        default:
            return 0;
    }
}

/** Mirrors RGLFile::RGLFile(std::ifstream&): reads the header and every supported field. */
export function parseRGLTensorFile(buffer: ArrayBuffer): Map<string, RGLField> {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    if (new TextDecoder().decode(bytes.subarray(0, 11)) !== "tensor_file") throw new RuntimeError("RGLFile: invalid file header");
    const version = [bytes[12], bytes[13]];
    if (version[0] !== 1 || version[1] !== 0) throw new RuntimeError(`RGLFile: unsupported file version ${version.join(".")}`);
    const fieldCount = view.getUint32(14, true);

    const fields = new Map<string, RGLField>();
    let o = 18;
    for (let i = 0; i < fieldCount; i++) {
        const nameLength = view.getUint16(o, true);
        o += 2;
        const name = new TextDecoder().decode(bytes.subarray(o, o + nameLength));
        o += nameLength;
        const dim = view.getUint16(o, true);
        o += 2;
        const type = bytes[o]! as RGLFieldType;
        o += 1;
        const offset = Number(view.getBigUint64(o, true));
        o += 8;
        const shape: number[] = [];
        for (let d = 0; d < dim; d++) {
            shape.push(Number(view.getBigUint64(o, true)));
            o += 8;
        }
        const elemSize = fieldSize(type);
        if (elemSize === 0) continue; // unsupported type — not needed
        const count = shape.reduce((a, b) => a * b, 1);
        if (offset + count * elemSize > buffer.byteLength) throw new RuntimeError(`RGLFile: error parsing field '${name}': file truncated`);
        // Field payloads are not aligned in the file, so copy rather than view.
        const raw = bytes.slice(offset, offset + count * elemSize);
        const data =
            type === RGLFieldType.Float32
                ? new Float32Array(raw.buffer, raw.byteOffset, count)
                : type === RGLFieldType.UInt32
                  ? new Uint32Array(raw.buffer, raw.byteOffset, count)
                  : raw;
        fields.set(name, { name, type, shape, data });
    }
    return fields;
}

/** Mirrors RGLFile::saveFile: header, field descriptions, then 8-byte aligned data blocks. */
export function writeRGLTensorFile(fields: readonly RGLField[]): Uint8Array {
    const align = (a: number) => Math.ceil(a / 8) * 8;
    const names = fields.map((f) => new TextEncoder().encode(f.name));
    let headerSize = 18;
    fields.forEach((f, i) => (headerSize += 13 + names[i]!.length + 8 * f.shape.length));
    let total = align(headerSize);
    const offsets = fields.map((f) => {
        const o = total;
        total = align(total + fieldSize(f.type) * f.data.length);
        return o;
    });
    const bytes = new Uint8Array(total);
    const view = new DataView(bytes.buffer);
    bytes.set(new TextEncoder().encode("tensor_file"), 0);
    bytes[12] = 1;
    bytes[13] = 0;
    view.setUint32(14, fields.length, true);
    let o = 18;
    fields.forEach((f, i) => {
        view.setUint16(o, names[i]!.length, true);
        bytes.set(names[i]!, o + 2);
        o += 2 + names[i]!.length;
        view.setUint16(o, f.shape.length, true);
        bytes[o + 2] = f.type;
        view.setBigUint64(o + 3, BigInt(offsets[i]!), true);
        o += 11;
        for (const d of f.shape) (view.setBigUint64(o, BigInt(d), true), (o += 8));
        bytes.set(new Uint8Array(f.data.buffer, f.data.byteOffset, f.data.byteLength), offsets[i]!);
    });
    return bytes;
}

/**
 * Mirrors SamplableDistribution4D::build2DSlice: per 2D slice, a linearly
 * interpolated conditional CDF per row, a marginal CDF over rows, and the PDF
 * itself — all normalized by the slice integral. Accumulation is done in double
 * precision like native.
 */
function buildSlice(width: number, height: number, pdf: Float32Array, pdfBase: number, marginal: Float32Array, marginalBase: number, conditional: Float32Array, conditionalBase: number): void {
    let tableSum = 0;
    for (let i = 0; i < width * height; i++) tableSum += pdf[pdfBase + i]!;
    // Edge case: whole slice is zero. Reset to a uniform distribution.
    if (tableSum === 0) {
        for (let i = 0; i < width * height; i++) pdf[pdfBase + i] = 1 / (width * height);
        tableSum = 1;
    }

    // Step 1: row sums (trapezoidal, matching the shader's linear interpolation).
    for (let y = 0; y < height; y++) {
        let rowSum = 0;
        conditional[conditionalBase + y * width] = 0;
        for (let x = 1; x < width; x++) {
            const idx = x + y * width;
            rowSum += (pdf[pdfBase + idx - 1]! + pdf[pdfBase + idx]!) * 0.5;
            conditional[conditionalBase + idx] = rowSum;
        }
    }

    // Step 2: marginal over rows.
    let marginalSum = 0;
    marginal[marginalBase] = 0;
    for (let y = 1; y < height; y++) {
        marginalSum += (conditional[conditionalBase + y * width - 1]! + conditional[conditionalBase + (y + 1) * width - 1]!) * 0.5;
        marginal[marginalBase + y] = marginalSum;
    }

    // Step 3: normalize.
    for (let y = 0; y < height; y++) marginal[marginalBase + y] = marginal[marginalBase + y]! / marginalSum;
    for (let i = 0; i < width * height; i++) {
        pdf[pdfBase + i] = pdf[pdfBase + i]! / marginalSum;
        conditional[conditionalBase + i] = conditional[conditionalBase + i]! / marginalSum;
    }
}

/**
 * Mirrors SamplableDistribution4D: the 4D table is a 2D grid of 2D slices; each
 * slice gets its own marginal/conditional CDFs. `size` is (x, y, z, w) with z/w
 * the slice dimensions, as native passes them.
 */
export function buildSamplableDistribution4D(source: Float32Array, size: [number, number, number, number]): { pdf: Float32Array; marginal: Float32Array; conditional: Float32Array } {
    const [x, y, z, w] = size;
    const n = x * y * z * w;
    if (source.length < n) throw new RuntimeError(`RGLFile: distribution data too small (${source.length} < ${n})`);
    const pdf = Float32Array.from(source.subarray(0, n));
    const conditional = new Float32Array(n);
    const marginal = new Float32Array((n / z) | 0);
    const sliceStride = z * w;
    for (let i = 0; i < n; i += sliceStride) {
        buildSlice(z, w, pdf, i, marginal, (i / z) | 0, conditional, i);
    }
    return { pdf, marginal, conditional };
}

/** Mirrors RGLFile::validate + RGLMaterial::loadBRDF's size checks and table preparation. */
export function parseRGLFile(buffer: ArrayBuffer, name: string): RGLMeasurement {
    const fields = parseRGLTensorFile(buffer);
    const need = (key: string, type: RGLFieldType, dim: number) => {
        const f = fields.get(key);
        if (!f || f.type !== type || f.shape.length !== dim) throw new RuntimeError(`RGLFile: ${key} field missing or invalid`);
        return f;
    };
    const thetaI = need("theta_i", RGLFieldType.Float32, 1);
    const phiI = need("phi_i", RGLFieldType.Float32, 1);
    const sigma = need("sigma", RGLFieldType.Float32, 2);
    const ndf = need("ndf", RGLFieldType.Float32, 2);
    const vndf = need("vndf", RGLFieldType.Float32, 4);
    const luminance = need("luminance", RGLFieldType.Float32, 4);
    const rgb = need("rgb", RGLFieldType.Float32, 5);
    const description = fields.get("description");

    if (vndf.shape[0] !== phiI.shape[0] || vndf.shape[1] !== thetaI.shape[0]) throw new RuntimeError("RGLFile: vndf field missing or invalid");
    if (luminance.shape[0] !== phiI.shape[0] || luminance.shape[1] !== thetaI.shape[0] || luminance.shape[2] !== luminance.shape[3]) {
        throw new RuntimeError("RGLFile: luminance field missing or invalid");
    }
    if (rgb.shape[0] !== phiI.shape[0] || rgb.shape[1] !== thetaI.shape[0] || rgb.shape[2] !== 3) throw new RuntimeError("RGLFile: rgb field missing or invalid");

    const phiSize = phiI.shape[0]!;
    const thetaSize = thetaI.shape[0]!;
    const sigmaSize: [number, number] = [sigma.shape[1]!, sigma.shape[0]!];
    const ndfSize: [number, number] = [ndf.shape[1]!, ndf.shape[0]!];
    const vndfSize: [number, number] = [vndf.shape[3]!, vndf.shape[2]!];
    const lumiSize: [number, number] = [luminance.shape[3]!, luminance.shape[2]!];
    const tooLarge = [phiSize, thetaSize, ...sigmaSize, ...ndfSize, ...vndfSize, ...lumiSize].some((v) => v > kRGLMaxResolution);
    if (tooLarge) throw new RuntimeError(`RGLFile: measurement resolution too large in '${name}'`);

    const vndfDist = buildSamplableDistribution4D(vndf.data as Float32Array, [phiSize, thetaSize, vndfSize[0], vndfSize[1]]);
    const lumiDist = buildSamplableDistribution4D(luminance.data as Float32Array, [phiSize, thetaSize, lumiSize[0], lumiSize[1]]);

    return {
        name,
        description: description ? new TextDecoder().decode(description.data as Uint8Array) : "",
        // Mirrors RGLFile::validate: a single azimuth means an isotropic material.
        isotropic: phiSize <= 2,
        thetaI: thetaI.data as Float32Array,
        phiI: phiI.data as Float32Array,
        sigmaSize,
        sigma: sigma.data as Float32Array,
        ndfSize,
        ndf: ndf.data as Float32Array,
        vndfSize,
        vndf: vndfDist.pdf,
        lumiSize,
        luminance: lumiDist.pdf,
        rgb: rgb.data as Float32Array,
        vndfMarginal: vndfDist.marginal,
        vndfConditional: vndfDist.conditional,
        lumiMarginal: lumiDist.marginal,
        lumiConditional: lumiDist.conditional,
    };
}

/** Fetches and parses an RGL `.bsdf` (web divergence, docs §9: native reads it from disk). */
export async function loadRGLFile(url: string): Promise<RGLMeasurement> {
    const res = await fetch(url);
    if (!res.ok) throw new RuntimeError(`RGLMaterial: failed to fetch '${url}' (${res.status})`);
    const name = url.split("/").pop()!.replace(/\.[^.]*$/, "");
    const measurement = parseRGLFile(await res.arrayBuffer(), name);
    // RGLMaterial::prepareAlbedoLUT: a cached RGBA32Float 256x1 table beside the file is used as is.
    try {
        const lut = await fetch(url.replace(/\.[^.]*$/, ".dds"));
        if (lut.ok) {
            const { ImageIO } = await import("../../Utils/Image/ImageIO.js");
            const { ResourceFormat } = await import("../../Core/API/Formats.js");
            const bitmap = ImageIO.loadBitmapFromDDS(new Uint8Array(await lut.arrayBuffer()));
            if (bitmap.format === ResourceFormat.RGBA32Float && bitmap.width === kRGLAlbedoLUTSize && bitmap.height === 1) {
                measurement.albedoLUT = new Float32Array(bitmap.data.buffer.slice(bitmap.data.byteOffset, bitmap.data.byteOffset + kRGLAlbedoLUTSize * 16));
            }
        }
    } catch {
        /* no cached table: computed with the scene */
    }
    return measurement;
}
