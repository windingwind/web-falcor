/**
 * Mirrors Falcor/Source/Tools/ImageCompare: loads two images as RGBA32F the way the
 * tool's direct FreeImage calls do (ConvertToRGBAF: 8/16-bit unorm normalized, no sRGB
 * decode, alpha 1 when absent, 3-channel floats clamped to [0, 1]; PFM rows y-flipped,
 * which native Bitmap corrects but the tool does), averages a per-pixel error metric
 * over RGB (or RGBA) and optionally writes an error heat map. Verified against the
 * native tool. `scripts/image-compare.ts` is the command-line front end.
 */

import { Bitmap, BitmapExportFlags, BitmapFileFormat } from "./Bitmap.js";
import { FormatType, ResourceFormat, getFormatChannelCount, getFormatType, getNumChannelBits } from "../../Core/API/Formats.js";
import { float16ToFloat32 } from "../Math/Float16.js";
import { RuntimeError } from "../../Core/Error.js";

/** An RGBA32F image, rows top-down. */
export interface CompareImage {
    width: number;
    height: number;
    data: Float32Array;
}

/** Mirrors Image::loadFromFile (from encoded bytes). */
export async function loadCompareImage(bytes: Uint8Array, name: string): Promise<CompareImage> {
    const bmp = await Bitmap.createFromBytes(bytes, name, true);
    if (!bmp) throw new RuntimeError("Cannot read image");
    const { width, height, format } = bmp;
    const type = getFormatType(format);
    const channels = getFormatChannelCount(format);
    const bits = getNumChannelBits(format, 0);
    const buf = bmp.data.buffer.slice(bmp.data.byteOffset, bmp.data.byteOffset + bmp.data.byteLength);
    let read: (i: number) => number;
    if (type === FormatType.Float) {
        const f32 = bits === 32 ? new Float32Array(buf) : null;
        const f16 = bits === 16 ? new Uint16Array(buf) : null;
        read = (i) => (f32 ? f32[i]! : float16ToFloat32(f16![i]!));
    } else if (bits === 16) {
        const u16 = new Uint16Array(buf);
        read = (i) => u16[i]! / 65535;
    } else if (bits === 8) {
        const u8 = new Uint8Array(buf);
        read = (i) => u8[i]! / 255;
    } else {
        throw new RuntimeError(`Cannot convert ${ResourceFormat[format]} to RGBA float format`);
    }
    const bgr = ResourceFormat[format]!.startsWith("BGR");
    const hasAlpha = channels === 4 && !ResourceFormat[format]!.includes("X");
    // ConvertToRGBAF clamps float images without alpha (RGBAF sources are cloned);
    // FreeImage loads PFM and HDR files as RGBF.
    const ext = name.toLowerCase().slice(name.lastIndexOf("."));
    const rgbFloat = type === FormatType.Float && (!hasAlpha || ext === ".pfm" || ext === ".hdr");
    const clamp = rgbFloat ? (v: number) => Math.min(Math.max(v, 0), 1) : (v: number) => v;
    const flipY = ext === ".pfm";
    const data = new Float32Array(width * height * 4);
    for (let p = 0; p < width * height; p++) {
        const src = (c: number) => clamp(read(p * channels + c));
        const x = p % width;
        const y = Math.floor(p / width);
        const o = ((flipY ? height - 1 - y : y) * width + x) * 4;
        if (channels === 1) data.fill(src(0), o, o + 3);
        else for (let c = 0; c < Math.min(channels, 3); c++) data[o + c] = src(bgr && channels >= 3 ? 2 - c : c);
        data[o + 3] = hasAlpha ? src(3) : 1;
    }
    return { width, height, data };
}

/** Float square, as native's sqr<float>. */
const sqr = (x: number) => Math.fround(x * x);
/** Per-pixel metrics over `count` channels (float differences and squares, double sums, as native). */
type Metric = (a: Float32Array, b: Float32Array, o: number, count: number) => number;
const kMetrics: { name: string; desc: string; metric: Metric }[] = [
    {
        name: "mse",
        desc: "Mean Squared Error",
        metric: (a, b, o, n) => {
            let e = 0;
            for (let i = 0; i < n; i++) e += sqr(Math.fround(a[o + i]! - b[o + i]!));
            return e / n;
        },
    },
    {
        name: "rmse",
        desc: "Relative Mean Squared Error",
        metric: (a, b, o, n) => {
            let e = 0;
            for (let i = 0; i < n; i++) e += sqr(Math.fround(a[o + i]! - b[o + i]!)) / (sqr(a[o + i]!) + 1e-3);
            return e / n;
        },
    },
    {
        // Native computes fabs(sqr(a - b)) here, i.e. the squared error; kept as is.
        name: "mae",
        desc: "Mean Absolute Error",
        metric: (a, b, o, n) => {
            let e = 0;
            for (let i = 0; i < n; i++) e += Math.abs(sqr(Math.fround(a[o + i]! - b[o + i]!)));
            return e / n;
        },
    },
    {
        name: "mape",
        desc: "Mean Absolute Percentage Error",
        metric: (a, b, o, n) => {
            let e = 0;
            for (let i = 0; i < n; i++) e += Math.abs(Math.fround(a[o + i]! - b[o + i]!) / (a[o + i]! + 1e-3));
            return (100 * e) / n;
        },
    },
];

export const errorMetrics: readonly { name: string; desc: string }[] = kMetrics.map(({ name, desc }) => ({ name, desc }));

/** Mirrors compare<Metric>: the mean per-pixel error, and the per-pixel errors for a heat map. */
export function compareImages(a: CompareImage, b: CompareImage, metricName = "mse", alpha = false): { error: number; errorMap: Float32Array } {
    const metric = kMetrics.find((m) => m.name === metricName);
    if (!metric) throw new RuntimeError(`Unknown error metric '${metricName}'.`);
    if (a.width !== b.width || a.height !== b.height) throw new RuntimeError("Cannot compare images with different resolutions.");
    const count = a.width * a.height;
    const errorMap = new Float32Array(count);
    let sum = 0;
    for (let p = 0; p < count; p++) {
        const e = metric.metric(a.data, b.data, p * 4, alpha ? 4 : 3);
        errorMap[p] = e;
        sum += e;
    }
    return { error: sum / count, errorMap };
}

/** Mirrors generateHeatMap: blue-teal-green-yellow-red over the error range. */
export function generateHeatMap(width: number, height: number, errorMap: Float32Array): CompareImage {
    const colors = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
    ];
    let min = Infinity;
    let max = -Infinity;
    for (const e of errorMap) {
        min = Math.min(min, e);
        max = Math.max(max, e);
    }
    // Float arithmetic, as native.
    const f = Math.fround;
    const range = Math.max(f(1e-5), f(max - min));
    const data = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
        const t = Math.min(Math.max(f(f(errorMap[i]! - min) / range), 0), 1);
        const c = Math.min(Math.max(Math.floor(f(t * 4)), 0), 3);
        const u = f(f(t * 4) - c);
        for (let k = 0; k < 3; k++) data[i * 4 + k] = f(colors[c]![k]! + f(u * f(colors[c + 1]![k]! - colors[c]![k]!)));
        data[i * 4 + 3] = 1;
    }
    return { width, height, data };
}

/** Mirrors Image::saveToFile: EXR (half) / PFM as float, PNG/others as 8-bit (clamped, truncated); alpha only for EXR/PNG. */
export async function encodeCompareImage(image: CompareImage, fileName: string, writeAlpha = true): Promise<Uint8Array> {
    const ext = fileName.slice(fileName.lastIndexOf(".") + 1).toLowerCase();
    const format = Bitmap.getFormatFromFileExtension(ext);
    if (format === undefined) throw new RuntimeError("Unknown image format");
    const alpha = writeAlpha && (format === BitmapFileFormat.ExrFile || format === BitmapFileFormat.PngFile);
    const flags = alpha ? BitmapExportFlags.ExportAlpha : BitmapExportFlags.None;
    // FreeImage's default EXR save writes compressed halves (PIZ natively, ZIP here).
    if (format === BitmapFileFormat.ExrFile) return Bitmap.saveImage(image.width, image.height, format, flags, ResourceFormat.RGBA32Float, true, image.data);
    if (format === BitmapFileFormat.PfmFile) {
        // FreeImage writes PFM rows top to bottom (the reverse of what Bitmap writes).
        const rowLength = image.width * 4;
        const flipped = new Float32Array(image.data.length);
        for (let y = 0; y < image.height; y++) flipped.set(image.data.subarray(y * rowLength, (y + 1) * rowLength), (image.height - 1 - y) * rowLength);
        return Bitmap.saveImage(image.width, image.height, format, BitmapExportFlags.None, ResourceFormat.RGBA32Float, true, flipped);
    }
    const rgba8 = Uint8Array.from(image.data, (v) => Math.min(Math.max(Math.trunc(v * 255), 0), 255));
    return Bitmap.saveImage(image.width, image.height, format, flags, ResourceFormat.RGBA8Unorm, true, rgba8);
}
