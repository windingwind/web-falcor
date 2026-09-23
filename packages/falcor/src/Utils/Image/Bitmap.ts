/**
 * Mirrors Falcor/Utils/Image/Bitmap: an image in CPU memory with a
 * ResourceFormat, loaded from and saved to the formats FreeImage handles.
 *
 * Native decodes through FreeImage; the web uses its own PNG/TGA/HDR/PFM/EXR
 * codecs and the browser's decoders for JPEG/BMP/GIF/WebP, then applies
 * FreeImage's bit-depth -> format rules (24-bit becomes BGRX8Unorm, 96-bit
 * float becomes RGBA32Float, ...). There is no file system, so file paths
 * are URLs (loading) or download names (saving), and saveImage returns the
 * encoded bytes.
 */

import { RuntimeError } from "../../Core/Error.js";
import { FormatType, ResourceFormat, getFormatBytesPerBlock, getFormatChannelCount, getFormatType, getNumChannelBits } from "../../Core/API/Formats.js";
import { Logger } from "../Logger.js";
import { float16ToFloat32, float32ToFloat16 } from "../Math/Float16.js";
import { decodeExr, readExrChannels } from "./EXRDecoder.js";
import { encodeExr, encodeExrZip } from "./EXREncoder.js";
import { decodeHdr } from "./HDRDecoder.js";
import { decodePfm, isPfm } from "./PFMDecoder.js";
import { decodePng, encodePng, isGrayRamp, isPng } from "./PNGCodec.js";
import { decodeTGA } from "./TGADecoder.js";

export enum BitmapExportFlags {
    None = 0,
    /** Save alpha channel as well. */
    ExportAlpha = 1 << 0,
    /** Try to store in a lossy format. */
    Lossy = 1 << 1,
    /** Prefer faster load to a more compact file size. */
    Uncompressed = 1 << 2,
    /** Use half-float instead of float when writing EXRs. */
    ExrFloat16 = 1 << 3,
}

export enum BitmapImportFlags {
    None = 0,
    /** Convert HDR images to 16-bit float per channel on import. */
    ConvertToFloat16 = 1 << 0,
}

/** Order matches native (getFormatFromFileExtension indexes by it). */
export enum BitmapFileFormat {
    PngFile,
    JpegFile,
    TgaFile,
    BmpFile,
    PfmFile,
    ExrFile,
    DdsFile,
}

const kExtensions = ["png", "jpg", "tga", "bmp", "pfm", "exr", "dds"];

export interface FileDialogFilter {
    ext: string;
    desc: string;
}

/** Half, and 16/32-bit integer formats, save as float (native isConvertibleToRGBA32Float). */
function isConvertibleToRGBA32Float(format: ResourceFormat): boolean {
    const type = getFormatType(format);
    const bits = getNumChannelBits(format, 0);
    return (type === FormatType.Float && bits === 16) || ((type === FormatType.Uint || type === FormatType.Sint) && bits >= 16);
}

/** Mirrors convertToRGBA32Float: halves widen, integers normalize by their max; alpha defaults to 1. */
function convertToRGBA32Float(format: ResourceFormat, width: number, height: number, data: Uint8Array): Float32Array {
    const type = getFormatType(format);
    const channels = getFormatChannelCount(format);
    const bits = getNumChannelBits(format, 0);
    const n = width * height;
    const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    let read: (i: number) => number;
    if (type === FormatType.Float && bits === 16) {
        const src = new Uint16Array(buf);
        read = (i) => float16ToFloat32(src[i]!);
    } else {
        const src = bits === 16 ? (type === FormatType.Uint ? new Uint16Array(buf) : new Int16Array(buf)) : type === FormatType.Uint ? new Uint32Array(buf) : new Int32Array(buf);
        const max = bits === 16 ? (type === FormatType.Uint ? 65535 : 32767) : type === FormatType.Uint ? 4294967295 : 2147483647;
        read = (i) => Math.fround(Math.fround(src[i]!) / Math.fround(max));
    }
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < channels; c++) out[i * 4 + c] = read(i * channels + c);
        if (channels < 4) out[i * 4 + 3] = 1;
    }
    return out;
}

function flipRows(data: Uint8Array, rowPitch: number, height: number): Uint8Array {
    const out = new Uint8Array(data.length);
    for (let y = 0; y < height; y++) out.set(data.subarray(y * rowPitch, (y + 1) * rowPitch), (height - 1 - y) * rowPitch);
    return out;
}

/** JPEG component count from the first SOF marker (1 = grayscale). */
function jpegComponents(bytes: Uint8Array): number {
    let p = 2;
    while (p + 9 < bytes.length) {
        if (bytes[p] !== 0xff) return 3;
        const marker = bytes[p + 1]!;
        const len = (bytes[p + 2]! << 8) | bytes[p + 3]!;
        if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) return bytes[p + 9]!;
        p += 2 + len;
    }
    return 3;
}

export class Bitmap {
    readonly rowPitch: number;
    readonly size: number;

    protected constructor(
        readonly width: number,
        readonly height: number,
        readonly format: ResourceFormat,
        readonly data: Uint8Array,
    ) {
        this.rowPitch = width * getFormatBytesPerBlock(format);
        this.size = this.rowPitch * height;
        if (data.byteLength < this.size) throw new RuntimeError(`Bitmap: ${data.byteLength} bytes given, ${this.size} needed`);
    }

    /** Mirrors Bitmap::create (copies the data). */
    static create(width: number, height: number, format: ResourceFormat, data: ArrayBufferView): Bitmap {
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        return new Bitmap(width, height, format, bytes.slice(0, width * getFormatBytesPerBlock(format) * height));
    }

    getData(): Uint8Array {
        return this.data;
    }
    getWidth(): number {
        return this.width;
    }
    getHeight(): number {
        return this.height;
    }
    getFormat(): ResourceFormat {
        return this.format;
    }
    getRowPitch(): number {
        return this.rowPitch;
    }
    getSize(): number {
        return this.size;
    }

    /** Mirrors Bitmap::createFromFile; `path` is a URL. Returns null (with a warning) on failure. */
    static async createFromFile(path: string, isTopDown: boolean, importFlags = BitmapImportFlags.None): Promise<Bitmap | null> {
        let bytes: Uint8Array;
        try {
            const res = await fetch(path);
            if (!res.ok) {
                Logger.warning(`Error when loading image file. File '${path}' does not exist.`);
                return null;
            }
            bytes = new Uint8Array(await res.arrayBuffer());
        } catch {
            Logger.warning(`Error when loading image file. File '${path}' does not exist.`);
            return null;
        }
        return Bitmap.createFromBytes(bytes, path, isTopDown, importFlags);
    }

    /** createFromFile over bytes already in memory; `name` supplies the extension fallback. */
    static async createFromBytes(bytes: Uint8Array, name: string, isTopDown: boolean, importFlags = BitmapImportFlags.None): Promise<Bitmap | null> {
        const warn = (msg: string) => {
            Logger.warning(`Error when loading image file from '${name}': ${msg}`);
            return null;
        };
        try {
            const decoded = await decode(bytes, name, importFlags);
            if (!decoded) return warn("Image type unknown");
            const { width, height, format, data } = decoded;
            if (width === 0 || height === 0) return warn("Invalid image");
            const bmp = new Bitmap(width, height, format, data);
            // Decoders produce top-down rows; FreeImage's native order is bottom-up.
            return isTopDown ? bmp : new Bitmap(width, height, format, flipRows(data, bmp.rowPitch, height));
        } catch (e) {
            return warn(`Can't read image file (${(e as Error).message})`);
        }
    }

    /** Mirrors Bitmap::getFormatFromFileExtension (extension without the dot). */
    static getFormatFromFileExtension(ext: string): BitmapFileFormat {
        const i = kExtensions.indexOf(ext);
        if (i < 0) throw new RuntimeError(`Can't find a matching format for file extension '${ext}'.`);
        return i as BitmapFileFormat;
    }

    /** Mirrors Bitmap::getFileDialogFilters. */
    static getFileDialogFilters(format = ResourceFormat.Unknown): FileDialogFilter[] {
        let showHdr = true;
        let showLdr = true;
        if (format !== ResourceFormat.Unknown) {
            showHdr = getFormatType(format) === FormatType.Float || isConvertibleToRGBA32Float(format);
            showLdr = !showHdr;
        }
        const filters: FileDialogFilter[] = [];
        if (showHdr) filters.push({ ext: "exr", desc: "High Dynamic Range" }, { ext: "pfm", desc: "Portable Float Map" }, { ext: "hdr", desc: "Radiance HDR" });
        if (showLdr)
            filters.push(
                { ext: "png", desc: "Portable Network Graphics" },
                { ext: "jpg", desc: "JPEG" },
                { ext: "bmp", desc: "Bitmap Image File" },
                { ext: "tga", desc: "Truevision Graphics Adapter" },
            );
        filters.push({ ext: "dds", desc: "DirectDraw Surface" });
        if (format === ResourceFormat.Unknown) filters.push({ ext: "hdr", desc: "High Dynamic Range" });
        return filters;
    }

    /** Mirrors Bitmap::getFileExtFromResourceFormat. */
    static getFileExtFromResourceFormat(format: ResourceFormat): string {
        return Bitmap.getFileDialogFilters(format)[0]!.ext;
    }

    /**
     * Mirrors Bitmap::saveImage, returning the encoded file. Unlike native, the
     * caller's data is not modified (native swaps RGBA8 to BGRA8 in place).
     */
    static async saveImage(
        width: number,
        height: number,
        fileFormat: BitmapFileFormat,
        exportFlags: BitmapExportFlags,
        resourceFormat: ResourceFormat,
        isTopDown: boolean,
        data: ArrayBufferView,
    ): Promise<Uint8Array> {
        const has = (f: BitmapExportFlags) => (exportFlags & f) !== 0;
        if (fileFormat === BitmapFileFormat.DdsFile) throw new RuntimeError("Cannot save DDS files. Use ImageIO instead.");
        if (has(BitmapExportFlags.Uncompressed) && has(BitmapExportFlags.Lossy)) throw new RuntimeError("Incompatible flags: lossy cannot be combined with uncompressed.");
        if (has(BitmapExportFlags.ExrFloat16) && (!has(BitmapExportFlags.Uncompressed) || fileFormat !== BitmapFileFormat.ExrFile))
            throw new RuntimeError("Incompatible flags: EXR float16 can only be set for uncompressed EXR files.");
        const exportAlpha = has(BitmapExportFlags.ExportAlpha);
        let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        let bytesPerPixel = getFormatBytesPerBlock(resourceFormat);

        if (fileFormat === BitmapFileFormat.PfmFile || fileFormat === BitmapFileFormat.ExrFile) {
            if (isConvertibleToRGBA32Float(resourceFormat)) {
                const f = convertToRGBA32Float(resourceFormat, width, height, bytes);
                bytes = new Uint8Array(f.buffer);
                bytesPerPixel = 16;
            } else if (bytesPerPixel !== 16 && bytesPerPixel !== 12) {
                throw new RuntimeError("Only support for 32-bit/channel RGB/RGBA or 16-bit RGBA images as PFM/EXR files.");
            }
            if (fileFormat === BitmapFileFormat.PfmFile) {
                if (has(BitmapExportFlags.Lossy)) throw new RuntimeError("PFM does not support lossy compression mode.");
                if (exportAlpha) throw new RuntimeError("PFM does not support alpha channel.");
            }
            if (exportAlpha && bytesPerPixel !== 16) throw new RuntimeError("Requesting to export alpha-channel to EXR file, but the resource doesn't have an alpha-channel");
            // Native reads float rows top-down here regardless of isTopDown.
            const src = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + width * height * bytesPerPixel));
            const stride = bytesPerPixel / 4;
            const rgba = new Float32Array(width * height * 4);
            for (let i = 0; i < width * height; i++) {
                for (let c = 0; c < 3; c++) rgba[i * 4 + c] = src[i * stride + c]!;
                rgba[i * 4 + 3] = stride === 4 ? src[i * stride + 3]! : 1;
            }
            if (fileFormat === BitmapFileFormat.PfmFile) return encodePfm(rgba, width, height);
            // FreeImage flags: Uncompressed -> float (or half) without compression; otherwise half
            // (EXR_DEFAULT's PIZ, and B44 for Lossy, become lossless ZIP here).
            const uncompressed = has(BitmapExportFlags.Uncompressed);
            const opts = { alpha: exportAlpha, half: !uncompressed || has(BitmapExportFlags.ExrFloat16) };
            return uncompressed ? encodeExr(rgba, width, height, opts) : encodeExrZip(rgba, width, height, opts);
        }

        // 8-bit path: FreeImage takes 32-bit pixels as BGRA, 8-bit as gray.
        const n = width * height;
        const rgba = new Uint8Array(n * 4);
        const rgbaOrder = resourceFormat === ResourceFormat.RGBA8Unorm || resourceFormat === ResourceFormat.RGBA8Snorm || resourceFormat === ResourceFormat.RGBA8UnormSrgb;
        if (bytesPerPixel === 4) {
            for (let i = 0; i < n; i++) {
                const [r, b] = rgbaOrder ? [bytes[i * 4]!, bytes[i * 4 + 2]!] : [bytes[i * 4 + 2]!, bytes[i * 4]!];
                rgba.set([r, bytes[i * 4 + 1]!, b, rgbaOrder && !exportAlpha ? 255 : bytes[i * 4 + 3]!], i * 4);
            }
        } else if (bytesPerPixel === 1) {
            for (let i = 0; i < n; i++) rgba.set([bytes[i]!, bytes[i]!, bytes[i]!, 255], i * 4);
        } else {
            throw new RuntimeError(`Bitmap.saveImage: ${ResourceFormat[resourceFormat]} can't be saved as an 8-bit image.`);
        }
        const top = isTopDown ? rgba : flipRows(rgba, width * 4, height);
        const keepAlpha = exportAlpha && fileFormat !== BitmapFileFormat.JpegFile;

        const warnings: string[] = [];
        let out: Uint8Array;
        switch (fileFormat) {
            case BitmapFileFormat.JpegFile: {
                if (exportAlpha) warnings.push("JPEG format does not support alpha channel.");
                // JPEG_QUALITYSUPERB unless lossy (FreeImage's default quality is 75).
                const quality = has(BitmapExportFlags.Lossy) && !has(BitmapExportFlags.Uncompressed) ? 0.75 : 1;
                const opaque = top.slice();
                for (let i = 0; i < n; i++) opaque[i * 4 + 3] = 255;
                const canvas = new OffscreenCanvas(width, height);
                canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(opaque.buffer), width, height), 0, 0);
                out = new Uint8Array(await (await canvas.convertToBlob({ type: "image/jpeg", quality })).arrayBuffer());
                break;
            }
            case BitmapFileFormat.PngFile: {
                if (has(BitmapExportFlags.Lossy)) warnings.push("PNG format does not support lossy compression mode.");
                out = await encodePng(keepAlpha ? top : dropAlpha(top), width, height, keepAlpha ? 4 : 3, !has(BitmapExportFlags.Uncompressed));
                break;
            }
            case BitmapFileFormat.TgaFile:
                if (has(BitmapExportFlags.Lossy)) warnings.push("TGA format does not support lossy compression mode.");
                out = encodeTga(top, width, height, keepAlpha);
                break;
            case BitmapFileFormat.BmpFile:
                if (has(BitmapExportFlags.Lossy)) warnings.push("BMP format does not support lossy compression mode.");
                if (exportAlpha) warnings.push("BMP format does not support alpha channel.");
                out = encodeBmp(top, width, height, keepAlpha);
                break;
            default:
                throw new RuntimeError(`Bitmap.saveImage: unsupported file format ${fileFormat}`);
        }
        if (warnings.length > 0) Logger.warning(`Bitmap::saveImage: ${warnings.join(" ")}`);
        return out;
    }
}

function dropAlpha(rgba: Uint8Array): Uint8Array {
    const n = rgba.length / 4;
    const rgb = new Uint8Array(n * 3);
    for (let i = 0; i < n; i++) rgb.set(rgba.subarray(i * 4, i * 4 + 3), i * 3);
    return rgb;
}

/** Uncompressed true-colour TGA, bottom-left origin (as FreeImage writes it). */
function encodeTga(rgba: Uint8Array, width: number, height: number, alpha: boolean): Uint8Array {
    const bpp = alpha ? 4 : 3;
    const out = new Uint8Array(18 + width * height * bpp);
    out[2] = 2;
    new DataView(out.buffer).setUint16(12, width, true);
    new DataView(out.buffer).setUint16(14, height, true);
    out[16] = bpp * 8;
    out[17] = alpha ? 8 : 0;
    let o = 18;
    for (let y = height - 1; y >= 0; y--)
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            out[o++] = rgba[i + 2]!;
            out[o++] = rgba[i + 1]!;
            out[o++] = rgba[i]!;
            if (alpha) out[o++] = rgba[i + 3]!;
        }
    return out;
}

/** Uncompressed BMP (BITMAPINFOHEADER), bottom-up rows padded to 4 bytes. */
function encodeBmp(rgba: Uint8Array, width: number, height: number, alpha: boolean): Uint8Array {
    const bpp = alpha ? 4 : 3;
    const rowBytes = (width * bpp + 3) & ~3;
    const size = 54 + rowBytes * height;
    const out = new Uint8Array(size);
    const v = new DataView(out.buffer);
    out[0] = 0x42;
    out[1] = 0x4d;
    v.setUint32(2, size, true);
    v.setUint32(10, 54, true);
    v.setUint32(14, 40, true);
    v.setInt32(18, width, true);
    v.setInt32(22, height, true);
    v.setUint16(26, 1, true);
    v.setUint16(28, bpp * 8, true);
    v.setUint32(34, rowBytes * height, true);
    for (let y = 0; y < height; y++) {
        let o = 54 + (height - 1 - y) * rowBytes;
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            out[o++] = rgba[i + 2]!;
            out[o++] = rgba[i + 1]!;
            out[o++] = rgba[i]!;
            if (alpha) out[o++] = rgba[i + 3]!;
        }
    }
    return out;
}

/** "PF" RGB float, little endian (negative scale), rows bottom to top. */
function encodePfm(rgba: Float32Array, width: number, height: number): Uint8Array {
    const header = new TextEncoder().encode(`PF\n${width} ${height}\n-1.000000\n`);
    const out = new Uint8Array(header.length + width * height * 12);
    out.set(header);
    const v = new DataView(out.buffer, header.length);
    let o = 0;
    for (let y = height - 1; y >= 0; y--)
        for (let x = 0; x < width; x++)
            for (let c = 0; c < 3; c++) {
                v.setFloat32(o, rgba[(y * width + x) * 4 + c]!, true);
                o += 4;
            }
    return out;
}

interface Decoded {
    width: number;
    height: number;
    format: ResourceFormat;
    data: Uint8Array;
}

/** RGBA float pixels -> RGBA32Float, or RGBA16Float under ConvertToFloat16. */
function floatImage(width: number, height: number, rgba: Float32Array, half: boolean): Decoded {
    if (!half) return { width, height, format: ResourceFormat.RGBA32Float, data: new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength) };
    const h = new Uint16Array(rgba.length);
    for (let i = 0; i < rgba.length; i++) h[i] = float32ToFloat16(rgba[i]!);
    return { width, height, format: ResourceFormat.RGBA16Float, data: new Uint8Array(h.buffer) };
}

/** 8-bit gray/RGB/RGBA pixels -> FreeImage's 8bpp (R8), 24bpp (BGRX8) or 32bpp (BGRA8) layout. */
function ldrImage(width: number, height: number, src: Uint8Array | Uint8ClampedArray, channels: number, keepAlpha: boolean): Decoded {
    const n = width * height;
    if (channels === 1) return { width, height, format: ResourceFormat.R8Unorm, data: Uint8Array.from(src.subarray(0, n)) };
    const data = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
        const [r, g, b, a] = channels === 2 ? [src[i * 2]!, src[i * 2]!, src[i * 2]!, src[i * 2 + 1]!] : [src[i * channels]!, src[i * channels + 1]!, src[i * channels + 2]!, channels === 4 ? src[i * 4 + 3]! : 255];
        data.set([b, g, r, keepAlpha ? a : 255], i * 4);
    }
    return { width, height, format: keepAlpha ? ResourceFormat.BGRA8Unorm : ResourceFormat.BGRX8Unorm, data };
}

async function decodeWithBrowser(bytes: Uint8Array): Promise<{ width: number; height: number; rgba: Uint8ClampedArray }> {
    const bitmap = await createImageBitmap(new Blob([bytes as BlobPart]), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    return { width: bitmap.width, height: bitmap.height, rgba };
}

async function decode(bytes: Uint8Array, name: string, importFlags: BitmapImportFlags): Promise<Decoded | null> {
    let half = (importFlags & BitmapImportFlags.ConvertToFloat16) !== 0;
    const ext = name.split(/[?#]/)[0]!.split(".").pop()!.toLowerCase();
    const magic = (s: string) => s.split("").every((c, i) => bytes[i] === c.charCodeAt(0));

    if (isPng(bytes)) {
        const png = await decodePng(bytes);
        const { width, height } = png;
        if (png.bitDepth === 16) {
            const src = png.data as Uint16Array;
            if (png.channels === 1) return { width, height, format: ResourceFormat.R16Unorm, data: new Uint8Array(src.buffer) };
            // 48-bit RGB and 16-bit gray+alpha widen to RGBA16Unorm (FreeImage_ConvertToRGBA16).
            const out = new Uint16Array(width * height * 4);
            for (let i = 0; i < width * height; i++) {
                const c = png.channels;
                const px = c === 2 ? [src[i * 2]!, src[i * 2]!, src[i * 2]!, src[i * 2 + 1]!] : [src[i * c]!, src[i * c + 1]!, src[i * c + 2]!, c === 4 ? src[i * 4 + 3]! : 65535];
                out.set(px, i * 4);
            }
            return { width, height, format: ResourceFormat.RGBA16Unorm, data: new Uint8Array(out.buffer) };
        }
        // Palettes and gray+alpha load as 32 bits; RGB as 24 (-> BGRX).
        const keepAlpha = png.palettized || png.channels === 2 || png.channels === 4;
        return ldrImage(width, height, png.data as Uint8Array, png.channels, keepAlpha);
    }
    if (magic("#?")) {
        const hdr = decodeHdr(bytes);
        return floatImage(hdr.width, hdr.height, hdr.data, half);
    }
    if (isPfm(bytes)) {
        // "Pf" is one float channel: R32Float (FreeImage reports its 32bpp as BGRA8).
        const pfm = decodePfm(bytes);
        if (bytes[1] === 0x66) {
            const r = new Float32Array(pfm.width * pfm.height);
            for (let i = 0; i < r.length; i++) r[i] = pfm.data[i * 4]!;
            return { width: pfm.width, height: pfm.height, format: ResourceFormat.R32Float, data: new Uint8Array(r.buffer) };
        }
        return floatImage(pfm.width, pfm.height, pfm.data, half);
    }
    if (bytes[0] === 0x76 && bytes[1] === 0x2f && bytes[2] === 0x31 && bytes[3] === 0x01) {
        // All-half EXRs load as RGBA16Float (native checks the channel types).
        if (readExrChannels(bytes).every((c) => c.type === 1)) half = true;
        const exr = decodeExr(bytes.slice().buffer);
        return floatImage(exr.width, exr.height, exr.data, half);
    }
    if (bytes[0] === 0xff && bytes[1] === 0xd8) {
        const { width, height, rgba } = await decodeWithBrowser(bytes);
        return ldrImage(width, height, jpegComponents(bytes) === 1 ? rgba.filter((_, i) => i % 4 === 0) : rgba, jpegComponents(bytes) === 1 ? 1 : 4, false);
    }
    if (magic("BM")) {
        const bitCount = bytes[28]! | (bytes[29]! << 8);
        const { width, height, rgba } = await decodeWithBrowser(bytes);
        if (bitCount <= 8) {
            // BGRX palette after the info header; a gray ramp loads as R8 (FIC_MINISBLACK).
            const infoSize = bytes[14]! | (bytes[15]! << 8);
            const used = (bytes[46]! | (bytes[47]! << 8)) || 1 << bitCount;
            const rgb = new Uint8Array(used * 3);
            for (let i = 0; i < used; i++) rgb.set([bytes[14 + infoSize + i * 4 + 2]!, bytes[14 + infoSize + i * 4 + 1]!, bytes[14 + infoSize + i * 4]!], i * 3);
            if (isGrayRamp(rgb, 1 << bitCount)) return ldrImage(width, height, rgba.filter((_, i) => i % 4 === 0), 1, false);
        }
        return ldrImage(width, height, rgba, 4, bitCount === 32 || bitCount <= 8);
    }
    if (magic("GIF8") || (magic("RIFF") && String.fromCharCode(...bytes.subarray(8, 12)) === "WEBP")) {
        const { width, height, rgba } = await decodeWithBrowser(bytes);
        return ldrImage(width, height, rgba, 4, true);
    }
    if (magic("DDS ")) throw new RuntimeError("DDS files load through the DDS loader (ImageIO), not Bitmap");
    if (ext === "tga") {
        // TGA has no signature: the extension decides, as FreeImage_GetFIFFromFilename does.
        const tga = decodeTGA(bytes.slice().buffer);
        const pixelDepth = bytes[16]!;
        const type = bytes[2]! & ~8;
        if (type === 3) return ldrImage(tga.width, tga.height, tga.rgba.filter((_, i) => i % 4 === 0), 1, false);
        return ldrImage(tga.width, tga.height, tga.rgba, 4, pixelDepth === 32 || type === 1);
    }
    return null;
}
