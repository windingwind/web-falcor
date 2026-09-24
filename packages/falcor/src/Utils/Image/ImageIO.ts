/**
 * Mirrors Falcor/Utils/Image/ImageIO: DDS load and save. Saving compresses with
 * BCEncoder in place of NVTT (§9: valid BC blocks, not NVTT's bit pattern) and returns
 * the file bytes (the web has no file system; callers download or cache them).
 *
 * Like native: a surface is converted to RGBA (three-channel images get alpha 1, 8-bit
 * values are compressed as stored), two-channel images only take BC5, mipmaps are box
 * filtered, and compressing with generated mips clamps the base size to a multiple of 4.
 * BC6H and BC7 files carry the DX10 header, as do sRGB formats. Unlike native, an
 * already-compressed texture saved with mode None keeps its blocks instead of being
 * decompressed and recompressed.
 */

import type { Device } from "../../Core/API/Device.js";
import { FormatType, ResourceFormat, getFormatBytesPerBlock, getFormatChannelCount, getFormatType, getNumChannelBits, isCompressedFormat } from "../../Core/API/Formats.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { Texture } from "../../Core/API/Texture.js";
import { ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { RuntimeError } from "../../Core/Error.js";
import { parseDDS } from "../../Scene/Importer/DDSLoader.js";
import { Logger } from "../Logger.js";
import { float16ToFloat32 } from "../Math/Float16.js";
import { Bitmap } from "./Bitmap.js";
import { encodeBC, type BCFormat } from "./BCEncoder.js";

/** Mirrors ImageIO::CompressionMode. */
export enum CompressionMode {
    BC1,
    BC2,
    BC3,
    BC4,
    BC5,
    BC6,
    BC7,
    None,
}

/** The BC format a mode writes (native: BC6 is NVTT's Format_BC6S). */
const kModeFormat: Record<Exclude<CompressionMode, CompressionMode.None>, { bc: BCFormat; format: ResourceFormat; srgbFormat?: ResourceFormat; dxgi: number; dxgiSrgb?: number; fourCC?: string }> = {
    [CompressionMode.BC1]: { bc: "BC1", format: ResourceFormat.BC1Unorm, srgbFormat: ResourceFormat.BC1UnormSrgb, dxgi: 71, dxgiSrgb: 72, fourCC: "DXT1" },
    [CompressionMode.BC2]: { bc: "BC2", format: ResourceFormat.BC2Unorm, srgbFormat: ResourceFormat.BC2UnormSrgb, dxgi: 74, dxgiSrgb: 75, fourCC: "DXT3" },
    [CompressionMode.BC3]: { bc: "BC3", format: ResourceFormat.BC3Unorm, srgbFormat: ResourceFormat.BC3UnormSrgb, dxgi: 77, dxgiSrgb: 78, fourCC: "DXT5" },
    [CompressionMode.BC4]: { bc: "BC4", format: ResourceFormat.BC4Unorm, dxgi: 80, fourCC: "ATI1" },
    [CompressionMode.BC5]: { bc: "BC5", format: ResourceFormat.BC5Unorm, dxgi: 83, fourCC: "ATI2" },
    [CompressionMode.BC6]: { bc: "BC6HS", format: ResourceFormat.BC6HS16, dxgi: 96 },
    [CompressionMode.BC7]: { bc: "BC7", format: ResourceFormat.BC7Unorm, srgbFormat: ResourceFormat.BC7UnormSrgb, dxgi: 98, dxgiSrgb: 99 },
};

/** Mirrors convertFormatToMode (for compressed inputs). */
function modeOfFormat(format: ResourceFormat): CompressionMode {
    for (const [mode, info] of Object.entries(kModeFormat)) if (info.format === format || info.srgbFormat === format) return Number(mode) as CompressionMode;
    if (format === ResourceFormat.BC5Snorm) return CompressionMode.BC5;
    throw new RuntimeError("No corresponding compression mode for the provided ResourceFormat.");
}

/** DXGI codes of the uncompressed formats written with a DX10 header. */
const kUncompressedDxgi: Partial<Record<ResourceFormat, number>> = {
    [ResourceFormat.RGBA32Float]: 2,
    [ResourceFormat.RGBA16Float]: 10,
    [ResourceFormat.R32Float]: 41,
    [ResourceFormat.RGBA8UnormSrgb]: 29,
};

const isSrgbFormat = (format: ResourceFormat) => getFormatType(format) === FormatType.UnormSrgb;

interface Surface {
    width: number;
    height: number;
    rgba: Float32Array;
}

/** A surface as float RGBA (native setImage: BGRA order, padded channels, alpha 1). */
function toRGBA(format: ResourceFormat, width: number, height: number, data: Uint8Array): Float32Array {
    const channels = getFormatChannelCount(format);
    const type = getFormatType(format);
    const bits = getNumChannelBits(format, 0);
    const n = width * height;
    const buf = data.buffer.slice(data.byteOffset, data.byteOffset + n * getFormatBytesPerBlock(format));
    let read: (i: number) => number;
    if (type === FormatType.Float && bits === 32) {
        const src = new Float32Array(buf);
        read = (i) => src[i]!;
    } else if (type === FormatType.Float && bits === 16) {
        const src = new Uint16Array(buf);
        read = (i) => float16ToFloat32(src[i]!);
    } else if (bits === 8 && (type === FormatType.Unorm || type === FormatType.UnormSrgb || type === FormatType.Uint)) {
        const src = new Uint8Array(buf);
        read = (i) => src[i]! / 255;
    } else if (bits === 8 && (type === FormatType.Snorm || type === FormatType.Sint)) {
        const src = new Int8Array(buf);
        read = (i) => Math.max(-1, src[i]! / 127);
    } else {
        throw new RuntimeError("Image is in an unsupported ResourceFormat.");
    }
    if (channels === 3) Logger.warning("NVTT is incompatible with three channel images. This image will be padded with a solid alpha channel.");
    const bgra = format === ResourceFormat.BGRA8Unorm || format === ResourceFormat.BGRA8UnormSrgb || format === ResourceFormat.BGRX8Unorm || format === ResourceFormat.BGRX8UnormSrgb;
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 4; c++) out[i * 4 + c] = c < channels ? read(i * channels + c) : c === 3 ? 1 : 0;
        if (bgra) [out[i * 4], out[i * 4 + 2]] = [out[i * 4 + 2]!, out[i * 4]!];
        if (format === ResourceFormat.BGRX8Unorm || format === ResourceFormat.BGRX8UnormSrgb) out[i * 4 + 3] = 1;
    }
    return out;
}

/** nvtt::Surface::buildNextMipmap(MipmapFilter_Box). */
function nextMip(s: Surface): Surface {
    const w = Math.max(1, s.width >> 1);
    const h = Math.max(1, s.height >> 1);
    const rgba = new Float32Array(w * h * 4);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const xs = [Math.min(2 * x, s.width - 1), Math.min(2 * x + 1, s.width - 1)];
            const ys = [Math.min(2 * y, s.height - 1), Math.min(2 * y + 1, s.height - 1)];
            for (let c = 0; c < 4; c++) {
                let sum = 0;
                for (const yy of ys) for (const xx of xs) sum += s.rgba[(yy * s.width + xx) * 4 + c]!;
                rgba[(y * w + x) * 4 + c] = sum / 4;
            }
        }
    }
    return { width: w, height: h, rgba };
}

/** Crops a surface to the top-left `width` x `height`. */
function crop(s: Surface, width: number, height: number): Surface {
    if (width === s.width && height === s.height) return s;
    const rgba = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) rgba.set(s.rgba.subarray(y * s.width * 4, (y * s.width + width) * 4), y * width * 4);
    return { width, height, rgba };
}

const countMipmaps = (w: number, h: number) => 32 - Math.clz32(Math.max(w, h));

/** Uncompressed payload of a surface in the output format. */
function packUncompressed(s: Surface, format: ResourceFormat): Uint8Array {
    const n = s.width * s.height;
    if (format === ResourceFormat.RGBA32Float) return new Uint8Array(s.rgba.buffer.slice(0));
    if (format === ResourceFormat.R32Float) {
        const r = new Float32Array(n);
        for (let i = 0; i < n; i++) r[i] = s.rgba[i * 4]!;
        return new Uint8Array(r.buffer);
    }
    if (format === ResourceFormat.RGBA16Float) {
        const h = new Uint16Array(n * 4);
        for (let i = 0; i < n * 4; i++) h[i] = halfBits(s.rgba[i]!);
        return new Uint8Array(h.buffer);
    }
    // BGRA8 (NVTT's default RGBA pixel layout), or RGBA8 for the DX10 sRGB variant.
    const out = new Uint8Array(n * 4);
    const bgra = format !== ResourceFormat.RGBA8UnormSrgb;
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 4; c++) out[i * 4 + (bgra && c !== 3 ? 2 - c : c)] = Math.round(Math.min(1, Math.max(0, s.rgba[i * 4 + c]!)) * 255);
    }
    return out;
}

function halfBits(v: number): number {
    const f = new Float32Array([v]);
    const x = new Uint32Array(f.buffer)[0]!;
    const sign = (x >>> 16) & 0x8000;
    const e = ((x >>> 23) & 0xff) - 112;
    if (e <= 0) return sign;
    if (e >= 31) return sign | 0x7c00;
    let h = (e << 10) | ((x >>> 13) & 0x3ff);
    const rem = x & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++;
    return sign | h;
}

interface DDSOut {
    width: number;
    height: number;
    mipCount: number;
    faces: number;
    cube: boolean;
    format: ResourceFormat;
    dxgi?: number;
    fourCC?: string;
    /** Legacy BGRA8 mask header. */
    bgra8?: boolean;
    payload: Uint8Array[];
}

/** The DDS file (DDS_HEADER + optional DDS_HEADER_DXT10 + data). */
function writeDDS(d: DDSOut): Uint8Array {
    const dx10 = d.dxgi !== undefined && d.fourCC === undefined && !d.bgra8;
    const headerSize = 128 + (dx10 ? 20 : 0);
    const dataSize = d.payload.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(headerSize + dataSize);
    const dv = new DataView(out.buffer);
    const fourCC = (s: string) => s.charCodeAt(0) | (s.charCodeAt(1) << 8) | (s.charCodeAt(2) << 16) | (s.charCodeAt(3) << 24);
    const compressed = isCompressedFormat(d.format);
    dv.setUint32(0, fourCC("DDS "), true);
    dv.setUint32(4, 124, true);
    // CAPS | HEIGHT | WIDTH | PIXELFORMAT, plus MIPMAPCOUNT and PITCH / LINEARSIZE.
    dv.setUint32(8, 0x1 | 0x2 | 0x4 | 0x1000 | (d.mipCount > 1 ? 0x20000 : 0) | (compressed ? 0x80000 : 0x8), true);
    dv.setUint32(12, d.height, true);
    dv.setUint32(16, d.width, true);
    dv.setUint32(20, compressed ? d.payload[0]!.length : d.width * getFormatBytesPerBlock(d.format), true);
    dv.setUint32(28, d.mipCount, true);
    dv.setUint32(76, 32, true);
    if (d.bgra8) {
        dv.setUint32(80, 0x40 | 0x1, true); // RGB | ALPHAPIXELS
        dv.setUint32(88, 32, true);
        dv.setUint32(92, 0x00ff0000, true);
        dv.setUint32(96, 0x0000ff00, true);
        dv.setUint32(100, 0x000000ff, true);
        dv.setUint32(104, 0xff000000, true);
    } else {
        dv.setUint32(80, 0x4, true); // FOURCC
        dv.setUint32(84, fourCC(dx10 ? "DX10" : d.fourCC!), true);
    }
    // TEXTURE, plus COMPLEX | MIPMAP for chains and COMPLEX for cubes.
    dv.setUint32(108, 0x1000 | (d.mipCount > 1 ? 0x400008 : 0) | (d.cube ? 0x8 : 0), true);
    if (d.cube) dv.setUint32(112, 0x200 | 0xfc00, true); // CUBEMAP | all faces
    if (dx10) {
        dv.setUint32(128, d.dxgi!, true);
        dv.setUint32(132, 3, true); // TEXTURE2D
        dv.setUint32(136, d.cube ? 0x4 : 0, true);
        dv.setUint32(140, d.cube ? d.faces / 6 : d.faces, true);
    }
    let offset = headerSize;
    for (const p of d.payload) {
        out.set(p, offset);
        offset += p.length;
    }
    return out;
}

/** Output format and header choice for a mode and source format. */
function outputFor(mode: CompressionMode, source: ResourceFormat): Pick<DDSOut, "format" | "dxgi" | "fourCC" | "bgra8"> {
    const srgb = isSrgbFormat(source);
    if (mode !== CompressionMode.None) {
        const info = kModeFormat[mode];
        if (srgb && info.srgbFormat !== undefined) return { format: info.srgbFormat, dxgi: info.dxgiSrgb };
        // Native: BC6/BC7 always use the DX10 container.
        return mode === CompressionMode.BC6 || mode === CompressionMode.BC7 ? { format: info.format, dxgi: info.dxgi } : { format: info.format, dxgi: info.dxgi, fourCC: info.fourCC };
    }
    const type = getFormatType(source);
    if (type === FormatType.Float) {
        if (source === ResourceFormat.R32Float) return { format: ResourceFormat.R32Float, dxgi: 41 };
        return getNumChannelBits(source, 0) === 16 ? { format: ResourceFormat.RGBA16Float, dxgi: 10 } : { format: ResourceFormat.RGBA32Float, dxgi: 2 };
    }
    if (srgb) return { format: ResourceFormat.RGBA8UnormSrgb, dxgi: kUncompressedDxgi[ResourceFormat.RGBA8UnormSrgb] };
    return { format: ResourceFormat.BGRA8Unorm, bgra8: true };
}

function encodeSurface(s: Surface, mode: CompressionMode, out: ResourceFormat): Uint8Array {
    return mode === CompressionMode.None ? packUncompressed(s, out) : encodeBC(kModeFormat[mode].bc, s.rgba, s.width, s.height);
}

export class ImageIO {
    /** Mirrors loadBitmapFromDDS: the first image (compressed formats stay compressed). */
    static loadBitmapFromDDS(bytes: Uint8Array): Bitmap {
        const image = parseDDS(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer, false);
        const level = image.levels[0]!;
        return Bitmap.create(level.width, level.height, image.format, level.data);
    }

    /** Mirrors loadTextureFromDDS: every stored mip, sRGB variant when `loadAsSrgb`. */
    static loadTextureFromDDS(device: Device, bytes: Uint8Array, loadAsSrgb: boolean): Texture {
        const image = parseDDS(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer, loadAsSrgb);
        const texture = new Texture(device, {
            type: ResourceType.Texture2D,
            width: image.width,
            height: image.height,
            format: image.format,
            mipLevels: image.levels.length,
            bindFlags: ResourceBindFlags.ShaderResource,
        });
        image.levels.forEach((l, m) => texture.setSubresourceBlob(m, 0, l.data));
        return texture;
    }

    /** Mirrors saveToDDS(path, bitmap, mode, generateMips); returns the file. */
    static saveToDDS(bitmap: Bitmap, mode = CompressionMode.None, generateMips = false): Uint8Array {
        const format = bitmap.getFormat();
        if (getFormatChannelCount(format) === 2 && mode !== CompressionMode.BC5) throw new RuntimeError("Only BC5 compression is supported for two channel images.");
        if (isCompressedFormat(format)) {
            // Native recompresses; the blocks are already the requested encoding.
            if (mode === CompressionMode.None || mode === modeOfFormat(format)) {
                return writeDDS({ width: bitmap.getWidth(), height: bitmap.getHeight(), mipCount: 1, faces: 1, cube: false, ...outputFor(modeOfFormat(format), format), format, payload: [bitmap.getData()] });
            }
            throw new RuntimeError("Recompressing a compressed bitmap to another format is not supported.");
        }
        let surface: Surface = { width: bitmap.getWidth(), height: bitmap.getHeight(), rgba: toRGBA(format, bitmap.getWidth(), bitmap.getHeight(), bitmap.getData()) };
        surface = ImageIO.clampForMips(surface, mode, generateMips);
        const [width, height] = [surface.width, surface.height];
        const out = outputFor(mode, format);
        const mipCount = generateMips ? countMipmaps(surface.width, surface.height) : 1;
        const payload: Uint8Array[] = [];
        for (let m = 0; m < mipCount; m++) {
            payload.push(encodeSurface(surface, mode, out.format));
            if (m + 1 < mipCount) surface = nextMip(surface);
        }
        return writeDDS({ width, height, mipCount, faces: 1, cube: false, ...out, payload });
    }

    /**
     * Mirrors saveToDDS(context, path, texture, mode, generateMips): every array slice (cube
     * faces) and mip, read back from the GPU; returns the file.
     */
    static async saveTextureToDDS(ctx: RenderContext, texture: Texture, mode = CompressionMode.None, generateMips = false): Promise<Uint8Array> {
        if (texture.type !== ResourceType.Texture2D && texture.type !== ResourceType.TextureCube) throw new RuntimeError("Invalid texture type. Only 2D, 3D, and Cube are currently supported.");
        const format = texture.format;
        if (getFormatChannelCount(format) === 2 && mode !== CompressionMode.BC5) throw new RuntimeError("Only BC5 compression is supported for two channel images.");
        const cube = texture.type === ResourceType.TextureCube;
        const faces = cube ? 6 * texture.arraySize : texture.arraySize;
        const payload: Uint8Array[] = [];
        if (isCompressedFormat(format)) {
            if (mode !== CompressionMode.None && mode !== modeOfFormat(format)) throw new RuntimeError("Recompressing a compressed texture to another format is not supported.");
            const mips = texture.mipCount;
            for (let f = 0; f < faces; f++) for (let m = 0; m < mips; m++) payload.push(new Uint8Array(await ctx.readTextureSubresource(texture, m, f)));
            return writeDDS({ width: texture.width, height: texture.height, mipCount: mips, faces, cube, ...outputFor(modeOfFormat(format), format), format, payload });
        }
        const out = outputFor(mode, format);
        const base = ImageIO.clampForMips({ width: texture.width, height: texture.height, rgba: new Float32Array(0) }, mode, generateMips);
        const mipCount = generateMips ? countMipmaps(base.width, base.height) : texture.mipCount;
        for (let f = 0; f < faces; f++) {
            let surface: Surface | null = null;
            for (let m = 0; m < mipCount; m++) {
                if (m === 0 || !generateMips) {
                    const w = Math.max(1, texture.width >> m);
                    const h = Math.max(1, texture.height >> m);
                    const data = new Uint8Array(await ctx.readTextureSubresource(texture, m, f));
                    surface = { width: w, height: h, rgba: toRGBA(format, w, h, data) };
                    if (m === 0) surface = crop(surface, base.width, base.height);
                } else {
                    surface = nextMip(surface!);
                }
                payload.push(encodeSurface(surface, mode, out.format));
            }
        }
        return writeDDS({ width: base.width, height: base.height, mipCount, faces, cube, ...out, payload });
    }

    /** Mirrors clampIfNeeded: compressed + generated mips need a base size that is a multiple of 4. */
    private static clampForMips(s: Surface, mode: CompressionMode, generateMips: boolean): Surface {
        if (!generateMips || mode === CompressionMode.None) return s;
        const width = s.width > 1 && s.width % 4 !== 0 ? Math.max(1, s.width - (s.width % 4)) : s.width;
        const height = s.height > 1 && s.height % 4 !== 0 ? Math.max(1, s.height - (s.height % 4)) : s.height;
        if (width === s.width && height === s.height) return s;
        Logger.warning("Saving DDS image with clamped image dimensions to accomodate mipmaps and compression.");
        return s.rgba.length ? crop(s, width, height) : { width, height, rgba: s.rgba };
    }
}
