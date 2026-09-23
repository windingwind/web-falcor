/**
 * Mirrors Texture::createFromFile / Texture::createMippedFromFiles: images load
 * through Bitmap (DDS through the DDS parser) and upload with the bitmap's format.
 * Paths are URLs. §9: formats WebGPU lacks are widened on upload — BGRX8 becomes
 * BGRA8 (alpha is already 255) and 16-bit unorm becomes 32-bit float — and BC
 * textures not sized in whole blocks are allocated padded to a multiple of 4
 * (WebGPU requires it), so their width/height report the padded size.
 */

import type { Device } from "./Device.js";
import { Texture, kMaxPossible } from "./Texture.js";
import { ResourceBindFlags, ResourceType } from "./Types.js";
import { ResourceFormat, getFormatBytesPerBlock, isCompressedFormat, toGpuTextureFormat } from "./Formats.js";
import { Logger } from "../../Utils/Logger.js";
import { Bitmap, BitmapImportFlags } from "../../Utils/Image/Bitmap.js";
import { parseDDS } from "../../Scene/Importer/DDSLoader.js";

/** Mirrors linearToSrgbFormat (formats without an sRGB variant stay as they are). */
export function linearToSrgbFormat(format: ResourceFormat): ResourceFormat {
    switch (format) {
        case ResourceFormat.RGBA8Unorm:
            return ResourceFormat.RGBA8UnormSrgb;
        case ResourceFormat.BGRA8Unorm:
            return ResourceFormat.BGRA8UnormSrgb;
        case ResourceFormat.BGRX8Unorm:
            return ResourceFormat.BGRX8UnormSrgb;
        case ResourceFormat.BC1Unorm:
            return ResourceFormat.BC1UnormSrgb;
        case ResourceFormat.BC2Unorm:
            return ResourceFormat.BC2UnormSrgb;
        case ResourceFormat.BC3Unorm:
            return ResourceFormat.BC3UnormSrgb;
        case ResourceFormat.BC7Unorm:
            return ResourceFormat.BC7UnormSrgb;
        default:
            return format;
    }
}

/** A format WebGPU can store, plus the per-level data converted to it. */
function uploadable(format: ResourceFormat, data: Uint8Array): { format: ResourceFormat; data: ArrayBufferView } {
    if (toGpuTextureFormat(format) !== undefined) return { format, data };
    if (format === ResourceFormat.BGRX8Unorm) return { format: ResourceFormat.BGRA8Unorm, data };
    if (format === ResourceFormat.BGRX8UnormSrgb) return { format: ResourceFormat.BGRA8UnormSrgb, data };
    if (format === ResourceFormat.R16Unorm || format === ResourceFormat.RGBA16Unorm) {
        const src = new Uint16Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
        return { format: format === ResourceFormat.R16Unorm ? ResourceFormat.R32Float : ResourceFormat.RGBA32Float, data: Float32Array.from(src, (v) => v / 65535) };
    }
    throw new Error(`Texture loading: ${ResourceFormat[format]} has no WebGPU texture format`);
}

async function fetchBytes(url: string): Promise<Uint8Array | null> {
    try {
        const res = await fetch(url);
        return res.ok ? new Uint8Array(await res.arrayBuffer()) : null;
    } catch {
        return null;
    }
}

interface LoadedLevels {
    width: number;
    height: number;
    format: ResourceFormat;
    levels: Uint8Array[];
}

async function loadLevels(url: string, loadAsSrgb: boolean, importFlags: BitmapImportFlags): Promise<LoadedLevels | null> {
    if (/\.dds(\?|#|$)/i.test(url)) {
        const bytes = await fetchBytes(url);
        if (!bytes) return null;
        let dds;
        try {
            dds = parseDDS(bytes.slice().buffer, loadAsSrgb);
        } catch (e) {
            Logger.warning(`Failed to load DDS image from '${url}': ${(e as Error).message}`);
            return null;
        }
        return { width: dds.width, height: dds.height, format: dds.format, levels: dds.levels.map((l) => l.data) };
    }
    const bmp = await Bitmap.createFromFile(url, true, importFlags);
    if (!bmp) return null;
    return { width: bmp.width, height: bmp.height, format: loadAsSrgb ? linearToSrgbFormat(bmp.format) : bmp.format, levels: [bmp.data] };
}

/** Copies a level's block rows into the (larger) block grid of a padded mip. */
function padBlocks(data: Uint8Array, bpb: number, from: [number, number], to: [number, number]): Uint8Array {
    if (from[0] === to[0] && from[1] === to[1]) return data;
    const out = new Uint8Array(to[0] * to[1] * bpb);
    for (let y = 0; y < from[1]; y++) out.set(data.subarray(y * from[0] * bpb, (y + 1) * from[0] * bpb), y * to[0] * bpb);
    return out;
}

function createTexture(device: Device, img: LoadedLevels, mipLevels: number, bindFlags: ResourceBindFlags): Texture {
    const first = uploadable(img.format, img.levels[0]!);
    const compressed = isCompressedFormat(first.format);
    const pad = (v: number) => (compressed ? Math.ceil(v / 4) * 4 : v);
    const tex = new Texture(device, { type: ResourceType.Texture2D, width: pad(img.width), height: pad(img.height), format: first.format, mipLevels, bindFlags });
    const blocks = (w: number, h: number, m: number): [number, number] => [Math.ceil(Math.max(1, w >> m) / 4), Math.ceil(Math.max(1, h >> m) / 4)];
    img.levels.forEach((level, m) => {
        if (m >= tex.mipCount) return;
        const data = uploadable(img.format, level).data;
        if (!compressed) return tex.setSubresourceBlob(m, 0, data);
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        tex.setSubresourceBlob(m, 0, padBlocks(bytes, getFormatBytesPerBlock(first.format), blocks(img.width, img.height, m), blocks(tex.width, tex.height, m)));
    });
    return tex;
}

/** Mirrors Texture::createFromFile. Returns null (with a warning) on failure. */
export async function createTextureFromFile(
    device: Device,
    url: string,
    generateMipLevels: boolean,
    loadAsSrgb: boolean,
    bindFlags = ResourceBindFlags.ShaderResource,
    importFlags = BitmapImportFlags.None,
): Promise<Texture | null> {
    const img = await loadLevels(url, loadAsSrgb, importFlags);
    if (!img) {
        Logger.warning(`Error when loading image file. File '${url}' does not exist.`);
        return null;
    }
    const dds = img.levels.length > 1;
    // Generated mips need render-attachment use for the blit chain.
    const flags = generateMipLevels && !dds ? bindFlags | ResourceBindFlags.RenderTarget : bindFlags;
    const tex = createTexture(device, img, dds ? img.levels.length : generateMipLevels ? kMaxPossible : 1, flags);
    if (generateMipLevels && !dds) tex.generateMips(device.renderContext);
    tex.name = url;
    return tex;
}

/** Mirrors Texture::createMippedFromFiles: one file per mip, each half the previous size. */
export async function createMippedTextureFromFiles(
    device: Device,
    urls: string[],
    loadAsSrgb: boolean,
    bindFlags = ResourceBindFlags.ShaderResource,
    importFlags = BitmapImportFlags.None,
): Promise<Texture | null> {
    const mips: LoadedLevels[] = [];
    for (const url of urls) {
        const img = await loadLevels(url, false, importFlags);
        if (!img) {
            Logger.warning(`Error loading mip ${mips.length}. Loading failed for image file '${url}'.`);
            break;
        }
        const prev = mips[mips.length - 1];
        if (prev) {
            if (prev.format !== img.format) {
                Logger.warning(`Error loading mip ${mips.length} from file ${url}. Texture format of all mip levels must match.`);
                break;
            }
            if (Math.max(prev.width >> 1, 1) !== img.width || Math.max(prev.height >> 1, 1) !== img.height) {
                Logger.warning(`Error loading mip ${mips.length} from file ${url}. Image resolution must decrease by half. (${img.width}, ${img.height}) != (${prev.width}, ${prev.height})/2`);
                break;
            }
        }
        mips.push(img);
    }
    if (mips.length === 0) return null;
    const base = mips[0]!;
    const tex = createTexture(
        device,
        { width: base.width, height: base.height, format: loadAsSrgb ? linearToSrgbFormat(base.format) : base.format, levels: mips.map((m) => m.levels[0]!) },
        mips.length,
        bindFlags,
    );
    tex.name = urls[0]!;
    return tex;
}
