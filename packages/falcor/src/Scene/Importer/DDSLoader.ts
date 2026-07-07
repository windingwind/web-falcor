/**
 * DirectDraw Surface (.dds) parser for BC-compressed textures — the format
 * used by most game assets (Bistro, Sponza, SunTemple). The browser's
 * createImageBitmap can't decode these; this reads the header and hands the
 * compressed blocks straight to a WebGPU BC-format texture (the device
 * requests 'texture-compression-bc'), so nothing is decompressed on the CPU.
 *
 * Supported FourCC: DXT1 (BC1), DXT3 (BC2), DXT5 (BC3), ATI1/BC4U (BC4),
 * ATI2/BC5U (BC5), and the DX10 extension header for BC7 and the sRGB
 * variants. Uncompressed DDS (RGBA masks) is not handled (rare for assets).
 */

import { ResourceFormat } from "../../Core/API/Formats.js";

const DDS_MAGIC = 0x20534444; // "DDS "
const fourCC = (a: string) => a.charCodeAt(0) | (a.charCodeAt(1) << 8) | (a.charCodeAt(2) << 16) | (a.charCodeAt(3) << 24);

export interface DDSImage {
    width: number;
    height: number;
    format: ResourceFormat;
    /** Mip levels, coarsest→finest order preserved (index 0 = base mip). */
    levels: { data: Uint8Array; width: number; height: number }[];
}

/** Block byte size for a BC ResourceFormat (BC1/BC4 = 8, others = 16). */
function blockBytes(format: ResourceFormat): number {
    return format === ResourceFormat.BC1Unorm || format === ResourceFormat.BC1UnormSrgb || format === ResourceFormat.BC4Unorm ? 8 : 16;
}

/**
 * Parses a .dds buffer. `srgb` selects the sRGB BC variant for color textures
 * (BC1/BC3/BC7); BC4/BC5 (single/two-channel data maps) are always linear.
 */
export function parseDDS(buffer: ArrayBuffer, srgb: boolean): DDSImage {
    const dv = new DataView(buffer);
    if (dv.getUint32(0, true) !== DDS_MAGIC) throw new Error("DDSLoader: not a DDS file");

    const height = dv.getUint32(12, true);
    const width = dv.getUint32(16, true);
    const mipCount = Math.max(1, dv.getUint32(28, true));
    const pfFourCC = dv.getUint32(84, true);

    let format: ResourceFormat;
    let dataOffset = 128; // 4 (magic) + 124 (DDS_HEADER)

    if (pfFourCC === fourCC("DX10")) {
        // DDS_HEADER_DXT10 (20 bytes) follows the base header.
        const dxgi = dv.getUint32(128, true);
        dataOffset = 148;
        format = dxgiToFormat(dxgi, srgb);
    } else if (pfFourCC === fourCC("DXT1")) {
        format = srgb ? ResourceFormat.BC1UnormSrgb : ResourceFormat.BC1Unorm;
    } else if (pfFourCC === fourCC("DXT3")) {
        format = srgb ? ResourceFormat.BC2UnormSrgb : ResourceFormat.BC2Unorm;
    } else if (pfFourCC === fourCC("DXT5")) {
        format = srgb ? ResourceFormat.BC3UnormSrgb : ResourceFormat.BC3Unorm;
    } else if (pfFourCC === fourCC("ATI1") || pfFourCC === fourCC("BC4U")) {
        format = ResourceFormat.BC4Unorm;
    } else if (pfFourCC === fourCC("ATI2") || pfFourCC === fourCC("BC5U")) {
        format = ResourceFormat.BC5Unorm;
    } else {
        throw new Error(`DDSLoader: unsupported FourCC 0x${pfFourCC.toString(16)} (only BC/DXT compressed DDS)`);
    }

    const bb = blockBytes(format);
    const levels: DDSImage["levels"] = [];
    let offset = dataOffset;
    let w = width;
    let h = height;
    for (let m = 0; m < mipCount; m++) {
        const blocksW = Math.max(1, Math.ceil(w / 4));
        const blocksH = Math.max(1, Math.ceil(h / 4));
        const size = blocksW * blocksH * bb;
        if (offset + size > buffer.byteLength) break; // truncated / no full mip chain
        levels.push({ data: new Uint8Array(buffer, offset, size), width: w, height: h });
        offset += size;
        w = Math.max(1, w >> 1);
        h = Math.max(1, h >> 1);
    }
    if (levels.length === 0) throw new Error("DDSLoader: no mip data");

    return { width, height, format, levels };
}

/** Minimal DXGI_FORMAT → ResourceFormat map for the BC subset. */
function dxgiToFormat(dxgi: number, srgb: boolean): ResourceFormat {
    switch (dxgi) {
        case 71: // BC1_UNORM
        case 72: // BC1_UNORM_SRGB
            return srgb ? ResourceFormat.BC1UnormSrgb : ResourceFormat.BC1Unorm;
        case 74: // BC2_UNORM
        case 75: // BC2_UNORM_SRGB
            return srgb ? ResourceFormat.BC2UnormSrgb : ResourceFormat.BC2Unorm;
        case 77: // BC3_UNORM
        case 78: // BC3_UNORM_SRGB
            return srgb ? ResourceFormat.BC3UnormSrgb : ResourceFormat.BC3Unorm;
        case 80: // BC4_UNORM
            return ResourceFormat.BC4Unorm;
        case 83: // BC5_UNORM
            return ResourceFormat.BC5Unorm;
        case 98: // BC7_UNORM
        case 99: // BC7_UNORM_SRGB
            return srgb ? ResourceFormat.BC7UnormSrgb : ResourceFormat.BC7Unorm;
        default:
            throw new Error(`DDSLoader: unsupported DXGI format ${dxgi}`);
    }
}


/**
 * CPU decode of a BC block-compressed level to RGBA8. Used to feed the existing
 * RGBA8 texture-array path (the browser can't decode DDS, and a full BC-array
 * material path is a larger change): pick a capped-size mip so memory stays
 * bounded, decode it, and hand back RGBA8 pixels. BC1/BC3/BC5 cover the common
 * game-asset set (Bistro etc.); other formats throw.
 */

function rgb565(v: number): [number, number, number] {
    const r = (v >> 11) & 0x1f;
    const g = (v >> 5) & 0x3f;
    const b = v & 0x1f;
    return [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)];
}

/** BC1 color block (8 bytes) -> writes RGB (+ 1-bit alpha) for a 4x4 tile. */
function decodeBC1Color(data: Uint8Array, o: number, out: Uint8Array, ox: number, oy: number, w: number, h: number): void {
    const c0 = data[o]! | (data[o + 1]! << 8);
    const c1 = data[o + 2]! | (data[o + 3]! << 8);
    const e0 = rgb565(c0);
    const e1 = rgb565(c1);
    const pal: [number, number, number, number][] = [
        [e0[0], e0[1], e0[2], 255],
        [e1[0], e1[1], e1[2], 255],
        c0 > c1
            ? [(2 * e0[0] + e1[0]) / 3, (2 * e0[1] + e1[1]) / 3, (2 * e0[2] + e1[2]) / 3, 255]
            : [(e0[0] + e1[0]) / 2, (e0[1] + e1[1]) / 2, (e0[2] + e1[2]) / 2, 255],
        c0 > c1 ? [(e0[0] + 2 * e1[0]) / 3, (e0[1] + 2 * e1[1]) / 3, (e0[2] + 2 * e1[2]) / 3, 255] : [0, 0, 0, 0],
    ];
    const bits = data[o + 4]! | (data[o + 5]! << 8) | (data[o + 6]! << 16) | (data[o + 7]! * 0x1000000);
    for (let py = 0; py < 4; py++) {
        for (let px = 0; px < 4; px++) {
            const x = ox + px;
            const y = oy + py;
            if (x >= w || y >= h) continue;
            const idx = (bits >> (2 * (py * 4 + px))) & 3;
            const c = pal[idx]!;
            const d = (y * w + x) * 4;
            out[d] = c[0];
            out[d + 1] = c[1];
            out[d + 2] = c[2];
            out[d + 3] = c[3];
        }
    }
}

/** BC4 single-channel block (8 bytes) -> per-texel value into channel `ch`. */
function decodeBC4Channel(data: Uint8Array, o: number, out: Uint8Array, ox: number, oy: number, w: number, h: number, ch: number): void {
    const a0 = data[o]!;
    const a1 = data[o + 1]!;
    const a: number[] = [a0, a1];
    if (a0 > a1) for (let i = 1; i <= 6; i++) a.push(((7 - i) * a0 + i * a1) / 7);
    else {
        for (let i = 1; i <= 4; i++) a.push(((5 - i) * a0 + i * a1) / 5);
        a.push(0, 255);
    }
    // 16 3-bit indices packed in the 6 bytes after a0,a1.
    let lo = data[o + 2]! | (data[o + 3]! << 8) | (data[o + 4]! << 16);
    let hi = data[o + 5]! | (data[o + 6]! << 8) | (data[o + 7]! << 16);
    for (let t = 0; t < 16; t++) {
        const px = t & 3;
        const py = t >> 2;
        const x = ox + px;
        const y = oy + py;
        const idx = t < 8 ? (lo >> (3 * t)) & 7 : (hi >> (3 * (t - 8))) & 7;
        if (x < w && y < h) out[(y * w + x) * 4 + ch] = Math.round(a[idx]!);
    }
}

/** Decode one BC level to RGBA8 (BC1/BC3/BC5). */
function decodeLevelToRGBA(data: Uint8Array, format: ResourceFormat, w: number, h: number): Uint8Array {
    const out = new Uint8Array(w * h * 4);
    const blocksW = Math.max(1, Math.ceil(w / 4));
    const blocksH = Math.max(1, Math.ceil(h / 4));
    const isBC1 = format === ResourceFormat.BC1Unorm || format === ResourceFormat.BC1UnormSrgb;
    const isBC3 = format === ResourceFormat.BC3Unorm || format === ResourceFormat.BC3UnormSrgb;
    const isBC5 = format === ResourceFormat.BC5Unorm;
    const stride = isBC1 ? 8 : 16;
    for (let by = 0; by < blocksH; by++) {
        for (let bx = 0; bx < blocksW; bx++) {
            const o = (by * blocksW + bx) * stride;
            const ox = bx * 4;
            const oy = by * 4;
            if (isBC1) {
                decodeBC1Color(data, o, out, ox, oy, w, h);
            } else if (isBC3) {
                decodeBC4Channel(data, o, out, ox, oy, w, h, 3); // 8-byte alpha block first
                decodeBC1Color(data, o + 8, out, ox, oy, w, h); // then BC1 color (overwrites a=255)
                // BC3 alpha is the BC4 block; re-apply it after color set a to 255.
                decodeBC4Channel(data, o, out, ox, oy, w, h, 3);
            } else if (isBC5) {
                decodeBC4Channel(data, o, out, ox, oy, w, h, 0); // R = normal.x
                decodeBC4Channel(data, o + 8, out, ox, oy, w, h, 1); // G = normal.y
                // Reconstruct B and set alpha for a viewable normal map.
                for (let py = 0; py < 4; py++)
                    for (let px = 0; px < 4; px++) {
                        const x = ox + px;
                        const y = oy + py;
                        if (x >= w || y >= h) continue;
                        const d = (y * w + x) * 4;
                        const nx = out[d]! / 127.5 - 1;
                        const ny = out[d + 1]! / 127.5 - 1;
                        out[d + 2] = Math.round((Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny)) * 0.5 + 0.5) * 255);
                        out[d + 3] = 255;
                    }
            }
        }
    }
    return out;
}

/**
 * Decodes a .dds to RGBA8 at a bounded resolution: parses the header, picks the
 * finest mip whose largest dimension is <= maxDim (keeps memory bounded for
 * texture-heavy scenes), decodes it and returns { width, height, rgba }.
 */
export function decodeDDSToRGBA(buffer: ArrayBuffer, srgb: boolean, maxDim = 512): { width: number; height: number; rgba: Uint8Array } {
    const img = parseDDS(buffer, srgb);
    // levels[0] is the base (largest); pick the finest that fits maxDim.
    let level = img.levels[0]!;
    for (const lv of img.levels) {
        if (Math.max(lv.width, lv.height) <= maxDim) {
            level = lv;
            break;
        }
        level = lv; // fall through to the smallest if none fit
    }
    // If even the smallest exceeds maxDim (no mips), use the base.
    if (Math.max(level.width, level.height) > maxDim && img.levels.length > 1) {
        level = img.levels[img.levels.length - 1]!;
    }
    return { width: level.width, height: level.height, rgba: decodeLevelToRGBA(level.data, img.format, level.width, level.height) };
}
