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
