/**
 * Truevision TGA decoder.
 *
 * Web substitute for the format support native Falcor gets from FreeImage:
 * browsers decode PNG/JPEG/WebP but not TGA, which is what pbrt-v4 scenes and
 * many FBX/OBJ assets ship their textures as.
 *
 * Covers the variants those files use: uncompressed and run-length encoded
 * true-colour (24/32-bit), grayscale (8-bit), and colour-mapped images, with
 * both origin conventions. Decoding only — nothing in the port writes TGA.
 */

import { RuntimeError } from "../../Core/Error.js";

export interface TGAImage {
    width: number;
    height: number;
    /** Tightly packed RGBA8, top-left origin. */
    rgba: Uint8Array;
}

/** Image type codes (TGA field 3). */
const kColorMapped = 1;
const kTrueColor = 2;
const kGrayscale = 3;
const kRleColorMapped = 9;
const kRleTrueColor = 10;
const kRleGrayscale = 11;

/** Expands one source pixel (BGR/BGRA/gray/index) into RGBA8. */
function makeExpander(imageType: number, pixelDepth: number, colorMap: Uint8Array | null, colorMapDepth: number) {
    const bytesPerPixel = pixelDepth >> 3;
    const mapEntryBytes = colorMapDepth >> 3;

    return (src: Uint8Array, at: number, dst: Uint8Array, to: number): void => {
        if (imageType === kColorMapped || imageType === kRleColorMapped) {
            const index = bytesPerPixel === 1 ? src[at]! : src[at]! | (src[at + 1]! << 8);
            const entry = index * mapEntryBytes;
            if (!colorMap || entry + mapEntryBytes > colorMap.length) throw new RuntimeError("TGA: colour map index out of range");
            if (mapEntryBytes === 2) {
                // 16-bit entries are A1R5G5B5.
                const v = colorMap[entry]! | (colorMap[entry + 1]! << 8);
                dst[to] = ((v >> 10) & 0x1f) * 255 / 31;
                dst[to + 1] = ((v >> 5) & 0x1f) * 255 / 31;
                dst[to + 2] = (v & 0x1f) * 255 / 31;
                dst[to + 3] = 255;
            } else {
                dst[to] = colorMap[entry + 2]!;
                dst[to + 1] = colorMap[entry + 1]!;
                dst[to + 2] = colorMap[entry]!;
                dst[to + 3] = mapEntryBytes === 4 ? colorMap[entry + 3]! : 255;
            }
            return;
        }
        if (imageType === kGrayscale || imageType === kRleGrayscale) {
            const v = src[at]!;
            dst[to] = v;
            dst[to + 1] = v;
            dst[to + 2] = v;
            dst[to + 3] = bytesPerPixel === 2 ? src[at + 1]! : 255;
            return;
        }
        if (bytesPerPixel === 2) {
            // 16-bit true colour is A1R5G5B5.
            const v = src[at]! | (src[at + 1]! << 8);
            dst[to] = (((v >> 10) & 0x1f) * 255) / 31;
            dst[to + 1] = (((v >> 5) & 0x1f) * 255) / 31;
            dst[to + 2] = ((v & 0x1f) * 255) / 31;
            dst[to + 3] = v & 0x8000 ? 255 : 255;
            return;
        }
        // 24/32-bit true colour is stored BGR(A).
        dst[to] = src[at + 2]!;
        dst[to + 1] = src[at + 1]!;
        dst[to + 2] = src[at]!;
        dst[to + 3] = bytesPerPixel === 4 ? src[at + 3]! : 255;
    };
}

/** Decodes a TGA file to top-left-origin RGBA8. */
export function decodeTGA(buffer: ArrayBuffer): TGAImage {
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 18) throw new RuntimeError("TGA: file shorter than its header");
    const view = new DataView(buffer);

    const idLength = bytes[0]!;
    const colorMapType = bytes[1]!;
    const imageType = bytes[2]!;
    const colorMapFirst = view.getUint16(3, true);
    const colorMapLength = view.getUint16(5, true);
    const colorMapDepth = bytes[7]!;
    const width = view.getUint16(12, true);
    const height = view.getUint16(14, true);
    const pixelDepth = bytes[16]!;
    const descriptor = bytes[17]!;

    if (width === 0 || height === 0) throw new RuntimeError("TGA: zero-sized image");
    if (![kColorMapped, kTrueColor, kGrayscale, kRleColorMapped, kRleTrueColor, kRleGrayscale].includes(imageType)) {
        throw new RuntimeError(`TGA: unsupported image type ${imageType}`);
    }
    if (![8, 15, 16, 24, 32].includes(pixelDepth)) throw new RuntimeError(`TGA: unsupported pixel depth ${pixelDepth}`);

    let offset = 18 + idLength;
    let colorMap: Uint8Array | null = null;
    if (colorMapType === 1) {
        const entryBytes = colorMapDepth >> 3;
        const mapBytes = colorMapLength * entryBytes;
        if (offset + mapBytes > bytes.length) throw new RuntimeError("TGA: truncated colour map");
        // Entries below `colorMapFirst` are absent; shift so indices line up.
        colorMap = new Uint8Array((colorMapFirst + colorMapLength) * entryBytes);
        colorMap.set(bytes.subarray(offset, offset + mapBytes), colorMapFirst * entryBytes);
        offset += mapBytes;
    } else if (imageType === kColorMapped || imageType === kRleColorMapped) {
        throw new RuntimeError("TGA: colour-mapped image without a colour map");
    }

    const bytesPerPixel = Math.ceil(pixelDepth / 8);
    const pixelCount = width * height;
    const expand = makeExpander(imageType, pixelDepth === 15 ? 16 : pixelDepth, colorMap, colorMapDepth);

    // Decode into scanline order as stored, then flip if the origin is bottom-left.
    const stored = new Uint8Array(pixelCount * 4);
    const isRle = imageType >= kRleColorMapped;
    if (!isRle) {
        if (offset + pixelCount * bytesPerPixel > bytes.length) throw new RuntimeError("TGA: truncated pixel data");
        for (let i = 0; i < pixelCount; i++) expand(bytes, offset + i * bytesPerPixel, stored, i * 4);
    } else {
        let read = offset;
        let written = 0;
        while (written < pixelCount) {
            if (read >= bytes.length) throw new RuntimeError("TGA: truncated RLE stream");
            const packet = bytes[read++]!;
            const count = (packet & 0x7f) + 1;
            if (written + count > pixelCount) throw new RuntimeError("TGA: RLE packet overruns the image");
            if (packet & 0x80) {
                // Run-length packet: one pixel repeated.
                if (read + bytesPerPixel > bytes.length) throw new RuntimeError("TGA: truncated RLE run");
                expand(bytes, read, stored, written * 4);
                for (let i = 1; i < count; i++) stored.copyWithin((written + i) * 4, written * 4, written * 4 + 4);
                read += bytesPerPixel;
                written += count;
            } else {
                // Raw packet: `count` literal pixels.
                if (read + count * bytesPerPixel > bytes.length) throw new RuntimeError("TGA: truncated RLE literals");
                for (let i = 0; i < count; i++) expand(bytes, read + i * bytesPerPixel, stored, (written + i) * 4);
                read += count * bytesPerPixel;
                written += count;
            }
        }
    }

    // Bit 5 of the descriptor selects a top-left origin; otherwise rows run bottom-up.
    if (descriptor & 0x20) return { width, height, rgba: stored };
    const rgba = new Uint8Array(pixelCount * 4);
    const rowBytes = width * 4;
    for (let y = 0; y < height; y++) {
        const src = (height - 1 - y) * rowBytes;
        rgba.set(stored.subarray(src, src + rowBytes), y * rowBytes);
    }
    return { width, height, rgba };
}
