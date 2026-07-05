/**
 * Radiance HDR (.hdr / RGBE) decoder — part of Utils/Image (mirrors the subset
 * of Falcor's ImageIO/stb needed by ImageLoader; PNG/JPG go through the
 * browser's native decoders instead).
 */

import { RuntimeError } from "../../Core/Error.js";

export interface HdrImage {
    width: number;
    height: number;
    /** RGBA32Float pixels, alpha = 1. */
    data: Float32Array;
}

export function decodeHdr(bytes: Uint8Array): HdrImage {
    let pos = 0;
    const readLine = (): string => {
        let line = "";
        while (pos < bytes.length) {
            const c = bytes[pos++]!;
            if (c === 0x0a) break;
            line += String.fromCharCode(c);
        }
        return line;
    };

    const signature = readLine();
    if (!signature.startsWith("#?RADIANCE") && !signature.startsWith("#?RGBE")) {
        throw new RuntimeError("Not a Radiance HDR file");
    }
    let format = "";
    for (;;) {
        const line = readLine();
        if (line === "") break; // header/body separator
        if (line.startsWith("FORMAT=")) format = line.slice(7).trim();
    }
    if (format !== "32-bit_rle_rgbe") throw new RuntimeError(`Unsupported HDR format '${format}'`);

    const dims = readLine();
    const m = /^-Y (\d+) \+X (\d+)$/.exec(dims);
    if (!m) throw new RuntimeError(`Unsupported HDR orientation '${dims}'`);
    const height = parseInt(m[1]!, 10);
    const width = parseInt(m[2]!, 10);

    const data = new Float32Array(width * height * 4);
    const scanline = new Uint8Array(width * 4); // planar RGBE per scanline

    for (let y = 0; y < height; y++) {
        // New-style RLE scanline starts with 0x02 0x02 and 16-bit width.
        if (width >= 8 && width < 32768 && bytes[pos] === 2 && bytes[pos + 1] === 2 && ((bytes[pos + 2]! << 8) | bytes[pos + 3]!) === width) {
            pos += 4;
            for (let channel = 0; channel < 4; channel++) {
                let x = 0;
                while (x < width) {
                    let count = bytes[pos++]!;
                    if (count > 128) {
                        // Run of the same value.
                        count -= 128;
                        const value = bytes[pos++]!;
                        for (let i = 0; i < count; i++) scanline[(x++) * 4 + channel] = value;
                    } else {
                        for (let i = 0; i < count; i++) scanline[(x++) * 4 + channel] = bytes[pos++]!;
                    }
                }
            }
        } else {
            // Flat (or old-style RLE, which these assets don't use).
            for (let x = 0; x < width; x++) {
                scanline[x * 4] = bytes[pos]!;
                scanline[x * 4 + 1] = bytes[pos + 1]!;
                scanline[x * 4 + 2] = bytes[pos + 2]!;
                scanline[x * 4 + 3] = bytes[pos + 3]!;
                pos += 4;
            }
        }
        for (let x = 0; x < width; x++) {
            const e = scanline[x * 4 + 3]!;
            const scale = e ? Math.pow(2, e - 136) : 0; // 2^(e-128) / 256
            const o = (y * width + x) * 4;
            data[o] = scanline[x * 4]! * scale;
            data[o + 1] = scanline[x * 4 + 1]! * scale;
            data[o + 2] = scanline[x * 4 + 2]! * scale;
            data[o + 3] = 1;
        }
    }
    return { width, height, data };
}
