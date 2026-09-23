/**
 * Portable FloatMap (.pfm) decoder — the float image format pbrt-v4 scenes
 * ship their environment maps in. Native Falcor reads it through FreeImage.
 *
 * Header: "PF" (RGB) or "Pf" (grayscale), then width and height, then a scale
 * whose sign gives the byte order (negative = little endian). Pixel rows run
 * bottom to top; the output here is top-to-bottom RGBA32F with alpha 1.
 */

import { RuntimeError } from "../../Core/Error.js";
import type { HdrImage } from "./HDRDecoder.js";

/** True when the bytes start with a PFM signature. */
export function isPfm(bytes: Uint8Array): boolean {
    return bytes.length >= 3 && bytes[0] === 0x50 && (bytes[1] === 0x46 || bytes[1] === 0x66) && /\s/.test(String.fromCharCode(bytes[2]!));
}

export function decodePfm(bytes: Uint8Array): HdrImage {
    // Three whitespace-separated header tokens after the signature, then one
    // whitespace byte before the raster.
    let pos = 0;
    const token = (): string => {
        while (pos < bytes.length && /\s/.test(String.fromCharCode(bytes[pos]!))) pos++;
        const start = pos;
        while (pos < bytes.length && !/\s/.test(String.fromCharCode(bytes[pos]!))) pos++;
        return new TextDecoder().decode(bytes.subarray(start, pos));
    };
    const signature = token();
    if (signature !== "PF" && signature !== "Pf") throw new RuntimeError(`PFM: bad signature '${signature}'`);
    const channels = signature === "PF" ? 3 : 1;
    const width = parseInt(token(), 10);
    const height = parseInt(token(), 10);
    const scale = parseFloat(token());
    if (!(width > 0 && height > 0) || !Number.isFinite(scale) || scale === 0) throw new RuntimeError("PFM: malformed header");
    pos++; // the single whitespace byte ending the header

    const littleEndian = scale < 0;
    const count = width * height * channels;
    if (pos + count * 4 > bytes.length) throw new RuntimeError("PFM: truncated raster");
    const view = new DataView(bytes.buffer, bytes.byteOffset + pos, count * 4);
    const data = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        // Rows are stored bottom to top.
        const srcRow = height - 1 - y;
        for (let x = 0; x < width; x++) {
            const src = (srcRow * width + x) * channels;
            const dst = (y * width + x) * 4;
            for (let c = 0; c < 3; c++) data[dst + c] = view.getFloat32((src + (channels === 3 ? c : 0)) * 4, littleEndian);
            data[dst + 3] = 1;
        }
    }
    return { width, height, data };
}
