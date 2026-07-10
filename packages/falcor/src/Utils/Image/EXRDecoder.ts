/**
 * EXR decode via parse-exr (half/float scanline, ZIP/PIZ — the formats
 * native Falcor captures write). Native reads EXR through OpenEXR; the web
 * substitutes a JS decoder with identical pixel output (lossless codecs).
 */

import parseExr from "parse-exr";

export interface ExrImage {
    /** RGBA float32, top-down rows. */
    data: Float32Array;
    width: number;
    height: number;
}

/** FloatType constant from parse-exr (decode to float32). */
const kFloatType = 1015;

export function decodeExr(buffer: ArrayBuffer): ExrImage {
    const { data, width, height } = parseExr(buffer, kFloatType) as { data: Float32Array; width: number; height: number };
    // parse-exr returns rows bottom-up (THREE.js texture convention); flip to
    // the top-down orientation textures and readbacks use everywhere else.
    const flipped = new Float32Array(data.length);
    const rowFloats = width * 4;
    for (let y = 0; y < height; y++) {
        flipped.set(data.subarray((height - 1 - y) * rowFloats, (height - y) * rowFloats), y * rowFloats);
    }
    return { data: flipped, width, height };
}
