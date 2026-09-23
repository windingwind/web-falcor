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

/** Channel names and pixel types (0 UINT, 1 HALF, 2 FLOAT) from an EXR header's chlist. */
export function readExrChannels(bytes: Uint8Array): { name: string; type: number }[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let p = 8; // magic + version
    const cstr = () => {
        const start = p;
        while (p < bytes.length && bytes[p] !== 0) p++;
        return new TextDecoder().decode(bytes.subarray(start, p++));
    };
    while (p < bytes.length) {
        const name = cstr();
        if (name === "") break;
        const type = cstr();
        const size = view.getInt32(p, true);
        p += 4;
        if (name === "channels" && type === "chlist") {
            const channels: { name: string; type: number }[] = [];
            const end = p + size;
            for (;;) {
                const ch = cstr();
                if (ch === "" || p >= end) break;
                channels.push({ name: ch, type: view.getInt32(p, true) });
                p += 16;
            }
            return channels;
        }
        p += size;
    }
    return [];
}
