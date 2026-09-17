/**
 * TGA decoding (the web substitute for FreeImage's TGA support). Ground truth:
 * files written here to the spec, plus the real textures shipped with the pbrt
 * scenes when they are present (`npm run download:scenes -- bathroom`).
 */

import { describe, expect, it } from "vitest";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { decodeTGA } from "../src/Utils/Image/TGADecoder.js";

const root = new URL("../../..", import.meta.url).pathname;

/** Builds a TGA file with the given header fields and payload. */
function makeTga(imageType: number, width: number, height: number, depth: number, descriptor: number, payload: number[], colorMap: number[] = [], colorMapDepth = 0) {
    const header = new Uint8Array(18);
    header[1] = colorMap.length ? 1 : 0;
    header[2] = imageType;
    new DataView(header.buffer).setUint16(5, colorMapDepth ? colorMap.length / (colorMapDepth >> 3) : 0, true);
    header[7] = colorMapDepth;
    new DataView(header.buffer).setUint16(12, width, true);
    new DataView(header.buffer).setUint16(14, height, true);
    header[16] = depth;
    header[17] = descriptor;
    const out = new Uint8Array(header.length + colorMap.length + payload.length);
    out.set(header, 0);
    out.set(colorMap, header.length);
    out.set(payload, header.length + colorMap.length);
    return out.buffer.slice(0) as ArrayBuffer;
}

describe("TGA decoder", () => {
    it("decodes uncompressed 24-bit BGR with a top-left origin", () => {
        // Two pixels: red then green, stored BGR.
        const file = makeTga(2, 2, 1, 24, 0x20, [0, 0, 255, 0, 255, 0]);
        const img = decodeTGA(file);
        expect([img.width, img.height]).toEqual([2, 1]);
        expect(Array.from(img.rgba)).toEqual([255, 0, 0, 255, 0, 255, 0, 255]);
    });

    it("decodes 32-bit BGRA and keeps alpha", () => {
        const file = makeTga(2, 1, 1, 32, 0x20, [10, 20, 30, 40]);
        expect(Array.from(decodeTGA(file).rgba)).toEqual([30, 20, 10, 40]);
    });

    it("flips bottom-left-origin images", () => {
        // Row 0 (stored first) is the bottom row: black then white.
        const file = makeTga(2, 1, 2, 24, 0x00, [0, 0, 0, 255, 255, 255]);
        const img = decodeTGA(file);
        // Top row of the decoded image must be the white one.
        expect(Array.from(img.rgba.subarray(0, 4))).toEqual([255, 255, 255, 255]);
        expect(Array.from(img.rgba.subarray(4, 8))).toEqual([0, 0, 0, 255]);
    });

    it("decodes run-length encoded true colour", () => {
        // One RLE packet of 3 blue pixels, then a raw packet of 1 red pixel.
        const file = makeTga(10, 4, 1, 24, 0x20, [0x82, 255, 0, 0, 0x00, 0, 0, 255]);
        const img = decodeTGA(file);
        expect(Array.from(img.rgba)).toEqual([0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255]);
    });

    it("decodes 8-bit grayscale", () => {
        const file = makeTga(3, 2, 1, 8, 0x20, [0, 128]);
        expect(Array.from(decodeTGA(file).rgba)).toEqual([0, 0, 0, 255, 128, 128, 128, 255]);
    });

    it("decodes colour-mapped images", () => {
        // Two-entry BGR map: red, green; pixels index 1 then 0.
        const file = makeTga(1, 2, 1, 8, 0x20, [1, 0], [0, 0, 255, 0, 255, 0], 24);
        expect(Array.from(decodeTGA(file).rgba)).toEqual([0, 255, 0, 255, 255, 0, 0, 255]);
    });

    it("rejects malformed files", () => {
        expect(() => decodeTGA(new ArrayBuffer(4))).toThrow(/header/);
        expect(() => decodeTGA(makeTga(2, 0, 0, 24, 0x20, []))).toThrow(/zero-sized/);
        expect(() => decodeTGA(makeTga(2, 4, 4, 24, 0x20, [1, 2, 3]))).toThrow(/truncated/);
    });

    const textureDir = `${root}Falcor/media/bathroom/textures`;
    it.skipIf(!existsSync(textureDir))("decodes the real pbrt scene textures", () => {
        const files = readdirSync(textureDir).filter((f) => f.toLowerCase().endsWith(".tga"));
        expect(files.length).toBeGreaterThan(0);
        for (const name of files) {
            const buf = readFileSync(`${textureDir}/${name}`);
            const img = decodeTGA(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
            expect(img.width).toBeGreaterThan(0);
            expect(img.rgba.length).toBe(img.width * img.height * 4);
            // Fully opaque, and not a uniform block (a wrong stride yields garbage or flat output).
            let alphaAlways255 = true;
            const first = img.rgba[0]!;
            let varies = false;
            for (let i = 0; i < img.rgba.length; i += 4) {
                if (img.rgba[i + 3] !== 255) alphaAlways255 = false;
                if (img.rgba[i] !== first) varies = true;
            }
            expect(alphaAlways255).toBe(true); // these are 24-bit files
            expect(varies).toBe(true);
        }
    });
});
