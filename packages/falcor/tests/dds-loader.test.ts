/**
 * DDSLoader parses real Bistro .dds textures (BC1/BC3/BC5) against a Python
 * reference (tests/oracle/assets/bistro-dds-ref.json): dimensions, mip count,
 * chosen BC ResourceFormat, and base-mip block-data length must all match.
 */

import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { parseDDS } from "../src/Scene/Importer/DDSLoader.js";
import { ResourceFormat, getFormatBytesPerBlock } from "../src/Core/API/Formats.js";

const bistro = resolve(__dirname, "../../../Falcor/media/Bistro_v5_2/Textures");
const ref = JSON.parse(readFileSync(resolve(__dirname, "../../../tests/oracle/assets/bistro-dds-ref.json"), "utf8")) as Record<
    string,
    { w: number; h: number; mips: number; fmt: string; level0Bytes: number }
>;

const fmtName: Record<number, string> = {
    [ResourceFormat.BC1Unorm]: "BC1",
    [ResourceFormat.BC1UnormSrgb]: "BC1",
    [ResourceFormat.BC3Unorm]: "BC3",
    [ResourceFormat.BC3UnormSrgb]: "BC3",
    [ResourceFormat.BC5Unorm]: "BC5",
};

// Skips when the Bistro media isn't present (e.g. CI without the full Falcor
// media download); runs in full locally. Parser correctness is also covered
// GPU-side by dds-cpu-decode/dds-bc-upload against the hardware BC decode.
describe.skipIf(!existsSync(bistro))("DDSLoader (Bistro BC textures)", () => {
    for (const [file, e] of Object.entries(ref)) {
        it(`parses ${file} (${e.fmt} ${e.w}x${e.h})`, () => {
            const isColor = file.includes("BaseColor");
            const buf = readFileSync(resolve(bistro, file));
            const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
            const img = parseDDS(ab, isColor);
            expect(img.width).toBe(e.w);
            expect(img.height).toBe(e.h);
            expect(fmtName[img.format]).toBe(e.fmt);
            expect(img.levels.length).toBe(e.mips);
            expect(img.levels[0]!.data.byteLength).toBe(e.level0Bytes);
            // sRGB variant only for color textures.
            if (isColor && e.fmt !== "BC5") {
                expect(img.format === ResourceFormat.BC1UnormSrgb || img.format === ResourceFormat.BC3UnormSrgb).toBe(true);
            }
            // Full mip chain: total block bytes account for the whole file tail.
            const bb = getFormatBytesPerBlock(img.format);
            let total = 0;
            for (const lv of img.levels) total += Math.ceil(lv.width / 4) * Math.ceil(lv.height / 4) * bb;
            expect(total).toBeLessThanOrEqual(ab.byteLength);
        });
    }
});

describe("parseDDS (uncompressed surfaces)", () => {
    /** Builds a minimal DDS: DDPF_RGB (+alpha) with explicit channel masks. */
    const makeDds = (w: number, h: number, bitCount: number, masks: [number, number, number, number], pixels: number[][]) => {
        const bytesPer = bitCount / 8;
        const pitch = w * bytesPer;
        const buf = new ArrayBuffer(128 + pitch * h);
        const dv = new DataView(buf);
        dv.setUint32(0, 0x20534444, true); // "DDS "
        dv.setUint32(4, 124, true);
        dv.setUint32(8, 0x1 | 0x2 | 0x4 | 0x8 | 0x1000, true); // caps|height|width|pitch|pixelformat
        dv.setUint32(12, h, true);
        dv.setUint32(16, w, true);
        dv.setUint32(20, pitch, true);
        dv.setUint32(28, 1, true);
        dv.setUint32(76, 32, true);
        dv.setUint32(80, masks[3] ? 0x41 : 0x40, true); // DDPF_RGB (| DDPF_ALPHAPIXELS)
        dv.setUint32(88, bitCount, true);
        masks.forEach((m, i) => dv.setUint32(92 + i * 4, m, true));
        pixels.forEach(([r, g, b, a], i) => {
            let px = 0;
            const put = (v: number, mask: number) => {
                if (!mask) return;
                px |= (v << (31 - Math.clz32(mask & -mask))) & mask;
            };
            put(r!, masks[0]);
            put(g!, masks[1]);
            put(b!, masks[2]);
            put(a ?? 0, masks[3]);
            for (let byte = 0; byte < bytesPer; byte++) dv.setUint8(128 + i * bytesPer + byte, (px >>> (8 * byte)) & 0xff);
        });
        return buf;
    };

    it("swizzles 32-bit BGRA (the upstream font atlas layout) to RGBA8", () => {
        const img = parseDDS(makeDds(2, 1, 32, [0xff0000, 0xff00, 0xff, 0xff000000], [[10, 20, 30, 40], [200, 150, 100, 255]]), false);
        expect(img.format).toBe(ResourceFormat.RGBA8Unorm);
        expect(img.levels).toHaveLength(1);
        expect(Array.from(img.levels[0]!.data)).toEqual([10, 20, 30, 40, 200, 150, 100, 255]);
    });

    it("expands 24-bit RGB to opaque RGBA8 and honours the sRGB flag", () => {
        const img = parseDDS(makeDds(1, 2, 24, [0xff, 0xff00, 0xff0000, 0], [[1, 2, 3], [4, 5, 6]]), true);
        expect(img.format).toBe(ResourceFormat.RGBA8UnormSrgb);
        expect(Array.from(img.levels[0]!.data)).toEqual([1, 2, 3, 255, 4, 5, 6, 255]);
    });

    it("decodes the shipped dejavu-sans-mono-14 atlas when the Falcor tree is present", () => {
        const path = resolve(__dirname, "../../../Falcor/data/framework/fonts/dejavu-sans-mono-14.dds");
        if (!existsSync(path)) return;
        const img = parseDDS(readFileSync(path).buffer.slice(0) as ArrayBuffer, false);
        expect([img.width, img.height]).toEqual([1024, 17]);
        expect(img.levels[0]!.data.byteLength).toBe(1024 * 17 * 4);
    });
});
