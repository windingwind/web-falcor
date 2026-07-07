/**
 * DDSLoader parses real Bistro .dds textures (BC1/BC3/BC5) against a Python
 * reference (tests/oracle/assets/bistro-dds-ref.json): dimensions, mip count,
 * chosen BC ResourceFormat, and base-mip block-data length must all match.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
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

describe("DDSLoader (Bistro BC textures)", () => {
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
