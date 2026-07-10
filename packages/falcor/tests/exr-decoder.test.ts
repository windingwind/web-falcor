import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { decodeExr } from "../src/Utils/Image/EXRDecoder.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

describe("EXRDecoder", () => {
    it("decodes a native Falcor capture (fp16 ZIP scanlines) top-down", () => {
        // Cornell box radiance from the MPT oracle: the directly-lit floor
        // (LOWER rows when top-down) far outshines the ceiling band — the
        // same orientation the GPU suites pin vs web readbacks.
        const file = readFileSync(resolve(repoRoot, "tests/oracle/out-native/oracle-imgtest-mpt.ToneMapper.dst.0.exr"));
        const { data, width, height } = decodeExr(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer);
        expect(width).toBe(256);
        expect(height).toBe(256);
        expect(data.length).toBe(256 * 256 * 4);

        let finite = true;
        const rowLum = (y: number) => {
            let s = 0;
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                s += data[i]! + data[i + 1]! + data[i + 2]!;
                finite &&= Number.isFinite(data[i]!);
            }
            return s / width;
        };
        let top = 0;
        let bottom = 0;
        for (let y = 0; y < 32; y++) {
            top += rowLum(y);
            bottom += rowLum(height - 1 - y);
        }
        expect(finite).toBe(true);
        // Directly-lit floor: the bottom band is much brighter than the ceiling band.
        expect(bottom).toBeGreaterThan(top * 1.5);
    });
});
