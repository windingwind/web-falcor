/**
 * `.sdfg` corner-value files (SDFGrid::loadValuesFromFile / writeValuesToFile).
 * Generate the fixtures with `node scripts/gen-assets.mjs sdf`.
 */

import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { encodeSDFGridValues, parseSDFGridValues, kSDFMaxDistance } from "../src/Scene/SDFs/SDFGridFile.js";

const root = new URL("../../..", import.meta.url).pathname;

describe("SDF grid value files", () => {
    it("round-trips a grid through the binary format", () => {
        const gridWidth = 4;
        const total = (gridWidth + 1) ** 3;
        const values = Float32Array.from({ length: total }, (_v, i) => Math.sin(i) * 0.5);
        const encoded = encodeSDFGridValues({ gridWidth, values });
        expect(encoded.length).toBe(4 + total * 4);
        const decoded = parseSDFGridValues(encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.byteLength));
        expect(decoded.gridWidth).toBe(gridWidth);
        expect(Array.from(decoded.values)).toEqual(Array.from(values));
    });

    it("rejects truncated files", () => {
        expect(() => parseSDFGridValues(new ArrayBuffer(2))).toThrow(/too short/);
        const short = new Uint8Array(4 + 8);
        new DataView(short.buffer).setUint32(0, 4, true);
        expect(() => parseSDFGridValues(short.buffer)).toThrow(/truncated/);
    });

    const spherePath = `${root}Falcor/media/sdf/sdf-sphere-64.sdfg`;
    it.skipIf(!existsSync(spherePath))("reads the generated sphere as an analytic distance field", () => {
        const buf = readFileSync(spherePath);
        const grid = parseSDFGridValues(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
        expect(grid.gridWidth).toBe(64);
        const widthInValues = grid.gridWidth + 1;
        expect(grid.values.length).toBe(widthInValues ** 3);

        let worst = 0;
        for (let z = 0; z < widthInValues; z++) {
            for (let y = 0; y < widthInValues; y++) {
                for (let x = 0; x < widthInValues; x++) {
                    // Corner (x, y, z) sits at xyz / gridWidth - 0.5 in the unit cube.
                    const p = [x, y, z].map((v) => v / grid.gridWidth - 0.5);
                    const expected = Math.min(Math.max(Math.hypot(p[0]!, p[1]!, p[2]!) - 0.4, -kSDFMaxDistance), kSDFMaxDistance);
                    worst = Math.max(worst, Math.abs(grid.values[x + widthInValues * (y + widthInValues * z)]! - expected));
                }
            }
        }
        expect(worst).toBeLessThan(1e-6);
    });
});
