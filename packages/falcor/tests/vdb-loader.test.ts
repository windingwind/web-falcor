/**
 * TS OpenVDB parser + NanoVDB builder vs the validated reference tools
 * (tools/vdb/*.py, native-validated 0/500): parses the real smoke.vdb,
 * checks 500 ground-truth samples, builds the NanoVDB grid buffer and
 * byte-compares it against the committed smoke.nvdb's embedded grid.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { parseOpenVDBFloatGrid, buildNanoVDBGrid, extractGridFromNVDB } from "../src/Scene/Volume/VDBLoader.js";

const root = new URL("../../..", import.meta.url).pathname;
const load = (p: string) => {
    const b = readFileSync(root + p);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

describe("VDBLoader", () => {
    const vdb = load("Falcor/media/test_scenes/volumes/smoke.vdb");
    const ref = JSON.parse(readFileSync(root + "tests/oracle/assets/smoke-vdb-samples.json", "utf8"));
    const grid = parseOpenVDBFloatGrid(vdb, "density");

    it("parses smoke.vdb structure", () => {
        expect(grid.leafOrigins.length).toBe(ref.leafCount);
        expect(grid.scale).toBeCloseTo(ref.scale, 12);
        expect(grid.translation[0]).toBeCloseTo(ref.translation[0], 12);
        let active = 0;
        for (const m of grid.leafMasks) for (let i = 0; i < 64; i++) { let v = m[i]!; while (v) { active += v & 1; v >>= 1; } }
        expect(active).toBe(ref.activeVoxels);
    });

    it("matches 500 ground-truth point samples", () => {
        const index = new Map<string, number>();
        grid.leafOrigins.forEach((o, i) => index.set(`${o[0]},${o[1]},${o[2]}`, i));
        let bad = 0;
        for (const [x, y, z, val, active] of ref.samples) {
            const org = `${x & ~7},${y & ~7},${z & ~7}`;
            const li = index.get(org)!;
            const n = ((x & 7) << 6) | ((y & 7) << 3) | (z & 7);
            const got = grid.leafValues[li]![n]!;
            const isActive = (grid.leafMasks[li]![n >> 3]! & (1 << (n & 7))) !== 0;
            if (Math.abs(got - val) > 1e-7 || isActive !== active) bad++;
        }
        expect(bad).toBe(0);
    });

    it("builds a byte-identical NanoVDB grid buffer", () => {
        const built = buildNanoVDBGrid(grid, "density");
        const refGrid = extractGridFromNVDB(load("tests/oracle/assets/smoke.nvdb"), "density");
        expect(built.length).toBe(refGrid.length);
        let diff = -1;
        for (let i = 0; i < built.length; i++) {
            if (built[i] !== refGrid[i]) { diff = i; break; }
        }
        expect(diff).toBe(-1);
    });
});
