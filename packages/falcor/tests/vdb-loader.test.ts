/**
 * TS OpenVDB parser + NanoVDB builder vs the validated reference tools
 * (scripts/vdb/*.py, native-validated 0/500): parses the real smoke.vdb,
 * checks 500 ground-truth samples, builds the NanoVDB grid buffer and
 * byte-compares it against the committed smoke.nvdb's embedded grid.
 */

import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { parseOpenVDBFloatGrid, buildNanoVDBGrid, extractGridFromNVDB, parsedGridStats, parsedGridValue } from "../src/Scene/Volume/VDBLoader.js";

const root = new URL("../../..", import.meta.url).pathname;
const load = (p: string) => {
    const b = readFileSync(root + p);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

// smoke.vdb is Falcor media (not fetched by scripts/setup-web.mjs); skip when
// absent (e.g. CI without the full media download). Runs in full locally.
const hasMedia = existsSync(root + "Falcor/media/test_scenes/volumes/smoke.vdb");

describe.skipIf(!hasMedia)("VDBLoader", () => {
    if (!hasMedia) return; // skipIf still runs the collector; don't do eager IO
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

/** A minimal uncompressed OpenVDB file: one upper node holding an active 128^3 tile and a lower child with an active 8^3 tile. */
function vdbWithTiles(): ArrayBuffer {
    const parts: number[] = [];
    const u8 = (...b: number[]) => parts.push(...b);
    const u32 = (v: number) => u8(v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff);
    const i64 = (v: number) => { u32(v); u32(0); };
    const f32 = (v: number) => { const b = new Uint8Array(new Float32Array([v]).buffer); u8(...b); };
    const f64 = (v: number) => { const b = new Uint8Array(new Float64Array([v]).buffer); u8(...b); };
    const str = (s: string) => { u32(s.length); for (const c of s) u8(c.charCodeAt(0)); };
    const mask = (bytes: number, set: number[]) => { const m = new Uint8Array(bytes); for (const n of set) m[n >> 3]! |= 1 << (n & 7); u8(...m); };
    const values = (count: number, at: Record<number, number>) => { u8(6); for (let i = 0; i < count; i++) f32(at[i] ?? 0); };
    i64(0x56444220); u32(224); u32(10); u32(0); u8(1); for (let i = 0; i < 36; i++) u8(48); u32(0);
    u32(1); str("density"); str("Tree_float_5_4_3"); str("");
    const posAt = parts.length; i64(0); i64(0); i64(0);
    const gridPos = parts.length;
    u32(0); u32(0); // no compression; no grid metadata
    str("UniformScaleMap"); for (let i = 0; i < 15; i++) f64(0.5);
    u32(1); f32(0); u32(0); u32(1); // buffers, background, root tiles, root children
    u32(0); u32(0); u32(0); // upper origin
    mask(4096, [1]); mask(4096, [0]); values(32768, { 0: 0.75 }); // child at 1, active tile at 0
    mask(512, []); mask(512, [3]); values(4096, { 3: 0.25 }); // the lower child: an active tile at slot 3
    const blockPos = parts.length;
    const bytes = Uint8Array.from(parts);
    const view = new DataView(bytes.buffer);
    [gridPos, blockPos, blockPos].forEach((v, i) => view.setBigInt64(posAt + i * 8, BigInt(v), true));
    return bytes.buffer;
}

describe("parseOpenVDBFloatGrid tiles", () => {
    it("expands active internal-node tiles into leaves with the tile value", () => {
        const g = parseOpenVDBFloatGrid(vdbWithTiles());
        expect(g.leafOrigins.length).toBe(16 * 16 * 16 + 1);
        expect(parsedGridValue(g, 5, 100, 127)).toBe(0.75); // inside the upper tile (0..127)^3
        expect(parsedGridValue(g, 0, 0, 128 + 3 * 8 + 2)).toBe(0.25); // lower tile at slot 3: z in 152..159
        expect(parsedGridValue(g, 0, 0, 200)).toBe(0); // background
        expect(parsedGridStats(g).voxelCount).toBe(128 ** 3 + 512);
        // The NanoVDB writer takes the tile leaves like any other (fully active, so its statistics hold).
        const nvdb = new DataView(buildNanoVDBGrid(g).buffer);
        expect(Number(nvdb.getBigUint64(672 + 56, true))).toBe(128 ** 3 + 512); // TreeData active voxel count
    });
});
