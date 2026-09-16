/**
 * Compression codecs the VDB loader needs (web substitutes for the zlib/blosc
 * libraries native Falcor links through OpenVDB), plus the real blosc-compressed
 * volumes from openvdb.org.
 *
 * Ground truth: Node's zlib for DEFLATE, and each .vdb's own `file_voxel_count`
 * grid metadata (written by OpenVDB) for the parsed active-voxel count.
 * Fetch the volumes with `npm run download:assets -- openvdb`.
 */

import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { deflateSync, deflateRawSync } from "node:zlib";
import { inflate, inflateRaw } from "../src/Utils/Compression/Inflate.js";
import { lz4Decompress } from "../src/Utils/Compression/LZ4.js";
import { bloscReadHeader } from "../src/Utils/Compression/Blosc.js";
import { parseOpenVDBFloatGrid } from "../src/Scene/Volume/VDBLoader.js";

const root = new URL("../../..", import.meta.url).pathname;

describe("Inflate", () => {
    const cases: [string, Uint8Array][] = [
        ["empty", new Uint8Array(0)],
        ["single byte", new Uint8Array([42])],
        ["repetitive (long matches)", new Uint8Array(70000).fill(7)],
        ["text-like (dynamic Huffman)", new TextEncoder().encode("the quick brown fox jumps over the lazy dog. ".repeat(500))],
        [
            "incompressible (stored blocks)",
            (() => {
                // Deterministic pseudo-random bytes: xorshift32 so the test never flakes.
                const data = new Uint8Array(40000);
                let s = 0x12345678;
                for (let i = 0; i < data.length; i++) {
                    s ^= s << 13;
                    s ^= s >>> 17;
                    s ^= s << 5;
                    data[i] = s & 0xff;
                }
                return data;
            })(),
        ],
    ];

    for (const [name, data] of cases) {
        for (const level of [0, 1, 6, 9]) {
            it(`round-trips ${name} at level ${level}`, () => {
                expect(Array.from(inflate(deflateSync(data, { level }), data.length))).toEqual(Array.from(data));
                expect(Array.from(inflateRaw(deflateRawSync(data, { level }), data.length))).toEqual(Array.from(data));
            });
        }
    }

    it("works without an expected size", () => {
        const data = new TextEncoder().encode("no size hint ".repeat(300));
        expect(inflateRaw(deflateRawSync(data)).length).toBe(data.length);
    });

    it("rejects a corrupt zlib header", () => {
        expect(() => inflate(new Uint8Array([0x00, 0x00, 0x00]))).toThrow(/zlib/i);
    });
});

describe("LZ4", () => {
    it("decodes a literals-only block", () => {
        // token 0x40 = 4 literals, no match (the last sequence is literals-only).
        const block = new Uint8Array([0x40, 1, 2, 3, 4]);
        expect(Array.from(lz4Decompress(block, 4))).toEqual([1, 2, 3, 4]);
    });

    it("decodes an overlapping match (run-length case)", () => {
        // 2 literals "ab", then a 4-byte match at offset 2 → "ababab".
        const block = new Uint8Array([0x20, 0x61, 0x62, 0x02, 0x00]);
        expect(new TextDecoder().decode(lz4Decompress(block, 6))).toBe("ababab");
    });

    it("rejects a match that reaches before the output start", () => {
        expect(() => lz4Decompress(new Uint8Array([0x10, 0x61, 0x05, 0x00]), 5)).toThrow(/offset/);
    });
});

/** Minimal .vdb header walk: returns each grid's own metadata (independent of the loader). */
function readGridMetadata(buffer: ArrayBuffer): Map<string, { type: string; voxelCount: number; compression: number }> {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    let o = 8 + 4 + 4 + 4 + 1 + 36; // magic, file version, library version, grid-offset flag, uuid
    const u32 = () => {
        const v = view.getUint32(o, true);
        o += 4;
        return v;
    };
    const i64 = () => {
        const v = Number(view.getBigInt64(o, true));
        o += 8;
        return v;
    };
    const str = () => {
        const n = u32();
        const s = new TextDecoder().decode(bytes.subarray(o, o + n));
        o += n;
        return s;
    };
    for (let i = u32(); i > 0; i--) {
        str();
        str();
        const size = u32(); // separate binding: `o += u32()` would drop the reader's own advance
        o += size;
    }
    const grids = new Map<string, { type: string; voxelCount: number; compression: number }>();
    for (let i = u32(); i > 0; i--) {
        const name = str().split("\x1e")[0]!;
        const type = str();
        str(); // instance parent
        const gridPos = i64();
        i64();
        i64();
        const save = o;
        o = gridPos;
        const compression = u32();
        let voxelCount = 0;
        for (let m = u32(); m > 0; m--) {
            const key = str();
            const valueType = str();
            const size = u32();
            if (key === "file_voxel_count" && valueType === "int64") voxelCount = Number(view.getBigInt64(o, true));
            o += size;
        }
        grids.set(name, { type, voxelCount, compression });
        o = save;
    }
    return grids;
}

const models = [
    { file: "cube.vdb", grid: "ls_cube", levelSet: true },
    { file: "sphere.vdb", grid: "ls_sphere", levelSet: true },
    { file: "torus.vdb", grid: "ls_torus", levelSet: true },
    { file: "smoke.vdb", grid: "density", levelSet: false },
];
const modelDir = `${root}Falcor/media/openvdb/`;
const hasModels = models.every((m) => existsSync(modelDir + m.file));

describe.skipIf(!hasModels)("OpenVDB blosc-compressed sample models", () => {
    if (!hasModels) return; // skipIf still runs the collector; don't do eager IO

    for (const model of models) {
        it(`parses ${model.file} and matches its own voxel count`, () => {
            const buf = readFileSync(modelDir + model.file);
            const data = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
            const meta = readGridMetadata(data).get(model.grid)!;
            expect(meta).toBeDefined();
            expect(meta.compression & 0x4).toBe(0x4); // BLOSC — the point of these fixtures

            const grid = parseOpenVDBFloatGrid(data, model.grid);

            let active = 0;
            for (const m of grid.leafMasks) {
                for (let i = 0; i < 64; i++) {
                    let v = m[i]!;
                    while (v) {
                        active += v & 1;
                        v >>= 1;
                    }
                }
            }
            // OpenVDB writes the active voxel count into the file; the decoded
            // topology must reproduce it exactly.
            expect(active).toBe(meta.voxelCount);
            expect(grid.scale).toBeGreaterThan(0);

            // Every decoded value must be finite, and the narrow band of a level
            // set has to straddle zero (a wrong codec yields garbage or zeros).
            let min = Infinity;
            let max = -Infinity;
            let nonFinite = 0;
            for (let li = 0; li < grid.leafValues.length; li++) {
                const values = grid.leafValues[li]!;
                const mask = grid.leafMasks[li]!;
                for (let i = 0; i < 512; i++) {
                    if ((mask[i >> 3]! & (1 << (i & 7))) === 0) continue;
                    const v = values[i]!;
                    if (!Number.isFinite(v)) nonFinite++;
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }
            expect(nonFinite).toBe(0);
            if (model.levelSet) {
                expect(min).toBeLessThan(0);
                expect(max).toBeGreaterThan(0);
                expect(Math.abs(min)).toBeLessThanOrEqual(grid.background * 1.001);
                expect(max).toBeLessThanOrEqual(grid.background * 1.001);
            } else {
                expect(min).toBeGreaterThanOrEqual(0);
                expect(max).toBeGreaterThan(0);
            }
        });
    }

    it("decodes sphere.vdb into an exact analytic signed distance field", () => {
        // Strongest available check on the *values*: a level-set sphere stores the
        // signed distance to its surface, so |p - center| - value must be the same
        // radius at every active voxel. Any slip in the LZ4 decode, the byte
        // unshuffle or the half conversion scatters these numbers immediately.
        const buf = readFileSync(`${modelDir}sphere.vdb`);
        const grid = parseOpenVDBFloatGrid(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), "ls_sphere");

        const radii: number[] = [];
        for (let li = 0; li < grid.leafOrigins.length; li++) {
            const [ox, oy, oz] = grid.leafOrigins[li]!;
            const mask = grid.leafMasks[li]!;
            const values = grid.leafValues[li]!;
            for (let i = 0; i < 512; i++) {
                if ((mask[i >> 3]! & (1 << (i & 7))) === 0) continue;
                // Voxel index within the 8^3 leaf is x-major (x<<6 | y<<3 | z).
                const x = (ox + ((i >> 6) & 7)) * grid.scale + grid.translation[0];
                const y = (oy + ((i >> 3) & 7)) * grid.scale + grid.translation[1];
                const z = (oz + (i & 7)) * grid.scale + grid.translation[2];
                radii.push(Math.hypot(x, y, z) - values[i]!); // the sphere is centred on the origin
            }
        }
        expect(radii.length).toBeGreaterThan(200000);
        const mean = radii.reduce((a, b) => a + b, 0) / radii.length;
        let maxDeviation = 0;
        for (const r of radii) maxDeviation = Math.max(maxDeviation, Math.abs(r - mean));
        expect(mean).toBeCloseTo(3, 3); // openvdb.org's sphere has radius 3
        expect(maxDeviation).toBeLessThan(1e-3);
    });

    it("reads a blosc chunk header from a leaf buffer", () => {
        // Sanity-check the container reader itself against a known chunk: the
        // sample models are written with LZ4 + byte shuffle.
        const buf = readFileSync(modelDir + "cube.vdb");
        // Scan for the first plausible blosc header (version 2, lz4, shuffle).
        let found: ReturnType<typeof bloscReadHeader> | null = null;
        for (let i = 0; i + 16 < buf.length && !found; i++) {
            if (buf[i] !== 2 || buf[i + 1] !== 1) continue;
            const h = bloscReadHeader(new Uint8Array(buf.buffer, buf.byteOffset + i, 16));
            if (h.codecName === "lz4" && h.nbytes > 0 && h.nbytes < 1 << 20 && h.cbytes <= h.nbytes + 64 && h.blocksize > 0) found = h;
        }
        expect(found).not.toBeNull();
        expect(found!.codecName).toBe("lz4");
        expect(found!.flags & 0x1).toBe(0x1); // byte shuffle
    });
});
