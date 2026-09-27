import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildNanoVDBGrid, buildSphereGrid, parseOpenVDBFloatGrid } from "../src/Scene/Volume/VDBLoader.js";
import { NanoVDBAccessor } from "../src/Scene/Volume/NanoVDBAccessor.js";
import { convertNanoVDBToBricks, decodeBC4Volume } from "../src/Scene/Volume/GridConverter.js";
import { float16ToFloat32 } from "../src/Utils/Math/Float16.js";

/** Checks every voxel of every non-empty brick: the range bounds it (with the halo) and the atlas reproduces it. */
function checkBricks(buffer: Uint8Array, bits: 4 | 8 | 16, stride = 1): { bricks: number; maxRelErr: number } {
    const a = new NanoVDBAccessor(buffer);
    const b = convertNanoVDBToBricks(buffer, bits);
    const [lx, ly, lz] = b.leafDim;
    const [ax, ay] = b.atlasSize;
    const texels = bits === 4 ? decodeBC4Volume(b.atlas, ...b.atlasSize) : bits === 8 ? Float32Array.from(b.atlas, (v) => v / 255) : Float32Array.from(new Uint16Array(b.atlas.buffer), (v) => v / 65535);
    const bbMin = a.indexBBox.min.map((v) => v & ~7);
    let [bricks, maxRelErr] = [0, 0];
    for (let z = 0; z < lz; z++)
        for (let y = 0; y < ly; y++)
            for (let x = 0; x < lx; x++) {
                const i = (z * ly + y) * lx + x;
                if (i % stride !== 0) continue;
                const range = b.rangeMips[0]![i]!;
                const [maj, min] = [float16ToFloat32(range & 0xffff), float16ToFloat32(range >>> 16)];
                const ptr = b.indirection[i]!;
                // Brick 0 of the atlas has pointer 0 too: non-empty bricks are the ones with a range.
                const empty = maj === min;
                for (let dz = -1; dz <= 8; dz++)
                    for (let dy = -1; dy <= 8; dy++)
                        for (let dx = -1; dx <= 8; dx++) {
                            const v = a.getValue(bbMin[0]! + x * 8 + dx, bbMin[1]! + y * 8 + dy, bbMin[2]! + z * 8 + dz);
                            // The majorant bounds the brick and its halo; empty bricks hold one value.
                            expect(v <= maj || empty).toBe(true);
                            if (empty || dx < 0 || dy < 0 || dz < 0 || dx > 7 || dy > 7 || dz > 7) continue;
                            const [px, py, pz] = [((ptr & 0xff) << 3) + dx, (((ptr >> 8) & 0xff) << 3) + dy, (((ptr >> 16) & 0xff) << 3) + dz];
                            const t = texels[(pz * ay + py) * ax + px]!;
                            maxRelErr = Math.max(maxRelErr, Math.abs(t * (maj - min) + min - v) / (maj - min));
                        }
                if (!empty) bricks++;
            }
    // Coarser mips bound the finer ones.
    for (let mip = 1; mip < 4; mip++) {
        const coarse = b.rangeMips[mip]!;
        const fine = b.rangeMips[mip - 1]!;
        const [fx, fy] = [lx >> (mip - 1), ly >> (mip - 1)];
        for (let i = 0; i < fine.length; i++) {
            const [x, y, z] = [i % fx, Math.floor(i / fx) % fy, Math.floor(i / (fx * fy))];
            const c = coarse[((z >> 1) * (fy >> 1) + (y >> 1)) * (fx >> 1) + (x >> 1)]!;
            expect(float16ToFloat32(c & 0xffff) >= float16ToFloat32(fine[i]! & 0xffff)).toBe(true);
            expect(float16ToFloat32(c >>> 16) <= float16ToFloat32(fine[i]! >>> 16)).toBe(true);
        }
    }
    if (stride === 1) expect(bricks).toBe(b.nonEmptyCount);
    return { bricks, maxRelErr };
}

describe("GridConverter", () => {
    it("bricks a procedural sphere in BC4, UNORM8 and UNORM16", () => {
        const buffer = buildNanoVDBGrid(buildSphereGrid(1, 0.05));
        // Quantization: BC4 (7 steps over a tile's 8-bit range, then 8-bit truncation), 1/255, 1/65535, plus f16 range rounding.
        for (const [bits, bound] of [[4, 0.1], [8, 0.006], [16, 0.002]] as const) {
            const { bricks, maxRelErr } = checkBricks(buffer, bits);
            expect(bricks).toBeGreaterThan(0);
            expect(maxRelErr).toBeLessThan(bound);
        }
    }, 120000);

    // Needs the media tree (full setup, or `npm run download:assets -- openvdb`).
    const torus = new URL("../../../Falcor/media/openvdb/torus.vdb", import.meta.url);
    it.skipIf(!existsSync(torus))("bricks torus.vdb", () => {
        const file = readFileSync(torus);
        const buffer = buildNanoVDBGrid(parseOpenVDBFloatGrid(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength), "ls_torus"), "ls_torus");
        const t0 = performance.now();
        const b = convertNanoVDBToBricks(buffer);
        console.log(`torus.vdb: ${b.nonEmptyCount} non-empty bricks of ${new NanoVDBAccessor(buffer).leafCount} leaves in ${(performance.now() - t0).toFixed(0)} ms`);
        expect(checkBricks(buffer, 4, 7).maxRelErr).toBeLessThan(0.1);
    }, 300000);
});
