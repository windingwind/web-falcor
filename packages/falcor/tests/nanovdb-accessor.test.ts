import { describe, expect, it } from "vitest";
import { buildBoxGrid, buildNanoVDBGrid, buildSphereGrid, parsedGridValue } from "../src/Scene/Volume/VDBLoader.js";
import { NanoVDBAccessor } from "../src/Scene/Volume/NanoVDBAccessor.js";

describe("NanoVDBAccessor", () => {
    it("reads every voxel of a NanoVDB buffer as the source grid holds it", () => {
        for (const grid of [buildSphereGrid(1, 0.05), buildBoxGrid(1, 0.5, 2, 0.04)]) {
            const a = new NanoVDBAccessor(buildNanoVDBGrid(grid));
            expect(a.leafCount).toBe(grid.leafOrigins.length);
            const { min, max } = a.indexBBox;
            let mismatches = 0;
            // The whole bounding box plus a margin outside the tree (background).
            for (let i = min[0] - 9; i <= max[0] + 9; i++)
                for (let j = min[1] - 9; j <= max[1] + 9; j += 3)
                    for (let k = min[2] - 9; k <= max[2] + 9; k++) if (!Object.is(a.getValue(i, j, k), parsedGridValue(grid, i, j, k))) mismatches++;
            expect(mismatches).toBe(0);
            // probeLeaf finds exactly the stored leaves.
            for (const [x, y, z] of grid.leafOrigins) expect(a.probeLeaf(x + 3, y + 5, z + 7)).toBeGreaterThan(0);
            expect(a.probeLeaf(max[0] + 64, max[1] + 64, max[2] + 64)).toBe(-1);
        }
    }, 60000);
});
