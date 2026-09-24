/**
 * packSBSGrids: several SBS grids share one gScene.sdfGrid0 binding. One grid packs to its own
 * build output unchanged; more grids get offset brick IDs and z-stacked indirection slabs.
 */
import { describe, it, expect } from "vitest";
import { SDFSBS, packSBSGrids } from "../src/Scene/SDFs/SDFSBS.js";

const cheese = (gridWidth: number, seed: number) => {
    const g = new SDFSBS();
    g.generateCheeseValues(gridWidth, seed);
    return g;
};

describe("packSBSGrids", () => {
    it("reproduces a single grid bit-exactly", () => {
        const g = cheese(32, 1);
        const p = packSBSGrids([g]);
        expect(p.bricksPerAxis).toEqual(g.bricksPerAxis);
        expect(p.brickTextureDimensions).toEqual(g.brickTextureDimensions);
        expect(p.indirectionDims).toEqual([g.virtualBricksPerAxis, g.virtualBricksPerAxis, g.virtualBricksPerAxis]);
        expect(p.indirection).toEqual(g.indirection);
        expect(p.brickTexture).toEqual(g.brickTexture);
        expect(p.aabbs).toEqual(g.aabbs);
    });

    it("offsets brick IDs and stacks indirection along z", () => {
        const [a, b] = [cheese(32, 1), cheese(16, 2)];
        const p = packSBSGrids([a, b]);
        expect(p.brickOffsets).toEqual([0, a.brickCount]);
        expect(p.zOffsets).toEqual([0, a.virtualBricksPerAxis]);
        expect(p.aabbs.length).toBe(a.brickCount + b.brickCount);
        const side = p.indirectionDims[0];
        const bwv = a.brickWidth + 1;
        const texel = (tex: Float32Array, texW: number, bpa: number, id: number, k: number) => tex[(id % bpa) * bwv * bwv + (k % (bwv * bwv)) + texW * (Math.floor(id / bpa) * bwv + Math.floor(k / (bwv * bwv)))];
        const v = b.virtualBricksPerAxis;
        for (let z = 0; z < v; z++)
            for (let y = 0; y < v; y++)
                for (let x = 0; x < v; x++) {
                    const local = b.indirection[x + v * (y + v * z)]!;
                    const packed = p.indirection[x + side * (y + side * (z + p.zOffsets[1]!))]!;
                    if (local === 0xffffffff) {
                        expect(packed).toBe(0xffffffff);
                        continue;
                    }
                    expect(packed).toBe(local + a.brickCount);
                    // The brick's texels moved with it.
                    for (const k of [0, 17, bwv * bwv * bwv - 1]) {
                        expect(texel(p.brickTexture, p.brickTextureDimensions[0], p.bricksPerAxis[0], packed, k)).toBe(texel(b.brickTexture, b.brickTextureDimensions[0], b.bricksPerAxis[0], local, k));
                    }
                }
    });
});
