/**
 * SDFSBS CPU brick-build internal-consistency checks (the GPU render oracle
 * cross-validates the layout end-to-end in phase 3b). Pins the deterministic
 * cheese(128, 0) build so a structural regression is caught immediately.
 */

import { describe, expect, it } from "vitest";
import { SDFSBS } from "../src/Scene/SDFs/SDFSBS.js";

describe("SDFSBS brick build (cheese 128, brickWidth 7)", () => {
    const sbs = new SDFSBS(7);
    sbs.generateCheeseValues(128, 0);
    const bw = 7;
    const bwv = 8;
    const vbpa = sbs.virtualBricksPerAxis;

    it("has a plausible sparse brick set", () => {
        expect(vbpa).toBe(Math.ceil(128 / 7)); // 19
        expect(sbs.brickCount).toBeGreaterThan(0);
        // Sparse: far fewer than all virtual bricks (cheese is a thin shell).
        expect(sbs.brickCount).toBeLessThan(vbpa ** 3);
        expect(sbs.aabbs.length).toBe(sbs.brickCount);
    });

    it("indirection round-trips valid bricks and marks the rest invalid", () => {
        let valid = 0;
        for (let i = 0; i < sbs.indirection.length; i++) {
            const id = sbs.indirection[i]!;
            if (id !== 0xffffffff) {
                expect(id).toBeLessThan(sbs.brickCount);
                valid++;
            }
        }
        expect(valid).toBe(sbs.brickCount);
    });

    it("brick AABBs are brick-aligned and inside the unit grid", () => {
        const step = bw / 128;
        for (const aabb of sbs.aabbs) {
            for (let c = 0; c < 3; c++) {
                expect(aabb.min[c]).toBeGreaterThanOrEqual(-0.5 - 1e-6);
                expect(aabb.max[c]).toBeLessThanOrEqual(0.5 + 1e-6);
                // min is on the brick lattice (-0.5 + k*step).
                const k = (aabb.min[c]! + 0.5) / step;
                expect(Math.abs(k - Math.round(k))).toBeLessThan(1e-4);
            }
        }
    });

    it("brick texels reproduce the dense field for interior voxels", () => {
        const tw = sbs.brickTextureDimensions[0];
        const bax = sbs.bricksPerAxis[0];
        // Reconstruct which virtual brick each brickID maps to.
        const brickToVirtual = new Map<number, number>();
        for (let i = 0; i < sbs.indirection.length; i++) {
            const id = sbs.indirection[i]!;
            if (id !== 0xffffffff) brickToVirtual.set(id, i);
        }
        // Field accessor (snorm8 decode, same as the host).
        const gwv = 129;
        const rawField = (sbs as unknown as { sdField: Int8Array }).sdField;
        const field = (x: number, y: number, z: number) => rawField[x + gwv * (y + gwv * z)]! / 127;

        let checked = 0;
        for (const [brickID, vbID] of brickToVirtual) {
            if (checked >= 20) break;
            const vbx = vbID % vbpa;
            const vby = Math.floor(vbID / vbpa) % vbpa;
            const vbz = Math.floor(vbID / (vbpa * vbpa));
            const tcx = (brickID % bax) * bwv * bwv;
            const tcy = Math.floor(brickID / bax) * bwv;
            // Interior voxel corner (avoid the 1.0-sentinel boundary layer).
            for (const [x, y, z] of [
                [0, 0, 0],
                [3, 4, 2],
                [6, 6, 6],
            ] as const) {
                const wx = vbx * bw + x;
                const wy = vby * bw + y;
                const wz = vbz * bw + z;
                const px = tcx + x + z * bwv;
                const py = tcy + y;
                const texel = sbs.brickTexture[px + tw * py]!;
                const expected = wx < 128 && wy < 128 && wz < 128 ? field(wx, wy, wz) : 1.0;
                expect(Math.abs(texel - expected)).toBeLessThan(1e-6);
            }
            checked++;
        }
        expect(checked).toBeGreaterThan(0);
    });

    it("every valid brick contains the surface", () => {
        const gwv = 129;
        const rawField = (sbs as unknown as { sdField: Int8Array }).sdField;
        const field = (x: number, y: number, z: number) => rawField[x + gwv * (y + gwv * z)]! / 127;
        for (let i = 0; i < sbs.indirection.length; i++) {
            if (sbs.indirection[i] === 0xffffffff) continue;
            const vbx = i % vbpa;
            const vby = Math.floor(i / vbpa) % vbpa;
            const vbz = Math.floor(i / (vbpa * vbpa));
            let anyNeg = false;
            let anyPos = false;
            for (let vz = vbz * bw; vz < Math.min((vbz + 1) * bw, 128); vz++)
                for (let vy = vby * bw; vy < Math.min((vby + 1) * bw, 128); vy++)
                    for (let vx = vbx * bw; vx < Math.min((vbx + 1) * bw, 128); vx++) {
                        for (let c = 0; c < 8; c++) {
                            const v = field(vx + (c & 1), vy + ((c >> 1) & 1), vz + ((c >> 2) & 1));
                            if (v <= 0) anyNeg = true;
                            if (v >= 0) anyPos = true;
                        }
                    }
            expect(anyNeg && anyPos).toBe(true);
        }
    });
});
