/**
 * LightBVH builder unit tests: hand-checkable trees against the native
 * algorithm (default options: BinnedSAOH, maxTriangleCountPerLeaf 10,
 * createLeavesASAP).
 */

import { describe, expect, it } from "vitest";
import { buildLightBVH, type EmissiveTriangleInput } from "../src/Rendering/Lights/LightBVHBuilder.js";
import { PackedNode, decodeNormal2x16Host, encodeNormal2x16Host } from "../src/Rendering/Lights/LightBVHTypes.js";

function quadTris(cx: number, cy: number, cz: number, flux: number): EmissiveTriangleInput[] {
    const n = decodeNormal2x16Host(encodeNormal2x16Host([0, 0, 1]));
    return [
        { posW: [[cx, cy, cz], [cx + 1, cy, cz], [cx + 1, cy + 1, cz]], normal: n, flux },
        { posW: [[cx, cy, cz], [cx + 1, cy + 1, cz], [cx, cy + 1, cz]], normal: n, flux },
    ];
}

function nodeAt(nodes: ArrayBuffer, index: number): PackedNode {
    const n = new PackedNode();
    n.data.set(new Uint32Array(nodes, index * 32, 8));
    return n;
}

describe("LightBVHBuilder", () => {
    it("creates a single leaf root for few triangles (createLeavesASAP)", () => {
        const result = buildLightBVH(quadTris(0, 0, 0, 1));
        expect(result.valid).toBe(true);
        expect(result.nodeCount).toBe(1);

        const root = nodeAt(result.nodes, 0);
        expect(root.isLeaf()).toBe(true);
        expect(root.getLeafTriangleCount()).toBe(2);
        expect(root.getLeafTriangleOffset()).toBe(0);
        expect(Array.from(result.triangleIndices)).toEqual([0, 1]);
        // Root-leaf triangles have an empty traversal path.
        expect(Array.from(result.triangleBitmasks)).toEqual([0, 0, 0, 0]);

        const attribs = root.getNodeAttributes();
        expect(attribs.origin[0]).toBeCloseTo(0.5, 5);
        expect(attribs.origin[1]).toBeCloseTo(0.5, 5);
        expect(attribs.origin[2]).toBeCloseTo(0, 5);
        expect(attribs.flux).toBeCloseTo(2, 5);
        // Degenerate exactly-parallel normals: sinTotalTheta == 0 in
        // computeCosConeAngle, so native marks the cone INVALID (-1) rather
        // than zero-angle — the port reproduces that conservative behavior.
        expect(attribs.cosConeAngle).toBe(-1);
        expect(attribs.coneDirection[2]).toBeCloseTo(1, 3);
    });

    it("splits two spatially separated clusters into an internal root", () => {
        const tris: EmissiveTriangleInput[] = [];
        for (let i = 0; i < 6; i++) tris.push(...quadTris(0, 0, i * 0.1, 1)); // 12 tris near origin
        for (let i = 0; i < 6; i++) tris.push(...quadTris(100, 0, i * 0.1, 1)); // 12 tris far away

        const result = buildLightBVH(tris);
        expect(result.valid).toBe(true);
        const root = nodeAt(result.nodes, 0);
        expect(root.isLeaf()).toBe(false);

        // Left child immediately follows the root; both children are leaves
        // covering 12 triangles each (within the 10-per-leaf limit they may
        // split further; assert structure generically).
        const left = nodeAt(result.nodes, 1);
        const right = nodeAt(result.nodes, root.getRightChildIdx());
        expect(root.getRightChildIdx()).toBeGreaterThan(1);

        // All triangles accounted for exactly once.
        expect(result.triangleIndices.length).toBe(24);
        expect(new Set(result.triangleIndices).size).toBe(24);

        // Bitmasks: triangles in the left subtree have bit0 = 0; right subtree bit0 = 1.
        const leftBounds = left.getNodeAttributes();
        for (let t = 0; t < 24; t++) {
            const lo = result.triangleBitmasks[t * 2]!;
            const inLeftCluster = tris[t]!.posW[0]![0]! < 50;
            const leftIsNearOrigin = leftBounds.origin[0]! < 50;
            expect((lo & 1) === 0).toBe(inLeftCluster === leftIsNearOrigin);
        }

        // Flux conserved at the root.
        expect(root.getNodeAttributes().flux).toBeCloseTo(24, 4);
    });

    it("culls zero-flux triangles under preintegration", () => {
        const tris = [...quadTris(0, 0, 0, 0), ...quadTris(2, 0, 0, 1)];
        const result = buildLightBVH(tris);
        expect(result.triangleIndices.length).toBe(2);
        // Culled triangles keep the invalid bitmask.
        expect(result.triangleBitmasks[0]).toBe(0xffffffff);
        expect(result.triangleBitmasks[1]).toBe(0xffffffff);
    });
});
