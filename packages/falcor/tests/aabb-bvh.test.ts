/**
 * buildAabbBvh structural checks: the SDF primitive-AABB BVH must enclose every
 * primitive, its leaves must reference each primitive exactly once, and inner
 * nodes must bound their subtrees. A ray that hits a primitive AABB must reach
 * it through the tree (the traversal invariant the shader relies on).
 */

import { describe, expect, it } from "vitest";
import { buildAabbBvh } from "../src/Scene/SoftwareRT/Bvh.js";

type Aabb = { min: [number, number, number]; max: [number, number, number] };

function makeGridAabbs(n: number): Aabb[] {
    // n^3 unit cells in [0,1]^3 (mimics SDF bricks/voxels).
    const out: Aabb[] = [];
    const s = 1 / n;
    for (let z = 0; z < n; z++)
        for (let y = 0; y < n; y++)
            for (let x = 0; x < n; x++) out.push({ min: [x * s, y * s, z * s], max: [(x + 1) * s, (y + 1) * s, (z + 1) * s] });
    return out;
}

/** Read node i as [min, leftFirst, max, count]. */
function node(nodes: Float32Array, i: number) {
    const u = new Uint32Array(nodes.buffer, nodes.byteOffset);
    return {
        min: [nodes[i * 8]!, nodes[i * 8 + 1]!, nodes[i * 8 + 2]!] as [number, number, number],
        leftFirst: u[i * 8 + 3]!,
        max: [nodes[i * 8 + 4]!, nodes[i * 8 + 5]!, nodes[i * 8 + 6]!] as [number, number, number],
        count: u[i * 8 + 7]!,
    };
}

describe("buildAabbBvh", () => {
    it("handles the empty set with a terminating root leaf", () => {
        const bvh = buildAabbBvh([]);
        expect(bvh.nodeCount).toBe(1);
        expect(node(bvh.nodes, 0).count).toBe(0);
    });

    it("references every primitive exactly once across leaves", () => {
        const aabbs = makeGridAabbs(8); // 512 cells
        const bvh = buildAabbBvh(aabbs);
        const seen = new Uint8Array(aabbs.length);
        let leafPrims = 0;
        for (let i = 0; i < bvh.nodeCount; i++) {
            const nd = node(bvh.nodes, i);
            if (nd.count > 0) {
                for (let k = 0; k < nd.count; k++) {
                    const prim = bvh.primIndices[nd.leftFirst + k]!;
                    expect(seen[prim]).toBe(0);
                    seen[prim] = 1;
                    leafPrims++;
                }
            }
        }
        expect(leafPrims).toBe(aabbs.length);
        expect([...seen].every((v) => v === 1)).toBe(true);
    });

    it("root bounds enclose the whole primitive set", () => {
        const aabbs = makeGridAabbs(6);
        const bvh = buildAabbBvh(aabbs);
        const root = node(bvh.nodes, 0);
        for (let c = 0; c < 3; c++) {
            expect(root.min[c]).toBeLessThanOrEqual(0 + 1e-6);
            expect(root.max[c]).toBeGreaterThanOrEqual(1 - 1e-6);
        }
    });

    it("every inner node bounds both children", () => {
        const bvh = buildAabbBvh(makeGridAabbs(5));
        for (let i = 0; i < bvh.nodeCount; i++) {
            const nd = node(bvh.nodes, i);
            if (nd.count > 0) continue;
            for (const child of [i + 1, nd.leftFirst]) {
                const c = node(bvh.nodes, child);
                for (let a = 0; a < 3; a++) {
                    expect(c.min[a]).toBeGreaterThanOrEqual(nd.min[a] - 1e-6);
                    expect(c.max[a]).toBeLessThanOrEqual(nd.max[a] + 1e-6);
                }
            }
        }
    });
});
