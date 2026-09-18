/**
 * `buildBvh` is the scene build's hot spot (two thirds of it on a 131k-triangle
 * scene), so it runs over flat index arrays instead of allocating a triangle
 * list per node. The output must stay *byte-identical* to the straightforward
 * recursive formulation: the scene cache compares BVH buffers between a fresh
 * and a cached build, and the oracle image tests assume a fixed triangle order.
 *
 * The reference below is that straightforward formulation, kept here as the
 * thing the fast path must agree with.
 */

import { describe, expect, it } from "vitest";
import { buildBvh, type BvhTriangle } from "../src/Scene/SoftwareRT/Bvh.js";
import { float3, sub3 } from "../src/Utils/Math/Vector.js";

/** Median-split BVH written the obvious way: one entry list per node. */
function buildBvhReference(triangles: BvhTriangle[]): { nodes: Float32Array; tris: Float32Array; nodeCount: number } {
    interface Entry {
        triIndex: number;
        centroid: float3;
        min: float3;
        max: float3;
    }
    const entries: Entry[] = triangles.map((t, i) => {
        const min = new float3(Math.min(t.v0.x, t.v1.x, t.v2.x), Math.min(t.v0.y, t.v1.y, t.v2.y), Math.min(t.v0.z, t.v1.z, t.v2.z));
        const max = new float3(Math.max(t.v0.x, t.v1.x, t.v2.x), Math.max(t.v0.y, t.v1.y, t.v2.y), Math.max(t.v0.z, t.v1.z, t.v2.z));
        return { triIndex: i, centroid: new float3((min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2), min, max };
    });

    const nodes = new Float32Array(2 * triangles.length * 8 + 8);
    const nodesU32 = new Uint32Array(nodes.buffer);
    let nodeCount = 0;
    const orderedTris: number[] = [];

    if (triangles.length === 0) {
        nodesU32[3] = 0;
        nodesU32[7] = 1;
        return { nodes: nodes.subarray(0, 8), tris: new Float32Array(12), nodeCount: 1 };
    }

    const writeNode = (index: number, min: float3, max: float3, leftFirst: number, count: number) => {
        nodes[index * 8 + 0] = min.x;
        nodes[index * 8 + 1] = min.y;
        nodes[index * 8 + 2] = min.z;
        nodesU32[index * 8 + 3] = leftFirst;
        nodes[index * 8 + 4] = max.x;
        nodes[index * 8 + 5] = max.y;
        nodes[index * 8 + 6] = max.z;
        nodesU32[index * 8 + 7] = count;
    };

    const bounds = (list: Entry[]): [float3, float3] => {
        const min = new float3(Infinity, Infinity, Infinity);
        const max = new float3(-Infinity, -Infinity, -Infinity);
        for (const e of list) {
            min.x = Math.min(min.x, e.min.x); min.y = Math.min(min.y, e.min.y); min.z = Math.min(min.z, e.min.z);
            max.x = Math.max(max.x, e.max.x); max.y = Math.max(max.y, e.max.y); max.z = Math.max(max.z, e.max.z);
        }
        return [min, max];
    };

    const kLeafSize = 4;
    const build = (list: Entry[]): number => {
        const nodeIndex = nodeCount++;
        const [min, max] = bounds(list);
        if (list.length <= kLeafSize) {
            writeNode(nodeIndex, min, max, orderedTris.length, list.length);
            for (const e of list) orderedTris.push(e.triIndex);
            return nodeIndex;
        }
        const extent = sub3(max, min);
        const axis = extent.x > extent.y ? (extent.x > extent.z ? "x" : "z") : extent.y > extent.z ? "y" : "z";
        const sorted = [...list].sort((a, b) => a.centroid[axis] - b.centroid[axis]);
        const half = Math.ceil(sorted.length / 2);
        build(sorted.slice(0, half));
        const rightIndex = build(sorted.slice(half));
        writeNode(nodeIndex, min, max, rightIndex, 0);
        return nodeIndex;
    };
    build(entries);

    const tris = new Float32Array(orderedTris.length * 12);
    const trisU32 = new Uint32Array(tris.buffer);
    orderedTris.forEach((triIndex, i) => {
        const t = triangles[triIndex]!;
        const e1 = sub3(t.v1, t.v0);
        const e2 = sub3(t.v2, t.v0);
        tris[i * 12 + 0] = t.v0.x; tris[i * 12 + 1] = t.v0.y; tris[i * 12 + 2] = t.v0.z;
        trisU32[i * 12 + 3] = t.instanceIndex;
        tris[i * 12 + 4] = e1.x; tris[i * 12 + 5] = e1.y; tris[i * 12 + 6] = e1.z;
        trisU32[i * 12 + 7] = t.primitiveIndex;
        tris[i * 12 + 8] = e2.x; tris[i * 12 + 9] = e2.y; tris[i * 12 + 10] = e2.z;
        trisU32[i * 12 + 11] = 0;
    });

    return { nodes: nodes.subarray(0, nodeCount * 8), tris, nodeCount };
}

/** Deterministic pseudo-random triangle soup (mulberry32). */
function makeTriangles(count: number, seed: number, quantize = 0): BvhTriangle[] {
    let s = seed >>> 0;
    const rnd = () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    // `quantize` snaps coordinates onto a coarse lattice, which forces ties in
    // the centroid sort — where a stable and an unstable split would diverge.
    const coord = () => (quantize > 0 ? Math.round(rnd() * quantize) / quantize : rnd() * 10 - 5);
    const vertex = () => new float3(Math.fround(coord()), Math.fround(coord()), Math.fround(coord()));
    return Array.from({ length: count }, (_v, i) => ({ v0: vertex(), v1: vertex(), v2: vertex(), instanceIndex: i % 7, primitiveIndex: i }));
}

const bytes = (a: Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

describe("buildBvh", () => {
    it("matches the reference build byte for byte", () => {
        for (const count of [1, 4, 5, 17, 64, 1000]) {
            const triangles = makeTriangles(count, 1234 + count);
            const fast = buildBvh(triangles);
            const reference = buildBvhReference(triangles);
            expect(fast.nodeCount, `node count for ${count} triangles`).toBe(reference.nodeCount);
            expect(bytes(fast.nodes), `nodes for ${count} triangles`).toEqual(bytes(reference.nodes));
            expect(bytes(fast.tris), `triangles for ${count} triangles`).toEqual(bytes(reference.tris));
        }
    });

    it("keeps the same split when centroids tie", () => {
        // A coarse lattice makes many centroids equal, so an unstable partition
        // would send different triangles left and produce a different tree.
        const triangles = makeTriangles(600, 77, 4);
        const fast = buildBvh(triangles);
        const reference = buildBvhReference(triangles);
        expect(bytes(fast.nodes)).toEqual(bytes(reference.nodes));
        expect(bytes(fast.tris)).toEqual(bytes(reference.tris));
    });

    it("emits the degenerate leaf for an empty scene", () => {
        const empty = buildBvh([]);
        expect(empty.nodeCount).toBe(1);
        expect(new Uint32Array(empty.nodes.buffer)[7]).toBe(1); // one degenerate triangle
        expect(empty.tris.length).toBe(12);
    });

    it("references every triangle exactly once", () => {
        const triangles = makeTriangles(500, 9);
        const { nodes, tris, nodeCount } = buildBvh(triangles);
        const u = new Uint32Array(nodes.buffer, nodes.byteOffset);
        let leafTotal = 0;
        for (let i = 0; i < nodeCount; i++) leafTotal += u[i * 8 + 7]!;
        expect(leafTotal).toBe(triangles.length);
        expect(tris.length / 12).toBe(triangles.length);
        const seen = new Set<number>();
        for (let i = 0; i < triangles.length; i++) seen.add(new Uint32Array(tris.buffer)[i * 12 + 7]!);
        expect(seen.size).toBe(triangles.length);
    });
});
