/**
 * Software ray tracing BVH (docs §5) — WebGPU has no ray tracing API.
 *
 * CPU median-split BVH over world-space triangles; animated scenes refit it (same
 * topology and triangle order, bounds recomputed bottom-up) and rebuild once the
 * refit tree's surface area has doubled (native refits dynamic BLASes the same way).
 * Layout is consumed by the
 * Scene/RaytracingInline.slang override:
 * - nodes: 2x float4 per node:
 *     [min.xyz, bits(leftFirst)], [max.xyz, bits(triCount)]
 *   triCount > 0 => leaf (leftFirst = first triangle), else inner
 *   (left child = nodeIndex + 1, right child = leftFirst).
 * - tris: 3x float4 per triangle (Moller-Trumbore precomputed):
 *     [v0.xyz, bits(instanceIndex)], [e1.xyz, bits(primitiveIndex)], [e2.xyz, 0]
 */

import { float3, sub3 } from "../../Utils/Math/Vector.js";

export interface BvhTriangle {
    v0: float3;
    v1: float3;
    v2: float3;
    instanceIndex: number;
    primitiveIndex: number;
}

export interface BvhBuildResult {
    nodes: Float32Array; // 8 floats per node
    tris: Float32Array; // 12 floats per triangle (reordered)
    nodeCount: number;
    /** Source triangle index of each tris slot (what a refit rewrites in place). */
    order: Uint32Array;
    /** Sum of node surface areas when built (refit quality reference). */
    buildArea: number;
}

/** Sum of the nodes' AABB surface areas (the SAH cost's geometric part). */
function totalArea(nodes: Float32Array, nodeCount: number): number {
    let area = 0;
    for (let i = 0; i < nodeCount; i++) {
        const dx = nodes[i * 8 + 4]! - nodes[i * 8]!;
        const dy = nodes[i * 8 + 5]! - nodes[i * 8 + 1]!;
        const dz = nodes[i * 8 + 6]! - nodes[i * 8 + 2]!;
        area += 2 * (dx * dy + dy * dz + dz * dx);
    }
    return area;
}

/** Writes the Moller-Trumbore data of triangles[order[i]] into slot i. */
function writeTris(tris: Float32Array, triangles: BvhTriangle[], order: Uint32Array): void {
    const trisU32 = new Uint32Array(tris.buffer, tris.byteOffset, tris.length);
    for (let i = 0; i < order.length; i++) {
        const t = triangles[order[i]!]!;
        const e1 = sub3(t.v1, t.v0);
        const e2 = sub3(t.v2, t.v0);
        tris[i * 12 + 0] = t.v0.x; tris[i * 12 + 1] = t.v0.y; tris[i * 12 + 2] = t.v0.z;
        trisU32[i * 12 + 3] = t.instanceIndex;
        tris[i * 12 + 4] = e1.x; tris[i * 12 + 5] = e1.y; tris[i * 12 + 6] = e1.z;
        trisU32[i * 12 + 7] = t.primitiveIndex;
        tris[i * 12 + 8] = e2.x; tris[i * 12 + 9] = e2.y; tris[i * 12 + 10] = e2.z;
        trisU32[i * 12 + 11] = 0;
    }
}

/**
 * Refits `prev` to moved triangles (same count and order as when it was built): triangle
 * data is rewritten and node bounds are recomputed bottom-up. Falls back to a full build
 * when the triangle count changed or the refit tree's surface area exceeds twice the
 * built tree's. Returns the BVH to use (prev's arrays are reused for a refit).
 */
export function refitBvh(prev: BvhBuildResult, triangles: BvhTriangle[]): BvhBuildResult {
    if (triangles.length === 0 || triangles.length !== prev.order.length) return buildBvh(triangles);
    const { nodes, nodeCount, order } = prev;
    const nodesU32 = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
    writeTris(prev.tris, triangles, order);
    // Pre-order layout: children (i + 1 and leftFirst) come after their parent.
    for (let i = nodeCount - 1; i >= 0; i--) {
        const o = i * 8;
        const count = nodesU32[o + 7]!;
        let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
        if (count > 0) {
            const first = nodesU32[o + 3]!;
            for (let k = first; k < first + count; k++) {
                const t = triangles[order[k]!]!;
                for (const v of [t.v0, t.v1, t.v2]) {
                    x0 = Math.min(x0, v.x); y0 = Math.min(y0, v.y); z0 = Math.min(z0, v.z);
                    x1 = Math.max(x1, v.x); y1 = Math.max(y1, v.y); z1 = Math.max(z1, v.z);
                }
            }
        } else {
            for (const c of [i + 1, nodesU32[o + 3]!]) {
                const co = c * 8;
                x0 = Math.min(x0, nodes[co]!); y0 = Math.min(y0, nodes[co + 1]!); z0 = Math.min(z0, nodes[co + 2]!);
                x1 = Math.max(x1, nodes[co + 4]!); y1 = Math.max(y1, nodes[co + 5]!); z1 = Math.max(z1, nodes[co + 6]!);
            }
        }
        nodes[o] = x0; nodes[o + 1] = y0; nodes[o + 2] = z0;
        nodes[o + 4] = x1; nodes[o + 5] = y1; nodes[o + 6] = z1;
    }
    if (totalArea(nodes, nodeCount) > 2 * prev.buildArea) return buildBvh(triangles);
    return prev;
}

interface BuildEntry {
    triIndex: number;
    centroid: float3;
    min: float3;
    max: float3;
}

export interface AabbBvhResult {
    /** 8 floats per node: [min.xyz, bits(leftFirst)], [max.xyz, bits(count)]. */
    nodes: Float32Array;
    /** Ordered primitive indices (leaf reads [leftFirst, leftFirst+count)). */
    primIndices: Uint32Array;
    nodeCount: number;
}

/**
 * Median-split BVH over procedural AABBs, consumed by the SDF brick/voxel
 * traversal in the RaytracingInline override (SBS bricks, SVS voxels). Same
 * node layout as buildBvh; leaves hold primitive indices instead of triangles.
 * A ray prunes tens of thousands of voxel AABBs to the handful it crosses,
 * which the flat per-primitive loop could not do at interactive speed.
 */
export function buildAabbBvh(aabbs: { min: [number, number, number]; max: [number, number, number] }[]): AabbBvhResult {
    if (aabbs.length === 0) {
        // Root LEAF with zero primitives (same empty-scene reasoning as buildBvh).
        const nodes = new Float32Array(8);
        new Uint32Array(nodes.buffer)[7] = 0;
        return { nodes, primIndices: new Uint32Array(1), nodeCount: 1 };
    }

    interface Entry {
        index: number;
        cx: number;
        cy: number;
        cz: number;
        min: [number, number, number];
        max: [number, number, number];
    }
    const entries: Entry[] = aabbs.map((a, i) => ({
        index: i,
        cx: (a.min[0] + a.max[0]) / 2,
        cy: (a.min[1] + a.max[1]) / 2,
        cz: (a.min[2] + a.max[2]) / 2,
        min: a.min,
        max: a.max,
    }));

    const nodes = new Float32Array(2 * aabbs.length * 8 + 8);
    const nodesU32 = new Uint32Array(nodes.buffer);
    let nodeCount = 0;
    const ordered: number[] = [];
    const kLeafSize = 4;

    const bounds = (list: Entry[]): [number[], number[]] => {
        const mn = [Infinity, Infinity, Infinity];
        const mx = [-Infinity, -Infinity, -Infinity];
        for (const e of list)
            for (let c = 0; c < 3; c++) {
                mn[c] = Math.min(mn[c]!, e.min[c]!);
                mx[c] = Math.max(mx[c]!, e.max[c]!);
            }
        return [mn, mx];
    };

    const write = (index: number, mn: number[], mx: number[], leftFirst: number, count: number) => {
        nodes[index * 8 + 0] = mn[0]!;
        nodes[index * 8 + 1] = mn[1]!;
        nodes[index * 8 + 2] = mn[2]!;
        nodesU32[index * 8 + 3] = leftFirst;
        nodes[index * 8 + 4] = mx[0]!;
        nodes[index * 8 + 5] = mx[1]!;
        nodes[index * 8 + 6] = mx[2]!;
        nodesU32[index * 8 + 7] = count;
    };

    const build = (list: Entry[]): number => {
        const nodeIndex = nodeCount++;
        const [mn, mx] = bounds(list);
        if (list.length <= kLeafSize) {
            write(nodeIndex, mn, mx, ordered.length, list.length);
            for (const e of list) ordered.push(e.index);
            return nodeIndex;
        }
        const ext = [mx[0]! - mn[0]!, mx[1]! - mn[1]!, mx[2]! - mn[2]!];
        const axis = ext[0]! > ext[1]! ? (ext[0]! > ext[2]! ? 0 : 2) : ext[1]! > ext[2]! ? 1 : 2;
        const key = (e: Entry) => (axis === 0 ? e.cx : axis === 1 ? e.cy : e.cz);
        const sorted = [...list].sort((a, b) => key(a) - key(b));
        const half = Math.ceil(sorted.length / 2);
        build(sorted.slice(0, half)); // left = nodeIndex + 1
        const rightIndex = build(sorted.slice(half));
        write(nodeIndex, mn, mx, rightIndex, 0);
        return nodeIndex;
    };
    build(entries);

    return { nodes: nodes.subarray(0, nodeCount * 8), primIndices: new Uint32Array(ordered), nodeCount };
}

export function buildBvh(triangles: BvhTriangle[]): BvhBuildResult {
    const n = triangles.length;

    // Worst case 2N-1 nodes.
    const nodes = new Float32Array(2 * n * 8 + 8);
    const nodesU32 = new Uint32Array(nodes.buffer);
    let nodeCount = 0;

    // Geometry-less scenes: an all-zero root is a degenerate INTERIOR node
    // whose child pointer loops back to itself -> the traversal spins forever
    // for rays crossing the origin. An inverted (+inf/-inf) AABB is no fix:
    // slab intersectors min/max-swap per axis, so an inverted box HITS
    // everything. Emit a root LEAF with one degenerate triangle instead --
    // the leaf branch always terminates and the triangle never intersects.
    if (n === 0) {
        nodesU32[3] = 0; // leftFirst
        nodesU32[7] = 1; // triCount: one degenerate (all-zero) triangle
        return { nodes: nodes.subarray(0, 8), tris: new Float32Array(12), nodeCount: 1, order: new Uint32Array(0), buildArea: 0 };
    }

    // Per-triangle bounds and centroids in flat arrays: the build sorts ranges
    // of an index array in place instead of allocating a list per node, which
    // is what dominated the build on scenes with >100k triangles. Centroids stay
    // float64 so the sort keys are bit-identical to computing them inline.
    const bmin = new Float32Array(n * 3);
    const bmax = new Float32Array(n * 3);
    const cent = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
        const t = triangles[i]!;
        for (let c = 0; c < 3; c++) {
            const a = c === 0 ? t.v0.x : c === 1 ? t.v0.y : t.v0.z;
            const b = c === 0 ? t.v1.x : c === 1 ? t.v1.y : t.v1.z;
            const d = c === 0 ? t.v2.x : c === 1 ? t.v2.y : t.v2.z;
            const lo = Math.min(a, b, d);
            const hi = Math.max(a, b, d);
            bmin[i * 3 + c] = lo;
            bmax[i * 3 + c] = hi;
            cent[i * 3 + c] = (lo + hi) / 2;
        }
    }

    const index = new Uint32Array(n);
    for (let i = 0; i < n; i++) index[i] = i;
    const scratch = new Uint32Array(n);
    const ordered = new Uint32Array(n);
    let orderedCount = 0;

    const writeNode = (at: number, min: number[], max: number[], leftFirst: number, count: number) => {
        nodes[at * 8 + 0] = min[0]!;
        nodes[at * 8 + 1] = min[1]!;
        nodes[at * 8 + 2] = min[2]!;
        nodesU32[at * 8 + 3] = leftFirst;
        nodes[at * 8 + 4] = max[0]!;
        nodes[at * 8 + 5] = max[1]!;
        nodes[at * 8 + 6] = max[2]!;
        nodesU32[at * 8 + 7] = count;
    };

    /** Stable merge sort of index[lo, hi) by centroid on `axis` (ties keep order). */
    const sortRange = (lo: number, hi: number, axis: number) => {
        const count = hi - lo;
        if (count < 2) return;
        const mid = lo + (count >> 1);
        sortRange(lo, mid, axis);
        sortRange(mid, hi, axis);
        let i = lo;
        let j = mid;
        let k = lo;
        while (i < mid && j < hi) {
            const a = index[i]!;
            const b = index[j]!;
            scratch[k++] = cent[a * 3 + axis]! <= cent[b * 3 + axis]! ? ((i++, a)) : ((j++, b));
        }
        while (i < mid) scratch[k++] = index[i++]!;
        while (j < hi) scratch[k++] = index[j++]!;
        index.set(scratch.subarray(lo, hi), lo);
    };

    const kLeafSize = 4;

    const build = (lo: number, hi: number): number => {
        const nodeIndex = nodeCount++;
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        for (let i = lo; i < hi; i++) {
            const e = index[i]! * 3;
            for (let c = 0; c < 3; c++) {
                if (bmin[e + c]! < min[c]!) min[c] = bmin[e + c]!;
                if (bmax[e + c]! > max[c]!) max[c] = bmax[e + c]!;
            }
        }
        const count = hi - lo;
        if (count <= kLeafSize) {
            writeNode(nodeIndex, min, max, orderedCount, count);
            for (let i = lo; i < hi; i++) ordered[orderedCount++] = index[i]!;
            return nodeIndex;
        }
        // Median split on the widest centroid axis.
        const ex = max[0]! - min[0]!;
        const ey = max[1]! - min[1]!;
        const ez = max[2]! - min[2]!;
        const axis = ex > ey ? (ex > ez ? 0 : 2) : ey > ez ? 1 : 2;
        sortRange(lo, hi, axis);
        const half = Math.ceil(count / 2);
        build(lo, lo + half); // left = nodeIndex + 1 by construction order
        const rightIndex = build(lo + half, hi);
        writeNode(nodeIndex, min, max, rightIndex, 0);
        return nodeIndex;
    };
    build(0, n);

    const tris = new Float32Array(orderedCount * 12);
    const order = ordered.subarray(0, orderedCount);
    writeTris(tris, triangles, order);
    const outNodes = nodes.subarray(0, nodeCount * 8);
    return { nodes: outNodes, tris, nodeCount, order, buildArea: totalArea(outNodes, nodeCount) };
}
