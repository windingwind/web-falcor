/**
 * Software ray tracing BVH (docs §5) — WebGPU has no ray tracing API.
 *
 * v1: CPU median-split BVH over world-space triangles (static scenes; GPU LBVH
 * refit arrives with animation support). Layout is consumed by the
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
    const entries: BuildEntry[] = triangles.map((t, i) => {
        const min = new float3(
            Math.min(t.v0.x, t.v1.x, t.v2.x),
            Math.min(t.v0.y, t.v1.y, t.v2.y),
            Math.min(t.v0.z, t.v1.z, t.v2.z),
        );
        const max = new float3(
            Math.max(t.v0.x, t.v1.x, t.v2.x),
            Math.max(t.v0.y, t.v1.y, t.v2.y),
            Math.max(t.v0.z, t.v1.z, t.v2.z),
        );
        return { triIndex: i, centroid: new float3((min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2), min, max };
    });

    // Worst case 2N-1 nodes.
    const nodes = new Float32Array((2 * triangles.length) * 8 + 8);
    const nodesU32 = new Uint32Array(nodes.buffer);
    let nodeCount = 0;
    const orderedTris: number[] = [];

    // Geometry-less scenes: an all-zero root is a degenerate INTERIOR node
    // whose child pointer loops back to itself -> the traversal spins forever
    // for rays crossing the origin. An inverted (+inf/-inf) AABB is no fix:
    // slab intersectors min/max-swap per axis, so an inverted box HITS
    // everything. Emit a root LEAF with one degenerate triangle instead --
    // the leaf branch always terminates and the triangle never intersects.
    if (triangles.length === 0) {
        nodesU32[3] = 0; // leftFirst
        nodesU32[7] = 1; // triCount: one degenerate (all-zero) triangle
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

    const bounds = (list: BuildEntry[]): [float3, float3] => {
        const min = new float3(Infinity, Infinity, Infinity);
        const max = new float3(-Infinity, -Infinity, -Infinity);
        for (const e of list) {
            min.x = Math.min(min.x, e.min.x); min.y = Math.min(min.y, e.min.y); min.z = Math.min(min.z, e.min.z);
            max.x = Math.max(max.x, e.max.x); max.y = Math.max(max.y, e.max.y); max.z = Math.max(max.z, e.max.z);
        }
        return [min, max];
    };

    const kLeafSize = 4;

    const build = (list: BuildEntry[]): number => {
        const nodeIndex = nodeCount++;
        const [min, max] = bounds(list);
        if (list.length <= kLeafSize) {
            writeNode(nodeIndex, min, max, orderedTris.length, list.length);
            for (const e of list) orderedTris.push(e.triIndex);
            return nodeIndex;
        }
        // Median split on the widest centroid axis.
        const extent = sub3(max, min);
        const axis = extent.x > extent.y ? (extent.x > extent.z ? "x" : "z") : extent.y > extent.z ? "y" : "z";
        const sorted = [...list].sort((a, b) => a.centroid[axis] - b.centroid[axis]);
        const half = Math.ceil(sorted.length / 2);
        build(sorted.slice(0, half)); // left = nodeIndex + 1 by construction order
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
