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
    /** Bit 0: the instance transform flips the winding (face culling reads it). */
    flags?: number;
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

/** Traversal stack entries the shaders' fixed `uint stack[32]` must hold for this tree. */
export const kBvhTraversalStackSize = 32;

/** Most right children pending at once in the shaders' DFS traversal (left = node + 1). */
export function bvhStackDepth(nodes: Float32Array): number {
    const u = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
    let max = 0;
    const todo: [number, number][] = [[0, 0]];
    while (todo.length > 0) {
        const [node, pending] = todo.pop()!;
        if (pending > max) max = pending;
        if (u[node * 8 + 7]! > 0) continue;
        todo.push([u[node * 8 + 3]!, pending], [node + 1, pending + 1]);
    }
    return max;
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
        trisU32[i * 12 + 11] = t.flags ?? 0;
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

/**
 * refitBvh over flat positions: input triangle k's vertices are verts[triVerts[3k + j] * 3 ..] (float64, as the
 * float3 path computes them), so animated frames allocate no triangle objects. Returns null when the refit tree
 * degrades (the caller rebuilds); instance/primitive/flag words are left as built.
 */
export function refitBvhIndexed(prev: BvhBuildResult, verts: Float64Array, triVerts: Uint32Array): BvhBuildResult | null {
    const { nodes, nodeCount, order, tris } = prev;
    if (triVerts.length !== order.length * 3) return null;
    const nodesU32 = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
    for (let i = 0; i < order.length; i++) {
        const k = order[i]! * 3;
        const [a, b, c] = [triVerts[k]! * 3, triVerts[k + 1]! * 3, triVerts[k + 2]! * 3];
        const o = i * 12;
        tris[o] = verts[a]!; tris[o + 1] = verts[a + 1]!; tris[o + 2] = verts[a + 2]!;
        tris[o + 4] = verts[b]! - verts[a]!; tris[o + 5] = verts[b + 1]! - verts[a + 1]!; tris[o + 6] = verts[b + 2]! - verts[a + 2]!;
        tris[o + 8] = verts[c]! - verts[a]!; tris[o + 9] = verts[c + 1]! - verts[a + 1]!; tris[o + 10] = verts[c + 2]! - verts[a + 2]!;
    }
    for (let i = nodeCount - 1; i >= 0; i--) {
        const o = i * 8;
        const count = nodesU32[o + 7]!;
        let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
        if (count > 0) {
            const first = nodesU32[o + 3]!;
            for (let s = first; s < first + count; s++) {
                const k = order[s]! * 3;
                for (let j = 0; j < 3; j++) {
                    const v = triVerts[k + j]! * 3;
                    const [x, y, z] = [verts[v]!, verts[v + 1]!, verts[v + 2]!];
                    if (x < x0) x0 = x; if (y < y0) y0 = y; if (z < z0) z0 = z;
                    if (x > x1) x1 = x; if (y > y1) y1 = y; if (z > z1) z1 = z;
                }
            }
        } else {
            for (const ch of [i + 1, nodesU32[o + 3]!]) {
                const co = ch * 8;
                x0 = Math.min(x0, nodes[co]!); y0 = Math.min(y0, nodes[co + 1]!); z0 = Math.min(z0, nodes[co + 2]!);
                x1 = Math.max(x1, nodes[co + 4]!); y1 = Math.max(y1, nodes[co + 5]!); z1 = Math.max(z1, nodes[co + 6]!);
            }
        }
        nodes[o] = x0; nodes[o + 1] = y0; nodes[o + 2] = z0;
        nodes[o + 4] = x1; nodes[o + 5] = y1; nodes[o + 6] = z1;
    }
    return totalArea(nodes, nodeCount) > 2 * prev.buildArea ? null : prev;
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

/** Per-triangle bounds and centroids in flat arrays (what the median-split build consumes). */
export interface BvhInput {
    n: number;
    bmin: Float32Array;
    bmax: Float32Array;
    cent: Float64Array;
}

/** A built (sub)tree: nodes in DFS order, leaves indexing into `ordered` (input triangle indices). */
export interface BvhSubtree {
    nodes: Float32Array;
    nodeCount: number;
    ordered: Uint32Array;
}

function bvhInput(triangles: BvhTriangle[]): BvhInput {
    const n = triangles.length;
    // The build sorts ranges
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

    return { n, bmin, bmax, cent };
}

/** The median-split builder's state over one input (shared by the whole-tree and subtree builds). */
function makeBuilder(input: BvhInput, sortOnly = false) {
    const { n, bmin, bmax, cent } = input;
    // Worst case 2N-1 nodes (none when only the sort is used).
    const nodes = new Float32Array(sortOnly ? 8 : 2 * n * 8 + 8);
    const nodesU32 = new Uint32Array(nodes.buffer);
    let nodeCount = 0;
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

    // Sort keys: each float64 centroid as two uint32 words ordered like the number (sign-flipped
    // IEEE bits), so a stable radix sort gives the same permutation as a stable comparison sort.
    const keyLo = new Uint32Array(n * 3);
    const keyHi = new Uint32Array(n * 3);
    {
        const bits = new Uint32Array(cent.buffer);
        for (let i = 0; i < n * 3; i++) {
            const lo = bits[i * 2]!;
            const hi = bits[i * 2 + 1]!;
            const negative = hi >>> 31 !== 0;
            keyLo[i] = negative ? ~lo >>> 0 : lo;
            keyHi[i] = negative ? ~hi >>> 0 : (hi | 0x80000000) >>> 0;
        }
    }
    const counts = new Uint32Array(1 << 11);

    /** Stable sort of index[lo, hi) by centroid on `axis` (ties keep order): LSD radix, 11-bit digits. */
    const sortRange = (lo: number, hi: number, axis: number) => {
        const count = hi - lo;
        if (count < 2) return;
        if (count <= 32) {
            // Stable insertion sort.
            for (let i = lo + 1; i < hi; i++) {
                const v = index[i]!;
                const vh = keyHi[v * 3 + axis]!;
                const vl = keyLo[v * 3 + axis]!;
                let j = i - 1;
                while (j >= lo) {
                    const u = index[j]!;
                    const uh = keyHi[u * 3 + axis]!;
                    if (uh < vh || (uh === vh && keyLo[u * 3 + axis]! <= vl)) break;
                    index[j + 1] = u;
                    j--;
                }
                index[j + 1] = v;
            }
            return;
        }
        let src = index.subarray(lo, hi);
        let dst = scratch.subarray(lo, hi);
        for (let pass = 0; pass < 6; pass++) {
            const words = pass < 3 ? keyLo : keyHi;
            const shift = (pass % 3) * 11;
            counts.fill(0);
            for (let i = 0; i < count; i++) counts[(words[src[i]! * 3 + axis]! >>> shift) & 0x7ff]!++;
            // A digit shared by every key leaves the order unchanged.
            if (counts[(words[src[0]! * 3 + axis]! >>> shift) & 0x7ff] === count) continue;
            let sum = 0;
            for (let d = 0; d < counts.length; d++) {
                const c = counts[d]!;
                counts[d] = sum;
                sum += c;
            }
            for (let i = 0; i < count; i++) {
                const v = src[i]!;
                dst[counts[(words[v * 3 + axis]! >>> shift) & 0x7ff]!++] = v;
            }
            [src, dst] = [dst, src];
        }
        if (src.buffer === scratch.buffer) index.set(src, lo);
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
    return {
        build,
        sortRange,
        index,
        subtree: (): BvhSubtree => ({ nodes: nodes.slice(0, nodeCount * 8), nodeCount, ordered: ordered.slice(0, orderedCount) }),
    };
}

/** Builds the tree over a whole input (a worker task for buildBvhParallel's subtrees). */
export function buildBvhSubtree(input: BvhInput): BvhSubtree {
    const builder = makeBuilder(input);
    builder.build(0, input.n);
    return builder.subtree();
}

/** The empty-scene tree and the tris/order/area of a built tree. */
function finishBvh(triangles: BvhTriangles, tree: BvhSubtree): BvhBuildResult {
    const tris = new Float32Array(tree.ordered.length * 12);
    if (Array.isArray(triangles)) writeTris(tris, triangles, tree.ordered);
    else for (let i = 0; i < tree.ordered.length; i++) tris.set(triangles.packed.subarray(tree.ordered[i]! * 12, tree.ordered[i]! * 12 + 12), i * 12);
    return { nodes: tree.nodes, tris, nodeCount: tree.nodeCount, order: tree.ordered, buildArea: totalArea(tree.nodes, tree.nodeCount) };
}

/**
 * Triangles already in the builder's flat layout (see PackedBvhTriangleWriter): large scenes skip one object and
 * three float3 per triangle. `packed` holds each input triangle's final 12-float row (v0, inst, e1, prim, e2, flags).
 */
export interface PackedBvhTriangles {
    input: BvhInput;
    packed: Float32Array;
}
export type BvhTriangles = BvhTriangle[] | PackedBvhTriangles;

export function bvhTriangleCount(t: BvhTriangles): number {
    return Array.isArray(t) ? t.length : t.input.n;
}

/** Fills PackedBvhTriangles exactly as bvhInput + writeTris compute them from BvhTriangle objects. */
export class PackedBvhTriangleWriter {
    readonly result: PackedBvhTriangles;
    private readonly u32: Uint32Array;
    private count = 0;
    constructor(n: number) {
        this.result = { input: { n, bmin: new Float32Array(n * 3), bmax: new Float32Array(n * 3), cent: new Float64Array(n * 3) }, packed: new Float32Array(n * 12) };
        this.u32 = new Uint32Array(this.result.packed.buffer);
    }
    /** Appends a triangle from float64 world positions (a = v0, b = v1, c = v2). */
    add(ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number, instance: number, primitive: number, flags: number): void {
        const i = this.count++;
        const { bmin, bmax, cent } = this.result.input;
        const p = this.result.packed;
        const coords = [ax, bx, cx, ay, by, cy, az, bz, cz];
        for (let c = 0; c < 3; c++) {
            const [a, b, d] = [coords[c * 3]!, coords[c * 3 + 1]!, coords[c * 3 + 2]!];
            const lo = Math.min(a, b, d);
            const hi = Math.max(a, b, d);
            bmin[i * 3 + c] = lo;
            bmax[i * 3 + c] = hi;
            cent[i * 3 + c] = (lo + hi) / 2;
        }
        const o = i * 12;
        p[o] = ax; p[o + 1] = ay; p[o + 2] = az; this.u32[o + 3] = instance;
        p[o + 4] = bx - ax; p[o + 5] = by - ay; p[o + 6] = bz - az; this.u32[o + 7] = primitive;
        p[o + 8] = cx - ax; p[o + 9] = cy - ay; p[o + 10] = cz - az; this.u32[o + 11] = flags;
    }
}

// Geometry-less scenes: an all-zero root is a degenerate INTERIOR node
// whose child pointer loops back to itself -> the traversal spins forever
// for rays crossing the origin. An inverted (+inf/-inf) AABB is no fix:
// slab intersectors min/max-swap per axis, so an inverted box HITS
// everything. Emit a root LEAF with one degenerate triangle instead --
// the leaf branch always terminates and the triangle never intersects.
function emptyBvh(): BvhBuildResult {
    const nodes = new Float32Array(8);
    const nodesU32 = new Uint32Array(nodes.buffer);
    nodesU32[3] = 0; // leftFirst
    nodesU32[7] = 1; // triCount: one degenerate (all-zero) triangle
    return { nodes, tris: new Float32Array(12), nodeCount: 1, order: new Uint32Array(0), buildArea: 0 };
}

/**
 * One BVH over two: a root whose left child is `a` (node 1) and right child is `b`, with `a`'s triangles first.
 * Animated scenes keep static geometry in `a` (built once) and refit only `b` per frame (see refreshStitchedBvh).
 */
export function stitchBvhs(a: BvhBuildResult, b: BvhBuildResult): BvhBuildResult {
    const aTris = a.tris.length / 12;
    const nodeCount = 1 + a.nodeCount + b.nodeCount;
    const nodes = new Float32Array(nodeCount * 8);
    const u = new Uint32Array(nodes.buffer);
    nodes.set(a.nodes.subarray(0, a.nodeCount * 8), 8);
    nodes.set(b.nodes.subarray(0, b.nodeCount * 8), (1 + a.nodeCount) * 8);
    // Inner nodes point at their right child (left = node + 1); leaves at their first triangle.
    const shift = (first: number, count: number, nodeShift: number, triShift: number) => {
        for (let i = first; i < first + count; i++) {
            const o = i * 8;
            u[o + 3] = u[o + 3]! + (u[o + 7]! > 0 ? triShift : nodeShift);
        }
    };
    shift(1, a.nodeCount, 1, 0);
    shift(1 + a.nodeCount, b.nodeCount, 1 + a.nodeCount, aTris);
    u[3] = 1 + a.nodeCount; // root: right child = b's root
    u[7] = 0;
    const tris = new Float32Array(a.tris.length + b.tris.length);
    tris.set(a.tris, 0);
    tris.set(b.tris, a.tris.length);
    const order = new Uint32Array(a.order.length + b.order.length);
    order.set(a.order, 0);
    for (let i = 0; i < b.order.length; i++) order[a.order.length + i] = b.order[i]! + a.order.length;
    const result = { nodes, tris, nodeCount, order, buildArea: a.buildArea + b.buildArea };
    refreshStitchedRoot(result);
    return result;
}

/** Root bounds of a stitched BVH = its two children's. */
function refreshStitchedRoot(st: BvhBuildResult): void {
    const u = new Uint32Array(st.nodes.buffer, st.nodes.byteOffset, st.nodes.length);
    const [l, r] = [8, u[3]! * 8];
    for (let k = 0; k < 3; k++) {
        st.nodes[k] = Math.min(st.nodes[l + k]!, st.nodes[r + k]!);
        st.nodes[4 + k] = Math.max(st.nodes[l + 4 + k]!, st.nodes[r + 4 + k]!);
    }
}

/**
 * After `b` was refit in place (same topology), copies its bounds and triangles into the stitched tree.
 * Returns the float ranges that changed: [nodes offset, nodes length] and [tris offset, tris length].
 */
export function refreshStitchedBvh(st: BvhBuildResult, a: BvhBuildResult, b: BvhBuildResult): { nodes: [number, number]; tris: [number, number] } {
    const base = (1 + a.nodeCount) * 8;
    for (let i = 0; i < b.nodeCount; i++) {
        const [src, dst] = [i * 8, base + i * 8];
        for (const k of [0, 1, 2, 4, 5, 6]) st.nodes[dst + k] = b.nodes[src + k]!;
    }
    st.tris.set(b.tris, a.tris.length);
    refreshStitchedRoot(st);
    return { nodes: [base, b.nodeCount * 8], tris: [a.tris.length, b.tris.length] };
}

export function buildBvh(triangles: BvhTriangles): BvhBuildResult {
    if (bvhTriangleCount(triangles) === 0) return emptyBvh();
    return finishBvh(triangles, buildBvhSubtree(Array.isArray(triangles) ? bvhInput(triangles) : triangles.input));
}

/** The top levels of a median-split build: inner nodes with their bounds, and leaf ranges of the permuted index. */
export type TopSplitNode = { kind: "inner"; min: number[]; max: number[]; left: TopSplitNode; right: TopSplitNode } | { kind: "leaf"; lo: number; hi: number };

/**
 * The first `depth` median splits of the build over `input` (as buildBvhSubtree performs them): the index
 * permutation and the split tree whose leaves are ranges into it. Runs on the main thread or in a worker.
 */
export function splitTopLevels(input: BvhInput, depth: number): { index: Uint32Array; tree: TopSplitNode } {
    const { bmin, bmax } = input;
    const { sortRange, index } = makeBuilder(input, true);
    const split = (lo: number, hi: number, level: number): TopSplitNode => {
        const count = hi - lo;
        if (level === depth || count <= 4 * 2) return { kind: "leaf", lo, hi };
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        for (let i = lo; i < hi; i++) {
            const e = index[i]! * 3;
            for (let k = 0; k < 3; k++) {
                if (bmin[e + k]! < min[k]!) min[k] = bmin[e + k]!;
                if (bmax[e + k]! > max[k]!) max[k] = bmax[e + k]!;
            }
        }
        const ex = max[0]! - min[0]!;
        const ey = max[1]! - min[1]!;
        const ez = max[2]! - min[2]!;
        const axis = ex > ey ? (ex > ez ? 0 : 2) : ey > ez ? 1 : 2;
        sortRange(lo, hi, axis);
        const half = Math.ceil(count / 2);
        return { kind: "inner", min, max, left: split(lo, lo + half, level + 1), right: split(lo + half, hi, level + 1) };
    };
    const tree = split(0, input.n, 0);
    return { index, tree };
}

/** The input's elements index[lo..hi), compacted in that order. */
function compactInput(input: BvhInput, index: Uint32Array, lo: number, hi: number): BvhInput {
    const count = hi - lo;
    const m = new Float32Array(count * 3);
    const M = new Float32Array(count * 3);
    const c = new Float64Array(count * 3);
    const { bmin, bmax, cent } = input;
    for (let i = 0, o = 0; i < count; i++, o += 3) {
        const e = index[lo + i]! * 3;
        m[o] = bmin[e]!; m[o + 1] = bmin[e + 1]!; m[o + 2] = bmin[e + 2]!;
        M[o] = bmax[e]!; M[o + 1] = bmax[e + 1]!; M[o + 2] = bmax[e + 2]!;
        c[o] = cent[e]!; c[o + 1] = cent[e + 1]!; c[o + 2] = cent[e + 2]!;
    }
    return { n: count, bmin: m, bmax: M, cent: c };
}

/**
 * buildBvh with its subtrees built concurrently (`run`: e.g. the WorkerPool). The top `depth` levels split first
 * (the root on this thread; with `runTop`, each half's next levels concurrently too), each remaining range is built by
 * `run` from its triangles in their current order, and the subtrees are stitched in DFS order, so the result is
 * byte-identical to buildBvh.
 */
export async function buildBvhParallel(
    triangles: BvhTriangles,
    run: (input: BvhInput) => Promise<BvhSubtree>,
    depth = 3,
    runTop?: (input: BvhInput, depth: number) => Promise<{ index: Uint32Array; tree: TopSplitNode }>,
): Promise<BvhBuildResult> {
    const n = bvhTriangleCount(triangles);
    if (n === 0) return emptyBvh();
    const input = Array.isArray(triangles) ? bvhInput(triangles) : triangles.input;

    // The root split here; with runTop, every split below runs as its own task and both children proceed in parallel.
    let { index, tree } = splitTopLevels(input, runTop ? 1 : depth);
    if (runTop && tree.kind === "inner" && depth > 1) {
        const splitBelow = async (leaf: TopSplitNode, level: number): Promise<TopSplitNode> => {
            if (leaf.kind !== "leaf" || level === depth || leaf.hi - leaf.lo <= 4 * 2) return leaf;
            const { lo, hi } = leaf;
            const sub = await runTop(compactInput(input, index, lo, hi), 1);
            const old = index.slice(lo, hi);
            for (let i = 0; i < sub.index.length; i++) index[lo + i] = old[sub.index[i]!]!;
            if (sub.tree.kind !== "inner") return leaf;
            const shift = (t: TopSplitNode): TopSplitNode => (t.kind === "leaf" ? { kind: "leaf", lo: t.lo + lo, hi: t.hi + lo } : { ...t, left: shift(t.left), right: shift(t.right) });
            const inner = shift(sub.tree) as Extract<TopSplitNode, { kind: "inner" }>;
            const [left, right] = await Promise.all([splitBelow(inner.left, level + 1), splitBelow(inner.right, level + 1)]);
            return { ...inner, left, right };
        };
        const [left, right] = await Promise.all([splitBelow(tree.left, 1), splitBelow(tree.right, 1)]);
        tree = { ...tree, left, right };
    }

    // Leaves become subtree jobs over their triangles, compacted in their current order.
    type Top = { kind: "inner"; min: number[]; max: number[]; left: Top; right: Top } | { kind: "job"; lo: number; hi: number; result: Promise<BvhSubtree> };
    const jobs: Promise<BvhSubtree>[] = [];
    const toJobs = (t: TopSplitNode): Top => {
        if (t.kind === "inner") return { kind: "inner", min: t.min, max: t.max, left: toJobs(t.left), right: toJobs(t.right) };
        const result = run(compactInput(input, index, t.lo, t.hi));
        jobs.push(result);
        return { kind: "job", lo: t.lo, hi: t.hi, result };
    };
    const top = toJobs(tree);
    await Promise.all(jobs);

    const nodes = new Float32Array(2 * n * 8 + 8);
    const nodesU32 = new Uint32Array(nodes.buffer);
    const ordered = new Uint32Array(n);
    let nodeCount = 0;
    let orderedCount = 0;
    const emit = async (t: Top): Promise<number> => {
        if (t.kind === "job") {
            const sub = await t.result;
            const at = nodeCount;
            nodes.set(sub.nodes, at * 8);
            for (let k = 0; k < sub.nodeCount; k++) {
                const o = (at + k) * 8;
                if (nodesU32[o + 7]! > 0) nodesU32[o + 3] = nodesU32[o + 3]! + orderedCount; // leaf: first triangle
                else nodesU32[o + 3] = nodesU32[o + 3]! + at; // inner: right child
            }
            for (let i = 0; i < sub.ordered.length; i++) ordered[orderedCount + i] = index[t.lo + sub.ordered[i]!]!;
            nodeCount += sub.nodeCount;
            orderedCount += sub.ordered.length;
            return at;
        }
        const at = nodeCount++;
        await emit(t.left); // left = at + 1 by construction order
        const right = await emit(t.right);
        nodes.set([t.min[0]!, t.min[1]!, t.min[2]!, 0, t.max[0]!, t.max[1]!, t.max[2]!, 0], at * 8);
        nodesU32[at * 8 + 3] = right;
        nodesU32[at * 8 + 7] = 0;
        return at;
    };
    await emit(top);
    return finishBvh(triangles, { nodes: nodes.slice(0, nodeCount * 8), nodeCount, ordered: ordered.slice(0, orderedCount) });
}
