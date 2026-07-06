/**
 * CPU LightBVH builder mirroring Rendering/Lights/LightBVHBuilder.cpp with the
 * default options (BinnedSAOH, maxTriangleCountPerLeaf 10, 16 bins,
 * createLeavesASAP, useLeafCreationCost, usePreintegration, useLightingCones,
 * coneUnionOld as the active cone-merge path).
 *
 * Parity notes (bit-for-bit with the native build is the goal):
 * - Float math mirrors native float ops with Math.fround at accumulation and
 *   stored results; trig runs in double then rounds (glibc acosf/sinf/cosf are
 *   not correctly rounded, so rare cost-comparison ties may flip vs native).
 * - Builder inputs must be the octahedral-DECODED triangle normals (native
 *   getMeshLightTriangles unpacks PackedEmissiveTriangle first).
 * - std::nth_element is ported as libstdc++ __introselect (median-of-3
 *   unguarded partition, insertion sort at <= 3 elements, heap-select at the
 *   2*floor(log2(n)) depth limit) so leaf triangle order matches.
 */

import {
    PackedNode,
    kInvalidCosConeAngle,
    kMaxBVHDepth,
    kMaxLeafTriangleCount,
    kMaxLeafTriangleOffset,
    type SharedNodeAttributes,
    type Vec3,
} from "./LightBVHTypes.js";
import { RuntimeError } from "../../Core/Error.js";

const f = Math.fround;
const FLT_MIN = 1.17549435e-38;

// ---- small float3/AABB helpers (Falcor float semantics) --------------------

function add3(a: Vec3, b: Vec3): Vec3 {
    return [f(a[0] + b[0]), f(a[1] + b[1]), f(a[2] + b[2])];
}
function dot3(a: Vec3, b: Vec3): number {
    return f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
}
function length3(a: Vec3): number {
    return f(Math.sqrt(dot3(a, a)));
}
/** Falcor normalize = v * (1/sqrt(dot(v,v))). */
function normalize3(a: Vec3): Vec3 {
    const invLen = f(1 / Math.sqrt(dot3(a, a)));
    return [f(a[0] * invLen), f(a[1] * invLen), f(a[2] * invLen)];
}

export class Aabb {
    min: Vec3 = [Infinity, Infinity, Infinity];
    max: Vec3 = [-Infinity, -Infinity, -Infinity];

    valid(): boolean {
        return this.min[0] <= this.max[0] && this.min[1] <= this.max[1] && this.min[2] <= this.max[2];
    }
    includePoint(p: Vec3): void {
        for (let i = 0; i < 3; i++) {
            this.min[i] = Math.min(this.min[i]!, p[i]!);
            this.max[i] = Math.max(this.max[i]!, p[i]!);
        }
    }
    includeAabb(b: Aabb): void {
        for (let i = 0; i < 3; i++) {
            this.min[i] = Math.min(this.min[i]!, b.min[i]!);
            this.max[i] = Math.max(this.max[i]!, b.max[i]!);
        }
    }
    center(): Vec3 {
        return [f(f(this.min[0] + this.max[0]) * 0.5), f(f(this.min[1] + this.max[1]) * 0.5), f(f(this.min[2] + this.max[2]) * 0.5)];
    }
    extent(): Vec3 {
        return [f(this.max[0] - this.min[0]), f(this.max[1] - this.min[1]), f(this.max[2] - this.min[2])];
    }
    area(): number {
        const e = this.extent();
        return f(f(f(f(e[0] * e[1]) + f(e[0] * e[2])) + f(e[1] * e[2])) * 2);
    }
    clone(): Aabb {
        const b = new Aabb();
        b.min = [...this.min];
        b.max = [...this.max];
        return b;
    }
}

function safeACos(v: number): number {
    return f(Math.acos(Math.min(Math.max(v, -1), 1)));
}
function sinFromCos(c: number): number {
    return f(Math.sqrt(Math.max(0, f(1 - f(c * c)))));
}

/** Minimum cone angle including a second bounding cone (LightBVHBuilder.cpp). */
export function computeCosConeAngle(coneDir: Vec3, cosTheta: number, otherConeDir: Vec3, cosOtherTheta: number): number {
    let cosResult = kInvalidCosConeAngle;
    if (cosTheta !== kInvalidCosConeAngle && cosOtherTheta !== kInvalidCosConeAngle) {
        const cosDiffTheta = dot3(coneDir, otherConeDir);
        const sinDiffTheta = sinFromCos(cosDiffTheta);
        const sinOtherTheta = sinFromCos(cosOtherTheta);
        const cosTotalTheta = f(f(cosOtherTheta * cosDiffTheta) - f(sinOtherTheta * sinDiffTheta));
        const sinTotalTheta = f(f(sinOtherTheta * cosDiffTheta) + f(cosOtherTheta * sinDiffTheta));
        if (sinTotalTheta > 0) cosResult = Math.min(cosTheta, cosTotalTheta);
    }
    return cosResult;
}

/** coneUnionOld — the active cone-merge path in computeLightingConesInternal. */
export function coneUnionOld(aDir: Vec3, aCosTheta: number, bDir: Vec3, bCosTheta: number): { dir: Vec3; cos: number } {
    let dir = add3(aDir, bDir);
    if (aCosTheta === kInvalidCosConeAngle || bCosTheta === kInvalidCosConeAngle || (dir[0] === 0 && dir[1] === 0 && dir[2] === 0)) {
        return { dir: [0, 0, 0], cos: kInvalidCosConeAngle };
    }
    dir = normalize3(dir);
    const aDiff = safeACos(dot3(dir, aDir));
    const bDiff = safeACos(dot3(dir, bDir));
    const cos = f(Math.cos(Math.max(f(aDiff + f(Math.acos(aCosTheta))), f(bDiff + f(Math.acos(bCosTheta))))));
    return { dir, cos };
}

// ---- libstdc++ nth_element (introselect) ------------------------------------

type Comp = (a: number, b: number) => boolean; // strict less on element indices' values

function medianOf3<T>(arr: T[], a: number, b: number, c: number, less: (x: T, y: T) => boolean): number {
    // libstdc++ __median(a,b,c): returns iterator of median value.
    if (less(arr[a]!, arr[b]!)) {
        if (less(arr[b]!, arr[c]!)) return b;
        else if (less(arr[a]!, arr[c]!)) return c;
        else return a;
    } else if (less(arr[a]!, arr[c]!)) return a;
    else if (less(arr[b]!, arr[c]!)) return c;
    else return b;
}

function insertionSort<T>(arr: T[], first: number, last: number, less: (x: T, y: T) => boolean): void {
    for (let i = first + 1; i < last; i++) {
        const val = arr[i]!;
        let j = i - 1;
        while (j >= first && less(val, arr[j]!)) {
            arr[j + 1] = arr[j]!;
            j--;
        }
        arr[j + 1] = val;
    }
}

function heapSelect<T>(arr: T[], first: number, middle: number, last: number, less: (x: T, y: T) => boolean): void {
    // Partial sort [first, middle): make heap of [first, middle), then sift.
    const heapLen = middle - first;
    const siftDown = (start: number, end: number) => {
        let root = start;
        for (;;) {
            let child = 2 * (root - first) + 1 + first;
            if (child >= end) break;
            if (child + 1 < end && less(arr[child]!, arr[child + 1]!)) child++;
            if (less(arr[root]!, arr[child]!)) {
                const t = arr[root]!;
                arr[root] = arr[child]!;
                arr[child] = t;
                root = child;
            } else break;
        }
    };
    for (let start = first + Math.floor((heapLen - 2) / 2); start >= first; start--) siftDown(start, first + heapLen);
    for (let i = middle; i < last; i++) {
        if (less(arr[i]!, arr[first]!)) {
            const t = arr[first]!;
            arr[first] = arr[i]!;
            arr[i] = t;
            siftDown(first, first + heapLen);
        }
    }
}

function unguardedPartition<T>(arr: T[], first: number, last: number, pivotIdx: number, less: (x: T, y: T) => boolean): number {
    // libstdc++ __unguarded_partition_pivot: swap pivot to first, partition (first+1, last) with pivot arr[first].
    const t0 = arr[first]!;
    arr[first] = arr[pivotIdx]!;
    arr[pivotIdx] = t0;
    const pivot = arr[first]!;
    let lo = first + 1;
    let hi = last - 1;
    for (;;) {
        while (less(arr[lo]!, pivot)) lo++;
        while (less(pivot, arr[hi]!)) hi--;
        if (lo >= hi) return lo;
        const t = arr[lo]!;
        arr[lo] = arr[hi]!;
        arr[hi] = t;
        lo++;
        hi--;
    }
}

/** libstdc++ std::nth_element (__introselect) over arr[first, last). */
export function nthElement<T>(arr: T[], first: number, nth: number, last: number, less: (x: T, y: T) => boolean): void {
    if (first === last || nth === last) return;
    let depthLimit = 2 * Math.floor(Math.log2(last - first));
    while (last - first > 3) {
        if (depthLimit === 0) {
            // __heap_select(first, nth+1, last) then swap first/nth.
            heapSelect(arr, first, nth + 1, last, less);
            const t = arr[first]!;
            arr[first] = arr[nth]!;
            arr[nth] = t;
            return;
        }
        depthLimit--;
        const mid = first + ((last - first) >> 1);
        const pivotIdx = medianOf3(arr, first + 1, mid, last - 1, less);
        // libstdc++ passes the median VALUE via iter refs: it swaps arr[first+? ]...
        // __introselect uses __unguarded_partition_pivot(first, last) with
        // __median(*(first+1), *(first+(last-first)/2), *(last-1)) moved to first.
        const cut = unguardedPartition(arr, first, last, pivotIdx, less);
        if (cut <= nth) first = cut;
        else last = cut;
    }
    insertionSort(arr, first, last, less);
}

// ---- builder ----------------------------------------------------------------

export interface LightBVHOptions {
    maxTriangleCountPerLeaf: number;
    binCount: number;
    volumeEpsilon: number;
    splitAlongLargest: boolean;
    useVolumeOverSA: boolean;
    useLeafCreationCost: boolean;
    createLeavesASAP: boolean;
    usePreintegration: boolean;
    useLightingCones: boolean;
}

export const kDefaultLightBVHOptions: LightBVHOptions = {
    maxTriangleCountPerLeaf: 10,
    binCount: 16,
    volumeEpsilon: 1e-3,
    splitAlongLargest: false,
    useVolumeOverSA: false,
    useLeafCreationCost: true,
    createLeavesASAP: true,
    usePreintegration: true,
    useLightingCones: true,
};

export interface EmissiveTriangleInput {
    /** World-space vertex positions. */
    posW: [Vec3, Vec3, Vec3];
    /** Octahedral-DECODED face normal (decodeNormal2x16Host(encodeNormal2x16Host(n))). */
    normal: Vec3;
    flux: number;
}

interface TriangleSortData {
    bounds: Aabb;
    center: Vec3;
    coneDirection: Vec3;
    cosConeAngle: number;
    flux: number;
    triangleIndex: number;
}

export interface LightBVHBuildResult {
    /** Packed nodes, 32B each. */
    nodes: ArrayBuffer;
    nodeCount: number;
    triangleIndices: Uint32Array;
    /** Per-global-triangle 64-bit traversal bitmask as (lo, hi) uint pairs. */
    triangleBitmasks: Uint32Array;
    valid: boolean;
}

interface SplitResult {
    axis: number;
    triangleIndex: number;
}

interface BuildingData {
    trianglesData: TriangleSortData[];
    nodes: PackedNode[];
    triangleIndices: number[];
    bitmasksLo: Uint32Array;
    bitmasksHi: Uint32Array;
    currentNodeFlux: number;
}

function evalSAH(bounds: Aabb, triangleCount: number, o: LightBVHOptions): number {
    const aabbCost = bounds.valid() ? (o.useVolumeOverSA ? aabbVolume(bounds, o.volumeEpsilon) : bounds.area()) : 0;
    return f(aabbCost * triangleCount);
}

function aabbVolume(bb: Aabb, epsilon: number): number {
    if (!bb.valid()) return -Infinity;
    const e = bb.extent();
    const dx = Math.max(epsilon, e[0]);
    const dy = Math.max(epsilon, e[1]);
    const dz = Math.max(epsilon, e[2]);
    return f(f(dx * dy) * dz);
}

/** Conty & Kulla orientation cost (flat diffuse emitters, theta_e = pi/2). */
function computeOrientationCost(theta_o: number): number {
    const theta_w = Math.min(f(theta_o + Math.PI / 2), Math.PI);
    const sin_theta_o = f(Math.sin(theta_o));
    const cos_theta_o = f(Math.cos(theta_o));
    return f(
        f(2 * Math.PI * f(1 - cos_theta_o)) +
            f(
                (Math.PI / 2) *
                    f(f(f(2 * theta_w * sin_theta_o) - f(Math.cos(f(theta_o - 2 * theta_w)))) - f(f(2 * theta_o * sin_theta_o) - cos_theta_o)),
            ),
    );
}

function evalSAOH(bounds: Aabb, flux: number, cosTheta: number, o: LightBVHOptions): number {
    const fluxCost = o.usePreintegration ? flux : 1;
    const aabbCost = bounds.valid() ? (o.useVolumeOverSA ? aabbVolume(bounds, o.volumeEpsilon) : bounds.area()) : 0;
    const theta = cosTheta !== kInvalidCosConeAngle ? safeACos(cosTheta) : Math.PI;
    const orientationCost = o.useLightingCones ? computeOrientationCost(theta) : 1;
    return f(f(fluxCost * aabbCost) * orientationCost);
}

function computeLightingCone(begin: number, end: number, data: BuildingData): { dir: Vec3; cosTheta: number } {
    let coneDirection: Vec3 = [0, 0, 0];
    let cosTheta = kInvalidCosConeAngle;
    let sum: Vec3 = [0, 0, 0];
    for (let i = begin; i < end; i++) sum = add3(sum, data.trianglesData[i]!.coneDirection);
    if (length3(sum) >= FLT_MIN) {
        coneDirection = normalize3(sum);
        cosTheta = 1;
        for (let i = begin; i < end; i++) {
            const td = data.trianglesData[i]!;
            cosTheta = computeCosConeAngle(coneDirection, cosTheta, td.coneDirection, td.cosConeAngle);
        }
    }
    return { dir: coneDirection, cosTheta };
}

function computeSplitWithEqual(_data: BuildingData, begin: number, end: number, nodeBounds: Aabb): SplitResult {
    const d = nodeBounds.extent();
    const dim = d[2] >= d[0] && d[2] >= d[1] ? 2 : d[1] >= d[0] ? 1 : 0;
    return { axis: dim, triangleIndex: begin + ((end - begin) >> 1) };
}

function computeSplitWithBinnedSAOH(data: BuildingData, begin: number, end: number, nodeBounds: Aabb, o: LightBVHOptions): SplitResult | null {
    let bestCost = Infinity;
    let best: SplitResult | null = null;

    const dims = nodeBounds.extent();
    const largestDimension = dims[2] >= dims[0] && dims[2] >= dims[1] ? 2 : dims[1] >= dims[0] && dims[1] >= dims[2] ? 1 : 0;

    interface Bin {
        bounds: Aabb;
        triangleCount: number;
        flux: number;
        coneDirection: Vec3;
        cosConeAngle: number;
    }
    const newBin = (): Bin => ({ bounds: new Aabb(), triangleCount: 0, flux: 0, coneDirection: [0, 0, 0], cosConeAngle: 1 });
    const mergeTd = (bin: Bin, td: TriangleSortData) => {
        bin.bounds.includeAabb(td.bounds);
        bin.triangleCount += 1;
        bin.flux = f(bin.flux + td.flux);
        bin.coneDirection = add3(bin.coneDirection, td.coneDirection);
    };
    const mergeBin = (a: Bin, b: Bin) => {
        a.bounds.includeAabb(b.bounds);
        a.triangleCount += b.triangleCount;
        a.flux = f(a.flux + b.flux);
        a.coneDirection = add3(a.coneDirection, b.coneDirection);
    };

    const bins: Bin[] = Array.from({ length: o.binCount }, newBin);
    const costs = new Float32Array(o.binCount - 1);

    const binAlongDimension = (dimension: number) => {
        const bmin = nodeBounds.min[dimension]!;
        const bmax = nodeBounds.max[dimension]!;
        const w = f(bmax - bmin);
        const scale = w > FLT_MIN ? f(o.binCount / w) : 0;
        const getBinId = (td: TriangleSortData) => Math.min(Math.trunc(f(f(td.bounds.center()[dimension]! - bmin) * scale)), o.binCount - 1);

        for (let i = 0; i < bins.length; i++) bins[i] = newBin();
        for (let i = begin; i < end; i++) mergeTd(bins[getBinId(data.trianglesData[i]!)]!, data.trianglesData[i]!);

        for (const bin of bins) {
            bin.cosConeAngle = length3(bin.coneDirection) < FLT_MIN ? kInvalidCosConeAngle : 1;
            bin.coneDirection = normalize3(bin.coneDirection);
        }
        for (let i = begin; i < end; i++) {
            const td = data.trianglesData[i]!;
            const bin = bins[getBinId(td)]!;
            bin.cosConeAngle = computeCosConeAngle(bin.coneDirection, bin.cosConeAngle, td.coneDirection, td.cosConeAngle);
        }

        // Left-to-right sweep.
        let total = newBin();
        for (let i = 0; i < costs.length; i++) {
            mergeBin(total, bins[i]!);
            let cosTheta = kInvalidCosConeAngle;
            if (length3(total.coneDirection) >= FLT_MIN) {
                cosTheta = 1;
                const coneDir = normalize3(total.coneDirection);
                for (let j = 0; j <= i; j++) cosTheta = computeCosConeAngle(coneDir, cosTheta, bins[j]!.coneDirection, bins[j]!.cosConeAngle);
            }
            costs[i] = evalSAOH(total.bounds, total.flux, cosTheta, o);
        }
        // Right-to-left sweep.
        total = newBin();
        for (let i = costs.length; i > 0; i--) {
            mergeBin(total, bins[i]!);
            let cosTheta = kInvalidCosConeAngle;
            if (length3(total.coneDirection) >= FLT_MIN) {
                cosTheta = 1;
                const coneDir = normalize3(total.coneDirection);
                for (let j = i; j <= costs.length; j++) cosTheta = computeCosConeAngle(coneDir, cosTheta, bins[j]!.coneDirection, bins[j]!.cosConeAngle);
            }
            costs[i - 1] = f(costs[i - 1]! + evalSAOH(total.bounds, total.flux, cosTheta, o));
        }

        let axisBestCost = Infinity;
        let axisBest: SplitResult = { axis: dimension, triangleIndex: 0 };
        for (let i = 0, triIdx = begin; i < costs.length; i++) {
            triIdx += bins[i]!.triangleCount;
            if (costs[i]! < axisBestCost) {
                axisBestCost = costs[i]!;
                axisBest = { axis: dimension, triangleIndex: triIdx };
            }
        }
        // Scale by extent ratio to discourage long skinny nodes.
        axisBestCost = f(axisBestCost * f(dims[largestDimension]! / dims[dimension]!));

        if (axisBest.triangleIndex === begin || axisBest.triangleIndex === end) return;
        if (axisBestCost < bestCost) {
            bestCost = axisBestCost;
            best = axisBest;
        }
    };

    if (o.splitAlongLargest) binAlongDimension(largestDimension);
    else for (let d = 0; d < 3; d++) binAlongDimension(d);

    if (!best) {
        if (end - begin <= o.maxTriangleCountPerLeaf) return null;
        return computeSplitWithEqual(data, begin, end, nodeBounds);
    }
    if (o.useLeafCreationCost && end - begin <= o.maxTriangleCountPerLeaf) {
        const { cosTheta } = computeLightingCone(begin, end, data);
        const leafCost = evalSAOH(nodeBounds, data.currentNodeFlux, cosTheta, o);
        if (leafCost <= bestCost) return null;
    }
    return best;
}

function buildInternal(
    o: LightBVHOptions,
    bitmaskLo: number,
    bitmaskHi: number,
    depth: number,
    begin: number,
    end: number,
    data: BuildingData,
): number {
    let nodeFlux = 0;
    const nodeBounds = new Aabb();
    for (let i = begin; i < end; i++) {
        nodeBounds.includeAabb(data.trianglesData[i]!.bounds);
        nodeFlux = f(nodeFlux + data.trianglesData[i]!.flux);
    }
    data.currentNodeFlux = nodeFlux;

    const trySplitting = end - begin > (o.createLeavesASAP ? o.maxTriangleCountPerLeaf : 1);
    const splitResult = trySplitting ? computeSplitWithBinnedSAOH(data, begin, end, nodeBounds, o) : null;

    if (splitResult) {
        const dim = splitResult.axis;
        nthElement(
            data.trianglesData,
            begin,
            splitResult.triangleIndex,
            end,
            (d1, d2) => d1.bounds.center()[dim]! < d2.bounds.center()[dim]!,
        );

        const nodeIndex = data.nodes.length;
        data.nodes.push(new PackedNode());

        if (depth >= kMaxBVHDepth) throw new RuntimeError(`LightBVH depth ${depth + 1} exceeds maximum ${kMaxBVHDepth}`);

        // Traversal bitmask: right child sets bit `depth`.
        const rightLo = depth < 32 ? (bitmaskLo | (1 << depth)) >>> 0 : bitmaskLo;
        const rightHi = depth >= 32 ? (bitmaskHi | (1 << (depth - 32))) >>> 0 : bitmaskHi;
        buildInternal(o, bitmaskLo, bitmaskHi, depth + 1, begin, splitResult.triangleIndex, data);
        const rightIndex = buildInternal(o, rightLo, rightHi, depth + 1, splitResult.triangleIndex, end, data);

        const attribs: SharedNodeAttributes = {
            origin: nodeBounds.center(),
            extent: [f(f(nodeBounds.max[0] - nodeBounds.min[0]) * 0.5), f(f(nodeBounds.max[1] - nodeBounds.min[1]) * 0.5), f(f(nodeBounds.max[2] - nodeBounds.min[2]) * 0.5)],
            flux: nodeFlux,
            cosConeAngle: kInvalidCosConeAngle, // computed later by the cone pass
            coneDirection: [0, 0, 0],
        };
        data.nodes[nodeIndex]!.setInternalNode(rightIndex, attribs);
        return nodeIndex;
    } else {
        const nodeIndex = data.nodes.length;
        data.nodes.push(new PackedNode());

        const { dir, cosTheta } = computeLightingCone(begin, end, data);
        const attribs: SharedNodeAttributes = {
            origin: nodeBounds.center(),
            extent: [f(f(nodeBounds.max[0] - nodeBounds.min[0]) * 0.5), f(f(nodeBounds.max[1] - nodeBounds.min[1]) * 0.5), f(f(nodeBounds.max[2] - nodeBounds.min[2]) * 0.5)],
            flux: nodeFlux,
            cosConeAngle: cosTheta,
            coneDirection: dir,
        };
        const triangleOffset = data.triangleIndices.length;
        for (let i = begin; i < end; i++) {
            const globalTriangleIndex = data.trianglesData[i]!.triangleIndex;
            data.triangleIndices.push(globalTriangleIndex);
            data.bitmasksLo[globalTriangleIndex] = bitmaskLo;
            data.bitmasksHi[globalTriangleIndex] = bitmaskHi;
        }
        data.nodes[nodeIndex]!.setLeafNode(end - begin, triangleOffset, attribs);
        return nodeIndex;
    }
}

/** Post-build per-node lighting cone pass: leaves feed back their QUANTIZED
 *  attributes; internal cones propagate unquantized upward. */
function computeLightingConesInternal(nodeIndex: number, data: BuildingData): { dir: Vec3; cos: number } {
    const packed = data.nodes[nodeIndex]!;
    if (!packed.isLeaf()) {
        const rightChildIdx = packed.getRightChildIdx();
        const left = computeLightingConesInternal(nodeIndex + 1, data);
        const right = computeLightingConesInternal(rightChildIdx, data);
        const union = coneUnionOld(left.dir, left.cos, right.dir, right.cos);
        const attribs = packed.getNodeAttributes();
        // Preserve the exact unquantized origin/extent/flux? Native re-reads the
        // quantized attributes and repacks — mirror that exactly.
        attribs.cosConeAngle = union.cos;
        attribs.coneDirection = union.dir;
        packed.setNodeAttributes(attribs);
        return union;
    }
    const attribs = packed.getNodeAttributes();
    return { dir: attribs.coneDirection, cos: attribs.cosConeAngle };
}

export function buildLightBVH(triangles: EmissiveTriangleInput[], options: LightBVHOptions = kDefaultLightBVHOptions): LightBVHBuildResult {
    const data: BuildingData = {
        trianglesData: [],
        nodes: [],
        triangleIndices: [],
        bitmasksLo: new Uint32Array(triangles.length).fill(0xffffffff),
        bitmasksHi: new Uint32Array(triangles.length).fill(0xffffffff),
        currentNodeFlux: 0,
    };

    triangles.forEach((tri, i) => {
        if (options.usePreintegration && !(tri.flux > 0)) return;
        const bounds = new Aabb();
        for (const p of tri.posW) bounds.includePoint(p);
        data.trianglesData.push({
            bounds,
            center: bounds.center(),
            coneDirection: tri.normal,
            cosConeAngle: 1,
            flux: tri.flux,
            triangleIndex: i,
        });
    });

    if (data.trianglesData.length === 0) {
        return { nodes: new ArrayBuffer(32), nodeCount: 0, triangleIndices: new Uint32Array(1), triangleBitmasks: new Uint32Array(2), valid: false };
    }
    if (options.maxTriangleCountPerLeaf > kMaxLeafTriangleCount) throw new RuntimeError("maxTriangleCountPerLeaf too large");
    if (data.trianglesData.length > kMaxLeafTriangleOffset + kMaxLeafTriangleCount) throw new RuntimeError("Too many emissive triangles");

    buildInternal(options, 0, 0, 0, 0, data.trianglesData.length, data);
    computeLightingConesInternal(0, data);

    const nodes = new ArrayBuffer(data.nodes.length * 32);
    const nv = new Uint32Array(nodes);
    data.nodes.forEach((n, i) => nv.set(n.data, i * 8));

    const bitmasks = new Uint32Array(triangles.length * 2);
    for (let i = 0; i < triangles.length; i++) {
        bitmasks[i * 2] = data.bitmasksLo[i]!; // low 32 bits in .x
        bitmasks[i * 2 + 1] = data.bitmasksHi[i]!;
    }

    return {
        nodes,
        nodeCount: data.nodes.length,
        triangleIndices: new Uint32Array(data.triangleIndices),
        triangleBitmasks: bitmasks,
        valid: true,
    };
}
