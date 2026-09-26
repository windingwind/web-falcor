/**
 * CPU tasks the worker pool runs (Utils/Threading/WorkerPool.ts). Each takes and returns
 * structured-cloneable values; the pool transfers the returned `transfer` buffers.
 */

import { decodeDDSToRGBA } from "../../Scene/Importer/DDSLoader.js";
import { decodeTGA } from "../Image/TGADecoder.js";
import { buildBvhSubtree, splitTopLevels, stableSortKeys, type BvhInput, type BvhSubtree, type TopSplitNode } from "../../Scene/SoftwareRT/Bvh.js";
import { generateTangentsAndMerge } from "../../Scene/TangentSpace.js";
import { createPackedVertices, kPackedVertexFloats, PackedVertex } from "../../Scene/SceneData.js";

export interface TaskResult<T> {
    value: T;
    transfer?: Transferable[];
}

export const kTasks = {
    /** decodeDDSToRGBA (the capped CPU decode for texture analysis). */
    decodeDDS(args: { buffer: ArrayBuffer; srgb: boolean; maxDim: number }): TaskResult<{ width: number; height: number; rgba: Uint8Array }> {
        const image = decodeDDSToRGBA(args.buffer, args.srgb, args.maxDim);
        return { value: image, transfer: [image.rgba.buffer] };
    },
    /** decodeTGA. */
    decodeTGA(args: { buffer: ArrayBuffer }): TaskResult<ReturnType<typeof decodeTGA>> {
        const image = decodeTGA(args.buffer);
        return { value: image, transfer: [image.rgba.buffer as ArrayBuffer] };
    },
    /** One subtree of buildBvhParallel. */
    buildBvhSubtree(args: BvhInput): TaskResult<BvhSubtree> {
        const tree = buildBvhSubtree(args);
        return { value: tree, transfer: [tree.nodes.buffer, tree.ordered.buffer] };
    },
    /** The top median splits of one half of buildBvhParallel (its levels below the root run in parallel). */
    splitTopLevels(args: { input: BvhInput; depth: number }): TaskResult<{ index: Uint32Array; tree: TopSplitNode }> {
        const r = splitTopLevels(args.input, args.depth);
        return { value: r, transfer: [r.index.buffer] };
    },
    /** One chunk of the BVH root split's stable sort. */
    stableSortKeys(args: { keys: Float64Array }): TaskResult<Uint32Array> {
        const order = stableSortKeys(args.keys);
        return { value: order, transfer: [order.buffer] };
    },
    /** generateTangentsAndMerge on one packed vertex store (MikkTSpace; the worker loads its wasm first). */
    tangentsAndMerge(args: { data: Float32Array; indices: Uint32Array; boneIDs?: Uint32Array; boneWeights?: Float32Array }): TaskResult<{ data: Float32Array; indices: Uint32Array; source: Uint32Array } | null> {
        const vertices = createPackedVertices(args.data.length / kPackedVertexFloats, args.data as Float32Array<ArrayBuffer>);
        const r = generateTangentsAndMerge(vertices, args.indices, { boneIDs: args.boneIDs, boneWeights: args.boneWeights });
        if (!r) return { value: null };
        const data = (r.vertices[0] as PackedVertex | undefined)?.data ?? new Float32Array(0);
        return { value: { data, indices: r.indices, source: r.source }, transfer: [data.buffer, r.indices.buffer, r.source.buffer] };
    },
} satisfies Record<string, (args: never) => TaskResult<unknown>>;

export type TaskName = keyof typeof kTasks;
