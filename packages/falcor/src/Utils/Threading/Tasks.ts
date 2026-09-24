/**
 * CPU tasks the worker pool runs (Utils/Threading/WorkerPool.ts). Each takes and returns
 * structured-cloneable values; the pool transfers the returned `transfer` buffers.
 */

import { decodeDDSToRGBA } from "../../Scene/Importer/DDSLoader.js";
import { decodeTGA } from "../Image/TGADecoder.js";
import { buildBvhSubtree, type BvhInput, type BvhSubtree } from "../../Scene/SoftwareRT/Bvh.js";

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
} satisfies Record<string, (args: never) => TaskResult<unknown>>;

export type TaskName = keyof typeof kTasks;
