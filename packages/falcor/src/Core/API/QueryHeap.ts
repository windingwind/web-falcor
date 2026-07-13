/**
 * Query heap mirroring Core/API/QueryHeap.h over GPUQuerySet. Occlusion
 * queries attach to RasterPass draws (setOcclusionQuery); timestamp heaps
 * exist for ad-hoc use (the per-pass Profiler manages its own set).
 * Divergence (docs §9): pipeline-statistics queries do not exist in WebGPU;
 * results read back asynchronously.
 */

import type { ComputeContext } from "./ComputeContext.js";
import type { Device } from "./Device.js";
import { ArgumentError } from "../Error.js";

export enum QueryHeapType {
    Timestamp,
    Occlusion,
}

export class QueryHeap {
    readonly gpuQuerySet: GPUQuerySet;
    private resolveBuffer: GPUBuffer;

    constructor(
        private readonly device: Device,
        public readonly type: QueryHeapType,
        public readonly count: number,
    ) {
        if (type === QueryHeapType.Timestamp && !device.hasFeature("timestamp-query")) {
            throw new ArgumentError("QueryHeap: timestamp-query feature unavailable");
        }
        this.gpuQuerySet = device.gpuDevice.createQuerySet({
            type: type === QueryHeapType.Timestamp ? "timestamp" : "occlusion",
            count,
        });
        this.resolveBuffer = device.gpuDevice.createBuffer({
            size: count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
    }

    /** Resolves query results (timestamps in ns ticks, occlusion in samples). */
    async resolve(ctx: ComputeContext, first = 0, count = this.count - first): Promise<BigUint64Array> {
        if (first + count > this.count) throw new ArgumentError("QueryHeap.resolve out of range");
        const encoder = ctx.getEncoder();
        encoder.resolveQuerySet(this.gpuQuerySet, first, count, this.resolveBuffer, first * 8);
        const staging = this.device.gpuDevice.createBuffer({
            size: count * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        encoder.copyBufferToBuffer(this.resolveBuffer, first * 8, staging, 0, count * 8);
        ctx.submit();
        await staging.mapAsync(GPUMapMode.READ);
        const results = new BigUint64Array(staging.getMappedRange().slice(0));
        staging.destroy();
        return results;
    }

    destroy(): void {
        this.gpuQuerySet.destroy();
        this.resolveBuffer.destroy();
    }
}
