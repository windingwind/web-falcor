/**
 * GPU timestamp timer mirroring Falcor/Core/API/GpuTimer.h.
 * Uses WebGPU timestamp queries ('timestamp-query' feature). When the feature
 * is unavailable, begin/end are no-ops and getElapsedTime returns 0 (same
 * graceful degradation as Falcor without timing support).
 */

import type { Device } from "./Device.js";
import type { CopyContext } from "./CopyContext.js";

export class GpuTimer {
    private querySet: GPUQuerySet | null = null;
    private resolveBuffer: GPUBuffer | null = null;
    private resultBuffer: GPUBuffer | null = null;
    private elapsedMs = 0;

    constructor(private readonly device: Device) {
        if (device.hasFeature("timestamp-query")) {
            this.querySet = device.gpuDevice.createQuerySet({ type: "timestamp", count: 2 });
            this.resolveBuffer = device.gpuDevice.createBuffer({
                size: 16,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
            this.resultBuffer = device.gpuDevice.createBuffer({
                size: 16,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
        }
    }

    private writeTimestamp(ctx: CopyContext, index: number): void {
        if (!this.querySet) return;
        // Encoder-level writeTimestamp is a Chromium extension
        // ('chromium-experimental-timestamp-query-inside-passes'); the portable
        // path (pass-descriptor timestampWrites) is wired via the contexts in M2.
        const encoder = ctx.getEncoder() as GPUCommandEncoder & {
            writeTimestamp?: (querySet: GPUQuerySet, queryIndex: number) => void;
        };
        encoder.writeTimestamp?.(this.querySet, index);
    }

    /** Mirrors GpuTimer::begin — timestamps the current encoder position. */
    begin(ctx: CopyContext): void {
        this.writeTimestamp(ctx, 0);
    }

    /** Mirrors GpuTimer::end. */
    end(ctx: CopyContext): void {
        this.writeTimestamp(ctx, 1);
    }

    /** Mirrors GpuTimer::resolve + getElapsedTime (async divergence). Returns milliseconds. */
    async resolve(ctx: CopyContext): Promise<number> {
        if (!this.querySet || !this.resolveBuffer || !this.resultBuffer) return 0;
        const encoder = ctx.getEncoder();
        encoder.resolveQuerySet(this.querySet, 0, 2, this.resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, 16);
        ctx.submit();
        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        const values = new BigUint64Array(this.resultBuffer.getMappedRange().slice(0));
        this.resultBuffer.unmap();
        this.elapsedMs = Number(values[1]! - values[0]!) / 1e6;
        return this.elapsedMs;
    }

    getElapsedTime(): number {
        return this.elapsedMs;
    }

    destroy(): void {
        this.querySet?.destroy();
        this.resolveBuffer?.destroy();
        this.resultBuffer?.destroy();
    }
}
