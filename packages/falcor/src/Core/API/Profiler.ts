/**
 * Minimal GPU profiler (subset of Falcor's Profiler): per-render-pass GPU
 * times via the standard WebGPU pass-descriptor `timestampWrites`. The
 * RenderGraph labels each pass; the contexts attach timestamp writes to
 * every GPU pass begun while a label is active. Times accumulate per label
 * within a frame and surface asynchronously (~1 frame late).
 */

import type { Device } from "./Device.js";

const kMaxTimestamps = 512;

export class Profiler {
    private querySet: GPUQuerySet | null = null;
    private resolveBuffer: GPUBuffer | null = null;
    private resultBuffer: GPUBuffer | null = null;
    private nextIndex = 0;
    private entries: { label: string; begin: number; end: number }[] = [];
    private stats = new Map<string, number>();
    private readbackInFlight = false;

    /** Label applied to GPU passes begun until the next setLabel (RenderGraph sets pass names). */
    currentLabel = "";

    constructor(private readonly device: Device) {
        if (device.hasFeature("timestamp-query")) {
            this.querySet = device.gpuDevice.createQuerySet({ type: "timestamp", count: kMaxTimestamps });
            this.resolveBuffer = device.gpuDevice.createBuffer({
                size: kMaxTimestamps * 8,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
            this.resultBuffer = device.gpuDevice.createBuffer({
                size: kMaxTimestamps * 8,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
        }
    }

    get available(): boolean {
        return this.querySet !== null;
    }

    /** Allocates a begin/end timestamp pair for the current label; the
     *  contexts attach the result to begin*Pass descriptors. */
    passTimestampWrites(): GPUComputePassTimestampWrites | undefined {
        if (!this.querySet || !this.currentLabel || this.nextIndex + 2 > kMaxTimestamps) return undefined;
        const begin = this.nextIndex;
        this.nextIndex += 2;
        this.entries.push({ label: this.currentLabel, begin, end: begin + 1 });
        return { querySet: this.querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
    }

    /** Resolves the frame's timestamps (call after graph execution; the
     *  readback lands asynchronously in getStats). */
    endFrame(encoder: GPUCommandEncoder): void {
        if (!this.querySet || this.nextIndex === 0 || this.readbackInFlight) {
            this.nextIndex = 0;
            this.entries = [];
            return;
        }
        encoder.resolveQuerySet(this.querySet, 0, this.nextIndex, this.resolveBuffer!, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer!, 0, this.resultBuffer!, 0, this.nextIndex * 8);
        const entries = this.entries;
        this.entries = [];
        this.nextIndex = 0;
        this.readbackInFlight = true;
        void this.device.gpuDevice.queue.onSubmittedWorkDone().then(async () => {
            await this.resultBuffer!.mapAsync(GPUMapMode.READ);
            const times = new BigUint64Array(this.resultBuffer!.getMappedRange().slice(0));
            this.resultBuffer!.unmap();
            const stats = new Map<string, number>();
            for (const e of entries) {
                const ms = Number(times[e.end]! - times[e.begin]!) / 1e6;
                stats.set(e.label, (stats.get(e.label) ?? 0) + ms);
            }
            this.stats = stats;
            this.readbackInFlight = false;
        });
    }

    /** Per-pass GPU milliseconds from the most recent resolved frame. */
    getStats(): Map<string, number> {
        return this.stats;
    }
}
