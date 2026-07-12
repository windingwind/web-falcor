/**
 * Aggregate side of Rendering/Utils/PixelStats: sums the packed per-pixel
 * stats buffer (see the PixelStats slang override) on the GPU and exposes
 * the native Stats fields. Web divergence (docs §9): the readback lands
 * asynchronously (~1 frame late) instead of native's fence wait; collection
 * itself lives in the pass owning the buffer (PathTracer).
 */

import { Buffer } from "../../Core/API/Buffer.js";
import type { ComputeContext } from "../../Core/API/ComputeContext.js";
import type { Device } from "../../Core/API/Device.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";

/** Regions: 0 visibility rays, 1 closestHit rays, 2 path length,
 *  3 path vertex count, 4 volume lookups. */
const kRegions = 5;

/** Mirrors PixelStats::Stats. */
export interface PixelStatsAggregate {
    valid: boolean;
    visibilityRays: number;
    closestHitRays: number;
    totalRays: number;
    pathVertices: number;
    volumeLookups: number;
    avgVisibilityRays: number;
    avgClosestHitRays: number;
    avgTotalRays: number;
    avgPathLength: number;
    avgPathVertices: number;
    avgVolumeLookups: number;
}

const kEmptyStats: PixelStatsAggregate = {
    valid: false,
    visibilityRays: 0,
    closestHitRays: 0,
    totalRays: 0,
    pathVertices: 0,
    volumeLookups: 0,
    avgVisibilityRays: 0,
    avgClosestHitRays: 0,
    avgTotalRays: 0,
    avgPathLength: 0,
    avgPathVertices: 0,
    avgVolumeLookups: 0,
};

export class PixelStats {
    private sumPass: ComputePass | null = null;
    private resultBuffer: Buffer | null = null;
    private readbackInFlight = false;
    private stats: PixelStatsAggregate = kEmptyStats;

    constructor(private readonly device: Device) {}

    /** Sums the packed stats buffer per region and starts the async readback
     *  (mirrors PixelStats::endFrame + copyStatsToCPU). */
    resolve(ctx: ComputeContext, statsBuffer: Buffer, frameDim: [number, number]): void {
        const n = frameDim[0] * frameDim[1];
        if (n === 0) return;
        this.sumPass ??= ComputePass.create(this.device, { path: "WebFalcor/PixelStatsSum.cs.slang" });
        this.resultBuffer ??= new Buffer(this.device, {
            size: kRegions * 4,
            structSize: 4,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name: "PixelStats::result",
        });
        ctx.clearBuffer(this.resultBuffer);
        const root = this.sumPass.getRootVar();
        (root["SumCB"] as ShaderVar)["gNumElems"] = n;
        root["gStatsBuffer"] = statsBuffer;
        root["gResult"] = this.resultBuffer;
        this.sumPass.execute(ctx, n, kRegions);

        if (this.readbackInFlight) return;
        this.readbackInFlight = true;
        void this.resultBuffer.getBlob().then((bytes) => {
            const totals = new Uint32Array(bytes.buffer, bytes.byteOffset, kRegions);
            const [visibilityRays, closestHitRays, pathLength, pathVertices, volumeLookups] = totals as unknown as [number, number, number, number, number];
            const totalRays = visibilityRays + closestHitRays;
            this.stats = {
                valid: true,
                visibilityRays,
                closestHitRays,
                totalRays,
                pathVertices,
                volumeLookups,
                avgVisibilityRays: visibilityRays / n,
                avgClosestHitRays: closestHitRays / n,
                avgTotalRays: totalRays / n,
                avgPathLength: pathLength / n,
                avgPathVertices: pathVertices / n,
                avgVolumeLookups: volumeLookups / n,
            };
            this.readbackInFlight = false;
        });
    }

    /** Mirrors PixelStats::getStats (most recent resolved frame). */
    getStats(): PixelStatsAggregate {
        return this.stats;
    }
}
