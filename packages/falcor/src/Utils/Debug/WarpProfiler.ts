/**
 * Mirrors Falcor/Utils/Debug/WarpProfiler: per-bin histograms of warp
 * utilization and divergence, recorded by Utils/Debug/WarpProfiler.slang.
 * Needs the WebGPU 'subgroups' feature (32-wide subgroups, as native assumes).
 * §9: readback is async, and saveWarpHistogramsAsCSV returns the CSV text.
 */

import type { Device } from "../../Core/API/Device.js";
import type { Buffer } from "../../Core/API/Buffer.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { ResourceBindFlags } from "../../Core/API/Types.js";
import { RuntimeError } from "../../Core/Error.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";

export class WarpProfiler {
    static readonly kWarpSize = 32;
    private readonly histogramBuffer: Buffer;
    private histograms: Uint32Array | null = null;
    private active = false;
    private dataWaiting = false;

    constructor(private readonly device: Device, private readonly binCount: number) {
        this.histogramBuffer = device.createStructuredBuffer(4, binCount * WarpProfiler.kWarpSize, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
    }

    bindShaderData(var_: ShaderVar): void {
        var_["gWarpHistogram"] = this.histogramBuffer;
    }

    begin(ctx: RenderContext): void {
        if (this.active) throw new RuntimeError("WarpProfiler: begin() already called.");
        ctx.clearBuffer(this.histogramBuffer);
        this.active = true;
        this.dataWaiting = false;
    }

    end(_ctx: RenderContext): void {
        if (!this.active) throw new RuntimeError("WarpProfiler: end() called without preceding begin().");
        this.active = false;
        this.dataWaiting = true;
    }

    private async readBackData(): Promise<void> {
        if (!this.dataWaiting) return;
        if (this.active) throw new RuntimeError("WarpProfiler: readBackData() called without preceding before()/end() calls.");
        const bytes = await this.histogramBuffer.getBlob();
        this.histograms = new Uint32Array(bytes.slice().buffer);
        this.dataWaiting = false;
    }

    /** Mirrors getWarpHistogram: the summed histogram of bins [binIndex, binIndex + binCount). */
    async getWarpHistogram(binIndex: number, binCount = 1): Promise<number[]> {
        await this.readBackData();
        if (binIndex + binCount > this.binCount) throw new RuntimeError("WarpProfiler: Bin index out of range.");
        if (!this.histograms) throw new RuntimeError("WarpProfiler: No available data. Did you call begin()/end()?");
        const histogram = new Array<number>(WarpProfiler.kWarpSize).fill(0);
        for (let i = binIndex; i < binIndex + binCount; i++) for (let j = 0; j < WarpProfiler.kWarpSize; j++) histogram[j]! += this.histograms[i * WarpProfiler.kWarpSize + j]!;
        return histogram;
    }

    /** Mirrors saveWarpHistogramsAsCSV: one line per bin, ';'-separated counts. */
    async saveWarpHistogramsAsCSV(): Promise<string> {
        await this.readBackData();
        if (!this.histograms) return "";
        const lines: string[] = [];
        for (let i = 0; i < this.binCount; i++) lines.push(Array.from(this.histograms.subarray(i * WarpProfiler.kWarpSize, (i + 1) * WarpProfiler.kWarpSize)).join(";"));
        return lines.join("\n") + "\n";
    }
}
