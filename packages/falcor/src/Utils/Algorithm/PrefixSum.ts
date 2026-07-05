/**
 * Parallel prefix sum mirroring Falcor/Utils/Algorithm/PrefixSum.h.
 * In-place exclusive scan over a uint32 buffer; multi-iteration for large inputs.
 */

import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { ResourceBindFlags } from "../../Core/API/Types.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import { assert } from "../../Core/Error.js";

const kShaderFile = "Utils/Algorithm/PrefixSum.cs.slang";
const kGroupSize = 1024;

export class PrefixSum {
    private groupPass: ComputePass;
    private finalizePass: ComputePass;
    private prefixGroupSums: Buffer;
    private totalSum: Buffer;
    private prevTotalSum: Buffer;

    constructor(public readonly device: Device) {
        const defines = { GROUP_SIZE: kGroupSize };
        this.groupPass = ComputePass.create(device, { path: kShaderFile, csEntry: "groupScan", defines });
        this.finalizePass = ComputePass.create(device, { path: kShaderFile, csEntry: "finalizeGroups", defines });
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        this.prefixGroupSums = new Buffer(device, { size: kGroupSize * 4, structSize: 4, bindFlags: flags, name: "PrefixSum::prefixGroupSums" });
        this.totalSum = new Buffer(device, { size: 4, structSize: 4, bindFlags: flags, name: "PrefixSum::totalSum" });
        this.prevTotalSum = new Buffer(device, { size: 4, structSize: 4, bindFlags: flags, name: "PrefixSum::prevTotalSum" });
    }

    /**
     * Mirrors PrefixSum::execute: in-place exclusive prefix sum over
     * elementCount uint32 elements. Returns the total sum if readTotalSum.
     */
    async execute(ctx: RenderContext, data: Buffer, elementCount: number, readTotalSum = false): Promise<number | undefined> {
        assert(elementCount > 0, "PrefixSum: elementCount must be > 0");
        assert(data.size >= elementCount * 4, "PrefixSum: data buffer too small");

        ctx.clearBuffer(this.totalSum);

        const maxElementCountPerIteration = kGroupSize * kGroupSize * 2;
        const iterationsCount = Math.ceil(elementCount / maxElementCountPerIteration);

        for (let iter = 0; iter < iterationsCount; iter++) {
            const numPrefixGroups = Math.max(1, Math.ceil(Math.min(elementCount, maxElementCountPerIteration) / (kGroupSize * 2)));
            assert(numPrefixGroups > 0 && numPrefixGroups <= kGroupSize, "PrefixSum: invalid group count");

            // Copy previous iteration's total sum.
            ctx.copyBuffer(this.prevTotalSum, this.totalSum);

            // Pass 1: per-group prefix sums.
            {
                ctx.clearBuffer(this.prefixGroupSums);
                const root = this.groupPass.getRootVar();
                root["CB"]["gNumGroups"] = numPrefixGroups;
                root["CB"]["gTotalNumElems"] = elementCount;
                root["CB"]["gIter"] = iter;
                root["gData"] = data;
                root["gPrefixGroupSums"] = this.prefixGroupSums;
                root["gTotalSum"] = this.totalSum;
                root["gPrevTotalSum"] = this.prevTotalSum;
                this.groupPass.execute(ctx, numPrefixGroups * kGroupSize);
            }

            // Pass 2: finalize (only needed with more than one group).
            if (numPrefixGroups > 1) {
                const dispatchSizeX = (numPrefixGroups - 1) * 2;
                const root = this.finalizePass.getRootVar();
                root["CB"]["gNumGroups"] = numPrefixGroups;
                root["CB"]["gTotalNumElems"] = elementCount;
                root["CB"]["gIter"] = iter;
                root["gData"] = data;
                root["gPrefixGroupSums"] = this.prefixGroupSums;
                root["gTotalSum"] = this.totalSum;
                root["gPrevTotalSum"] = this.prevTotalSum;
                this.finalizePass.execute(ctx, dispatchSizeX * kGroupSize);
            }
        }

        if (readTotalSum) {
            const bytes = await ctx.readBuffer(this.totalSum, 0, 4);
            return new Uint32Array(bytes.buffer, 0, 1)[0];
        }
        return undefined;
    }
}
