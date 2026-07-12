/**
 * In-place bitonic sort of 32-bit values in chunks, mirroring
 * Utils/Algorithm/BitonicSort. The kernel is a portable shared-memory
 * variant of the native NVAPI warp-shuffle implementation (docs §9);
 * chunk results are identical.
 */

import type { Buffer } from "../../Core/API/Buffer.js";
import type { ComputeContext } from "../../Core/API/ComputeContext.js";
import type { Device } from "../../Core/API/Device.js";
import { ArgumentError } from "../../Core/Error.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";

const kShaderFile = "Utils/Algorithm/BitonicSort.cs.slang";

export class BitonicSort {
    /** Compiled variants keyed by "chunk/group" (defines force recompiles). */
    private passes = new Map<string, ComputePass>();

    constructor(private readonly device: Device) {}

    /** Mirrors BitonicSort::execute: sorts data in-place in chunks of
     *  chunkSize elements (each chunk ascending; tails pad as UINT_MAX). */
    execute(ctx: ComputeContext, data: Buffer, totalSize: number, chunkSize: number, groupSize = 1024): void {
        const isPow2 = (v: number) => v > 0 && (v & (v - 1)) === 0;
        if (!(chunkSize <= groupSize && isPow2(chunkSize))) throw new ArgumentError(`BitonicSort: invalid chunkSize ${chunkSize}`);
        if (!(groupSize <= 1024 && isPow2(groupSize))) throw new ArgumentError(`BitonicSort: invalid groupSize ${groupSize}`);
        if (totalSize === 0 || chunkSize <= 1) return;

        const key = `${chunkSize}/${groupSize}`;
        let pass = this.passes.get(key);
        if (!pass) {
            pass = ComputePass.create(this.device, { path: kShaderFile, defines: { CHUNK_SIZE: `${chunkSize}`, GROUP_SIZE: `${groupSize}` } });
            this.passes.set(key, pass);
        }

        const numGroups = Math.ceil(totalSize / groupSize);
        const groupsX = Math.max(Math.floor(Math.sqrt(numGroups)), 1);
        const groupsY = Math.ceil(numGroups / groupsX);
        const root = pass.getRootVar();
        (root["CB"] as ShaderVar)["gTotalSize"] = totalSize;
        (root["CB"] as ShaderVar)["gDispatchX"] = groupsX;
        root["gData"] = data;
        pass.execute(ctx, groupsX * groupSize, groupsY);
    }
}
