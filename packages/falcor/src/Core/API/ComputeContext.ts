/**
 * Compute command context mirroring Falcor/Core/API/ComputeContext.h.
 * Dispatch entry points taking ComputeState/ProgramVars arrive with M2's
 * program system; raw pipeline dispatch is available now.
 */

import { CopyContext } from "./CopyContext.js";
import type { Buffer } from "./Buffer.js";

export class ComputeContext extends CopyContext {
    /** Raw dispatch used internally and by tests until the Program system lands (M2). */
    dispatchRaw(
        pipeline: GPUComputePipeline,
        bindGroups: (GPUBindGroup | null)[],
        groupsX: number,
        groupsY = 1,
        groupsZ = 1,
    ): void {
        const pass = this.getEncoder().beginComputePass();
        pass.setPipeline(pipeline);
        bindGroups.forEach((bg, i) => bg && pass.setBindGroup(i, bg));
        pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
        pass.end();
    }

    /** Mirrors ComputeContext::dispatchIndirect at the raw level. */
    dispatchRawIndirect(pipeline: GPUComputePipeline, bindGroups: (GPUBindGroup | null)[], argBuffer: Buffer, argOffset: number): void {
        const pass = this.getEncoder().beginComputePass();
        pass.setPipeline(pipeline);
        bindGroups.forEach((bg, i) => bg && pass.setBindGroup(i, bg));
        pass.dispatchWorkgroupsIndirect(argBuffer.gpuBuffer, argOffset);
        pass.end();
    }

    /** Mirrors ComputeContext::clearUAV for buffers (fill with zero; WebGPU clearBuffer). */
    clearBuffer(buffer: Buffer, offset = 0, size?: number): void {
        this.getEncoder().clearBuffer(buffer.gpuBuffer, offset, size);
    }
}
