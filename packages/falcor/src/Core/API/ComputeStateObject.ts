/**
 * Compute pipeline state object mirroring Falcor/Core/API/ComputeStateObject.h.
 */

import type { Device } from "./Device.js";

export interface ComputeStateObjectDesc {
    module: GPUShaderModule;
    entryPoint: string;
    layout?: GPUPipelineLayout | "auto";
    constants?: Record<string, number>;
}

export class ComputeStateObject {
    readonly gpuPipeline: GPUComputePipeline;

    constructor(device: Device, public readonly desc: ComputeStateObjectDesc) {
        this.gpuPipeline = device.gpuDevice.createComputePipeline({
            layout: desc.layout ?? "auto",
            compute: { module: desc.module, entryPoint: desc.entryPoint, constants: desc.constants },
        });
    }
}
