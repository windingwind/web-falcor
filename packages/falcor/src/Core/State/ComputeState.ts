/**
 * Mutable compute state mirroring Falcor/Core/State/ComputeState.h.
 */

import type { Device } from "../API/Device.js";
import { ComputeStateObject } from "../API/ComputeStateObject.js";
import { RuntimeError } from "../Error.js";
import type { EntryPointKernel } from "../Program/Program.js";

export class ComputeState {
    private kernel: EntryPointKernel | null = null;
    private csoCache = new Map<EntryPointKernel, ComputeStateObject>();

    constructor(public readonly device: Device) {}

    setKernel(kernel: EntryPointKernel): this {
        this.kernel = kernel;
        return this;
    }

    getCSO(layout?: GPUPipelineLayout): ComputeStateObject {
        if (!this.kernel) throw new RuntimeError("ComputeState: no kernel bound");
        let cso = this.csoCache.get(this.kernel);
        if (!cso) {
            cso = new ComputeStateObject(this.device, { module: this.kernel.module, entryPoint: this.kernel.name, layout });
            this.csoCache.set(this.kernel, cso);
        }
        return cso;
    }
}
