/**
 * Rasterizer state mirroring Falcor/Core/API/RasterizerState.h.
 *
 * WebGPU gaps (parity matrix §8.1):
 * - FillMode.Wireframe: no polygon fill mode; passes requesting wireframe get
 *   solid fill plus a logged warning (proper emulation via line-list rewrite is
 *   a debug-pass feature, tracked for M8).
 * - conservative rasterization, forcedSampleCount: absent from WebGPU.
 */

import { Logger } from "../../Utils/Logger.js";

export enum CullMode {
    None,
    Front,
    Back,
}

export enum FillMode {
    Wireframe,
    Solid,
}

export class RasterizerStateDesc {
    cullMode = CullMode.Back;
    fillMode = FillMode.Solid;
    frontCounterClockwise = false;
    depthBias = 0;
    slopeScaledDepthBias = 0;
    depthClampEnabled = false;
    scissorEnabled = false;

    setCullMode(mode: CullMode): this { this.cullMode = mode; return this; }
    setFillMode(mode: FillMode): this { this.fillMode = mode; return this; }
    setFrontCounterCW(ccw: boolean): this { this.frontCounterClockwise = ccw; return this; }
    setDepthBias(bias: number, slopeScaledBias: number): this {
        this.depthBias = bias;
        this.slopeScaledDepthBias = slopeScaledBias;
        return this;
    }
    setDepthClamp(enabled: boolean): this { this.depthClampEnabled = enabled; return this; }
    setScissorTest(enabled: boolean): this { this.scissorEnabled = enabled; return this; }
}

export class RasterizerState {
    constructor(public readonly desc: RasterizerStateDesc) {
        if (desc.fillMode === FillMode.Wireframe) {
            Logger.warning("RasterizerState: FillMode.Wireframe is not supported in WebGPU; using Solid (docs §8.1)");
        }
    }

    static create(desc: RasterizerStateDesc = new RasterizerStateDesc()): RasterizerState {
        return new RasterizerState(desc);
    }

    /** Lowers to the primitive-state portion of a render pipeline. */
    getGpuPrimitiveState(topology: GPUPrimitiveTopology): GPUPrimitiveState {
        return {
            topology,
            frontFace: this.desc.frontCounterClockwise ? "ccw" : "cw",
            cullMode: this.desc.cullMode === CullMode.None ? "none" : this.desc.cullMode === CullMode.Front ? "front" : "back",
            unclippedDepth: this.desc.depthClampEnabled ? true : undefined,
        };
    }

    /** Depth-bias values merge into GPUDepthStencilState at pipeline creation. */
    getGpuDepthBias(): Pick<GPUDepthStencilState, "depthBias" | "depthBiasSlopeScale" | "depthBiasClamp"> {
        return { depthBias: this.desc.depthBias, depthBiasSlopeScale: this.desc.slopeScaledDepthBias, depthBiasClamp: 0 };
    }
}
