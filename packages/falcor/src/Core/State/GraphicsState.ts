/**
 * Mutable graphics state mirroring Falcor/Core/State/GraphicsState.h.
 * Lazily resolves GraphicsStateObjects with a hash cache (same pattern as
 * Falcor's StateGraph, simplified to a key map).
 */

import type { Device } from "../API/Device.js";
import { GraphicsStateObject } from "../API/GraphicsStateObject.js";
import { BlendState } from "../API/BlendState.js";
import { DepthStencilState } from "../API/DepthStencilState.js";
import { RasterizerState } from "../API/RasterizerState.js";
import type { Fbo } from "../API/FBO.js";
import { Vao, Topology } from "../API/VAO.js";
import { RuntimeError } from "../Error.js";
import type { EntryPointKernel } from "../Program/Program.js";

export class GraphicsState {
    private fbo: Fbo | null = null;
    private vao: Vao | null = null;
    private blendState = BlendState.create();
    private rasterizerState = RasterizerState.create();
    private depthStencilState = DepthStencilState.create();
    private vertexKernel: EntryPointKernel | null = null;
    private fragmentKernel: EntryPointKernel | null = null;
    private gsoCache = new Map<string, GraphicsStateObject>();
    private cacheEpoch = 0;

    constructor(public readonly device: Device) {}

    setFbo(fbo: Fbo): this { this.fbo = fbo; return this; }
    getFbo(): Fbo | null { return this.fbo; }
    setVao(vao: Vao | null): this { this.vao = vao; return this; }
    getVao(): Vao | null { return this.vao; }
    setBlendState(state: BlendState): this { this.blendState = state; return this; }
    setRasterizerState(state: RasterizerState): this { this.rasterizerState = state; return this; }
    setDepthStencilState(state: DepthStencilState): this { this.depthStencilState = state; return this; }
    getBlendState(): BlendState { return this.blendState; }

    /** Binds the program's raster kernels (mirrors GraphicsState::setProgram). */
    setKernels(vertex: EntryPointKernel, fragment?: EntryPointKernel): this {
        this.vertexKernel = vertex;
        this.fragmentKernel = fragment ?? null;
        this.cacheEpoch++;
        return this;
    }

    /** Mirrors GraphicsState::getGSO: lazy resolve + cache. */
    getGSO(layout?: GPUPipelineLayout): GraphicsStateObject {
        if (!this.vertexKernel) throw new RuntimeError("GraphicsState: no program kernels bound");
        if (!this.fbo) throw new RuntimeError("GraphicsState: no FBO bound");
        const key = [
            this.cacheEpoch,
            this.fbo.getGpuColorFormats().join(","),
            this.fbo.getGpuDepthFormat() ?? "-",
            this.fbo.sampleCount,
            this.vao?.topology ?? Topology.TriangleList,
            JSON.stringify(this.blendState.desc),
            JSON.stringify(this.rasterizerState.desc),
            JSON.stringify(this.depthStencilState.desc),
            layout ? "explicit" : "auto",
        ].join("|");
        let gso = this.gsoCache.get(key);
        if (!gso) {
            gso = new GraphicsStateObject(this.device, {
                vertexModule: this.vertexKernel.module,
                vertexEntryPoint: this.vertexKernel.name,
                fragmentModule: this.fragmentKernel?.module,
                fragmentEntryPoint: this.fragmentKernel?.name,
                vertexLayout: this.vao?.vertexLayout ?? null,
                colorFormats: this.fbo.getGpuColorFormats(),
                depthFormat: this.fbo.getGpuDepthFormat(),
                sampleCount: this.fbo.sampleCount,
                blendState: this.blendState,
                rasterizerState: this.rasterizerState,
                depthStencilState: this.depthStencilState,
                topology: this.vao?.topology ?? Topology.TriangleList,
                layout,
            });
            this.gsoCache.set(key, gso);
        }
        return gso;
    }
}
