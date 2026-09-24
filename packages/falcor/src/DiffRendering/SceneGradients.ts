/**
 * Mirrors DiffRendering/SceneGradients: per-GradientType gradient buffers that differentiable
 * kernels accumulate into (hashed over `hashSize` slots to spread atomic contention) and an
 * aggregation pass that folds the slots into `grads`. §9: the shader side names the four
 * temporaries tmpGrads0..3 (WGSL has no resource arrays; see the SceneGradients.slang override).
 */

import { Buffer } from "../Core/API/Buffer.js";
import { ResourceBindFlags, MemoryType } from "../Core/API/Types.js";
import type { Device } from "../Core/API/Device.js";
import type { RenderContext } from "../Core/API/RenderContext.js";
import { ComputePass } from "../Core/Pass/ComputePass.js";
import type { ShaderVar } from "../Core/Program/ParameterBlock.js";

/** Mirrors GradientType (DiffRendering/SharedTypes.slang). */
export enum GradientType {
    Material = 0,
    MeshPosition = 1,
    MeshNormal = 2,
    MeshTangent = 3,
    Count = 4,
}

/** Mirrors GradientAggregateMode. */
export enum GradientAggregateMode {
    Direct = 0,
    HashGrid = 1,
}

/** Mirrors SceneGradients::GradConfig. */
export interface GradConfig {
    type: GradientType;
    dim: number;
    hashSize: number;
}

const kAggregateShader = "DiffRendering/AggregateGradients.cs.slang";

export class SceneGradients {
    private readonly infos: { active: boolean; dim: number; hashSize: number }[] = [];
    private readonly grads: (Buffer | null)[] = [];
    private readonly tmpGrads: (Buffer | null)[] = [];
    private readonly aggregatePass: ComputePass;
    /** Bound for inactive types (WebGPU needs a buffer per binding). */
    private readonly placeholders: Buffer[] = [];

    constructor(
        private readonly device: Device,
        configs: GradConfig[],
        readonly aggregateMode = GradientAggregateMode.HashGrid,
    ) {
        for (let i = 0; i < GradientType.Count; i++) this.infos.push({ active: false, dim: 0, hashSize: 0 });
        for (const c of configs) this.infos[c.type] = { active: true, dim: c.dim, hashSize: c.hashSize };
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const make = (bytes: number, name: string) => new Buffer(device, { size: Math.max(bytes, 4), structSize: 4, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name });
        this.infos.forEach((info, i) => {
            this.grads.push(info.active ? make(info.dim * 4, `SceneGradients::grads${i}`) : null);
            this.tmpGrads.push(info.active ? make(info.dim * info.hashSize * 4, `SceneGradients::tmpGrads${i}`) : null);
            this.placeholders.push(make(4, `SceneGradients::placeholder${i}`));
        });
        this.aggregatePass = ComputePass.create(device, { path: kAggregateShader, csEntry: aggregateMode === GradientAggregateMode.Direct ? "mainDirect" : "mainHashGrid" });
    }

    /** Mirrors SceneGradients::create. */
    static create(device: Device, configs: GradConfig[]): SceneGradients {
        return new SceneGradients(device, configs, GradientAggregateMode.HashGrid);
    }

    getGradDim(type: GradientType): number {
        return this.infos[type]!.dim;
    }

    getHashSize(type: GradientType): number {
        return this.infos[type]!.hashSize;
    }

    /** Mirrors SceneGradients::getGradsBuffer. */
    getGradsBuffer(type: GradientType): Buffer | null {
        return this.grads[type]!;
    }

    /** Mirrors SceneGradients::bindShaderData onto a `gSceneGradients` variable. */
    bindShaderData(v: ShaderVar): void {
        v["gradDim"] = this.infos.map((i) => i.dim);
        v["hashSize"] = this.infos.map((i) => i.hashSize);
        for (let i = 0; i < GradientType.Count; i++) v[`tmpGrads${i}`] = this.tmpGrads[i] ?? this.placeholders[i]!;
    }

    /** Mirrors SceneGradients::clearGrads. */
    clearGrads(ctx: RenderContext, type: GradientType): void {
        if (!this.infos[type]!.active) return;
        ctx.clearBuffer(this.tmpGrads[type]!);
        ctx.clearBuffer(this.grads[type]!);
    }

    /** Mirrors SceneGradients::aggregateGrads. */
    aggregateGrads(ctx: RenderContext, type: GradientType): void {
        const info = this.infos[type]!;
        if (!info.active) return;
        const hashSize = this.aggregateMode === GradientAggregateMode.Direct ? 1 : info.hashSize;
        const v = this.aggregatePass.getRootVar()["gAggregator"] as ShaderVar;
        v["gradDim"] = info.dim;
        v["hashSize"] = hashSize;
        v["tmpGrads"] = this.tmpGrads[type]!;
        v["grads"] = this.grads[type]!;
        this.aggregatePass.execute(ctx, info.dim, hashSize, 1);
    }

    clearAllGrads(ctx: RenderContext): void {
        for (let i = 0; i < GradientType.Count; i++) this.clearGrads(ctx, i);
    }

    aggregateAllGrads(ctx: RenderContext): void {
        for (let i = 0; i < GradientType.Count; i++) this.aggregateGrads(ctx, i);
    }
}
