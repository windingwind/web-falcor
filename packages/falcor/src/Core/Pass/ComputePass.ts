/**
 * Compute pass mirroring Falcor/Core/Pass/ComputePass.h.
 */

import type { Device } from "../API/Device.js";
import type { ComputeContext } from "../API/ComputeContext.js";
import { DefineList } from "../Program/DefineList.js";
import { ShaderType } from "../Program/SlangCompiler.js";
import { ParameterBlock, makeRootVar, type ShaderVar } from "../Program/ParameterBlock.js";
import type { Program, ProgramVersion, EntryPointKernel } from "../Program/Program.js";

export interface ComputePassDesc {
    path: string;
    csEntry?: string;
    defines?: DefineList | Record<string, string | number | boolean>;
}

export class ComputePass {
    readonly program: Program;
    private version: ProgramVersion;
    private kernel: EntryPointKernel;
    private pipeline: GPUComputePipeline;
    private vars: ParameterBlock;
    private root: ShaderVar;

    /** Mirrors ComputePass::create. */
    static create(device: Device, desc: ComputePassDesc): ComputePass {
        return new ComputePass(device, desc);
    }

    private constructor(
        public readonly device: Device,
        desc: ComputePassDesc,
    ) {
        const defines = desc.defines instanceof DefineList ? desc.defines : new DefineList().addAll(desc.defines ?? {});
        const entry = desc.csEntry ?? "main";
        this.program = device.programManager.createProgram({ path: desc.path, entryPoints: [{ name: entry, type: ShaderType.Compute }] }, defines);
        this.version = this.program.getActiveVersion();
        this.kernel = this.version.getKernel(entry);
        this.vars = new ParameterBlock(device, this.version.reflection, this.kernel.bindings);
        this.root = makeRootVar(this.vars);

        const groupIndices = this.vars.getGroupIndices();
        const maxGroup = groupIndices.length ? Math.max(...groupIndices) : -1;
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= maxGroup; g++) {
            layouts.push(this.vars.getBindGroupLayout(g) ?? device.gpuDevice.createBindGroupLayout({ entries: [] }));
        }
        this.pipeline = device.gpuDevice.createComputePipeline({
            layout: device.gpuDevice.createPipelineLayout({ bindGroupLayouts: layouts }),
            compute: { module: this.kernel.module, entryPoint: this.kernel.name },
        });
    }

    /** Mirrors ComputePass::getRootVar. */
    getRootVar(): ShaderVar {
        return this.root;
    }

    getParameterBlock(): ParameterBlock {
        return this.vars;
    }

    getThreadGroupSize(): [number, number, number] {
        return this.version.reflection.getEntryPoint(this.kernel.name).threadGroupSize;
    }

    /** Mirrors ComputePass::execute(ctx, nThreads): total threads, ceil-divided by group size. */
    execute(ctx: ComputeContext, threadsX: number, threadsY = 1, threadsZ = 1): void {
        const [gx, gy, gz] = this.getThreadGroupSize();
        const groups: [number, number, number] = [Math.ceil(threadsX / gx), Math.ceil(threadsY / gy), Math.ceil(threadsZ / gz)];
        const bindGroups = this.vars.getGroupIndices().map((g) => ({ index: g, group: this.vars.getBindGroup(g) }));
        const pass = ctx.getEncoder().beginComputePass();
        pass.setPipeline(this.pipeline);
        for (const { index, group } of bindGroups) pass.setBindGroup(index, group);
        pass.dispatchWorkgroups(...groups);
        pass.end();
    }
}
