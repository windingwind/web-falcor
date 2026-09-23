/**
 * Compute pass mirroring Falcor/Core/Pass/ComputePass.h.
 */

import type { Device } from "../API/Device.js";
import type { ComputeContext } from "../API/ComputeContext.js";
import { DefineList } from "../Program/DefineList.js";
import { ShaderType } from "../Program/SlangCompiler.js";
import { ParameterBlock, makeRootVar, type ShaderVar } from "../Program/ParameterBlock.js";
import type { ProgramReflection } from "../Program/ProgramReflection.js";
import type { Program, ProgramVersion, EntryPointKernel, ShaderModuleDesc, TypeConformance } from "../Program/Program.js";

export interface ComputePassDesc {
    path?: string;
    /** ProgramDesc shader modules (files and strings), after `path`. */
    modules?: ShaderModuleDesc[];
    /** ProgramDesc::addTypeConformances. */
    typeConformances?: TypeConformance[];
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
    private readonly entry: string;

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
        this.program = device.programManager.createProgram({ path: desc.path, modules: desc.modules, typeConformances: desc.typeConformances, entryPoints: [{ name: entry, type: ShaderType.Compute }] }, defines);
        this.entry = entry;
        ({ version: this.version, kernel: this.kernel, vars: this.vars, root: this.root, pipeline: this.pipeline } = this.build());
    }

    /** Builds the kernel, bindings and pipeline from the program's live version. */
    private build() {
        const version = this.program.getActiveVersion();
        const kernel = version.getKernel(this.entry);
        const vars = new ParameterBlock(this.device, version.reflection, kernel.bindings);
        const groupIndices = vars.getGroupIndices();
        const maxGroup = groupIndices.length ? Math.max(...groupIndices) : -1;
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= maxGroup; g++) {
            layouts.push(vars.getBindGroupLayout(g) ?? this.device.gpuDevice.createBindGroupLayout({ entries: [] }));
        }
        const pipeline = this.device.gpuDevice.createComputePipeline({
            layout: this.device.gpuDevice.createPipelineLayout({ bindGroupLayouts: layouts }),
            compute: { module: kernel.module, entryPoint: kernel.name },
        });
        return { version, kernel, vars, root: makeRootVar(vars), pipeline };
    }

    /** Rebuilds after a shader reload; bindings are set per frame, so callers
     *  that fetch the root var before executing pick the new kernel up. */
    private refresh(): void {
        if (this.program.getActiveVersion() === this.version) return;
        ({ version: this.version, kernel: this.kernel, vars: this.vars, root: this.root, pipeline: this.pipeline } = this.build());
    }

    /** Storage textures take their bound textures' formats (native UAV semantics): patch the WGSL declarations and rebuild the pipeline. */
    private retargetStorageFormats(): void {
        const changed = this.vars.retargetStorageFormats();
        if (!changed) return;
        let wgsl = this.kernel.wgsl;
        for (const [name, format] of changed) {
            wgsl = wgsl.replace(new RegExp(`(var\\s+${name}\\s*:\\s*texture_storage_\\w+<\\s*)\\w+`), `$1${format}`);
        }
        const gpu = this.device.gpuDevice;
        const module = gpu.createShaderModule({ code: wgsl, label: `${this.kernel.name} (storage formats retargeted)` });
        this.kernel = { ...this.kernel, wgsl, module };
        const groupIndices = this.vars.getGroupIndices();
        const maxGroup = groupIndices.length ? Math.max(...groupIndices) : -1;
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= maxGroup; g++) layouts.push(this.vars.getBindGroupLayout(g) ?? gpu.createBindGroupLayout({ entries: [] }));
        this.pipeline = gpu.createComputePipeline({ layout: gpu.createPipelineLayout({ bindGroupLayouts: layouts }), compute: { module, entryPoint: this.kernel.name } });
    }

    /** Mirrors ComputePass::getRootVar. */
    getRootVar(): ShaderVar {
        this.refresh();
        return this.root;
    }

    getParameterBlock(): ParameterBlock {
        this.refresh();
        return this.vars;
    }

    /** Mirrors getProgram()->getReflector(). */
    getReflector(): ProgramReflection {
        this.refresh();
        return this.version.reflection;
    }

    getThreadGroupSize(): [number, number, number] {
        return this.version.reflection.getEntryPoint(this.kernel.name).threadGroupSize;
    }

    /** Mirrors ComputePass::execute(ctx, nThreads): total threads, ceil-divided by group size. */
    execute(ctx: ComputeContext, threadsX: number, threadsY = 1, threadsZ = 1): void {
        this.refresh();
        this.retargetStorageFormats();
        const [gx, gy, gz] = this.getThreadGroupSize();
        const groups: [number, number, number] = [Math.ceil(threadsX / gx), Math.ceil(threadsY / gy), Math.ceil(threadsZ / gz)];
        const bindGroups = this.vars.getGroupIndices().map((g) => ({ index: g, group: this.vars.getBindGroup(g) }));
        const tw = ctx.device.profilerHook?.passTimestampWrites();
        const pass = ctx.getEncoder().beginComputePass(tw ? { timestampWrites: tw } : undefined);
        pass.setPipeline(this.pipeline);
        for (const { index, group } of bindGroups) pass.setBindGroup(index, group);
        pass.dispatchWorkgroups(...groups);
        pass.end();
    }
}
