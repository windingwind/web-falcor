/**
 * Raster pass mirroring Falcor/Core/Pass/RasterPass.h: program (VS+PS),
 * GraphicsState, ParameterBlock.
 */

import type { Device } from "../API/Device.js";
import type { RenderContext } from "../API/RenderContext.js";
import type { Fbo } from "../API/FBO.js";
import type { Vao } from "../API/VAO.js";
import { DefineList } from "../Program/DefineList.js";
import { ShaderType } from "../Program/SlangCompiler.js";
import { ParameterBlock, makeRootVar, type ShaderVar } from "../Program/ParameterBlock.js";
import { mergeWgslBindings } from "../Program/ProgramReflection.js";
import type { Program, ProgramVersion } from "../Program/Program.js";
import { GraphicsState } from "../State/GraphicsState.js";

export interface RasterPassDesc {
    /** Module path(s); vs/ps entries reference them via vsModuleIndex/psModuleIndex. */
    path: string | string[];
    vsEntry?: string;
    psEntry?: string;
    vsModuleIndex?: number;
    psModuleIndex?: number;
    defines?: DefineList | Record<string, string | number | boolean>;
}

export class RasterPass {
    readonly program: Program;
    readonly state: GraphicsState;
    protected version: ProgramVersion;
    protected vars: ParameterBlock;
    protected root: ShaderVar;
    protected pipelineLayout: GPUPipelineLayout;

    static create(device: Device, desc: RasterPassDesc): RasterPass {
        return new RasterPass(device, desc);
    }

    protected constructor(
        public readonly device: Device,
        desc: RasterPassDesc,
    ) {
        const defines = desc.defines instanceof DefineList ? desc.defines : new DefineList().addAll(desc.defines ?? {});
        const vsEntry = desc.vsEntry ?? "vsMain";
        const psEntry = desc.psEntry ?? "psMain";
        this.program = device.programManager.createProgram(
            {
                path: desc.path,
                entryPoints: [
                    { name: vsEntry, type: ShaderType.Vertex, moduleIndex: desc.vsModuleIndex ?? 0 },
                    { name: psEntry, type: ShaderType.Pixel, moduleIndex: desc.psModuleIndex ?? 0 },
                ],
            },
            defines,
        );
        this.version = this.program.getActiveVersion();
        const vs = this.version.getKernel(vsEntry, ShaderType.Vertex);
        const ps = this.version.getKernel(psEntry, ShaderType.Pixel);
        this.vars = new ParameterBlock(device, this.version.reflection, mergeWgslBindings(vs.bindings, ps.bindings));
        this.root = makeRootVar(this.vars);
        this.state = new GraphicsState(device).setKernels(vs, ps);

        const groupIndices = this.vars.getGroupIndices();
        const maxGroup = groupIndices.length ? Math.max(...groupIndices) : -1;
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= maxGroup; g++) {
            layouts.push(this.vars.getBindGroupLayout(g) ?? device.gpuDevice.createBindGroupLayout({ entries: [] }));
        }
        this.pipelineLayout = device.gpuDevice.createPipelineLayout({ bindGroupLayouts: layouts });
    }

    getRootVar(): ShaderVar {
        return this.root;
    }

    getParameterBlock(): ParameterBlock {
        return this.vars;
    }

    /** Mirrors RasterPass::draw: non-indexed draw into the state's FBO. */
    draw(ctx: RenderContext, fbo: Fbo, vertexCount: number, instanceCount = 1): void {
        this.state.setFbo(fbo);
        this.drawCommon(ctx, fbo, (pass) => pass.draw(vertexCount, instanceCount));
    }

    /** Indexed variant (mirrors RasterPass::drawIndexed). */
    drawIndexed(ctx: RenderContext, fbo: Fbo, indexCount: number, instanceCount = 1): void {
        this.state.setFbo(fbo);
        this.drawCommon(ctx, fbo, (pass, vao) => {
            if (vao?.indexBuffer) pass.setIndexBuffer(vao.indexBuffer.gpuBuffer, vao.getGpuIndexFormat());
            pass.drawIndexed(indexCount, instanceCount);
        });
    }

    private drawCommon(ctx: RenderContext, fbo: Fbo, emit: (pass: GPURenderPassEncoder, vao: Vao | null) => void): void {
        const gso = this.state.getGSO(this.pipelineLayout);
        const vao = this.state.getVao();
        // Resolve bind groups (and flush dirty cbuffer uploads onto the encoder)
        // BEFORE opening the render pass — the encoder is locked while a pass is open.
        const bindGroups = this.vars.getGroupIndices().map((g) => ({ index: g, group: this.vars.getBindGroup(g) }));
        const pass = ctx.getEncoder().beginRenderPass(fbo.getGpuRenderPassDescriptor());
        pass.setPipeline(gso.gpuPipeline);
        pass.setViewport(0, 0, fbo.width, fbo.height, 0, 1);
        const blendFactor = this.state.getBlendState().desc.blendFactor;
        if (blendFactor.some((v) => v !== 0)) {
            pass.setBlendConstant({ r: blendFactor[0], g: blendFactor[1], b: blendFactor[2], a: blendFactor[3] });
        }
        for (const { index, group } of bindGroups) pass.setBindGroup(index, group);
        vao?.vertexBuffers.forEach((vb, i) => pass.setVertexBuffer(i, vb.gpuBuffer));
        emit(pass, vao);
        pass.end();
    }
}
