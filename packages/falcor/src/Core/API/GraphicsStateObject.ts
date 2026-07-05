/**
 * Graphics pipeline state object mirroring Falcor/Core/API/GraphicsStateObject.h.
 *
 * Shader modules are raw GPUShaderModules until M2's ProgramKernels supply them;
 * the desc shape already matches Falcor's (program + states + layouts + formats).
 */

import type { Device } from "./Device.js";
import { BlendState } from "./BlendState.js";
import { DepthStencilState } from "./DepthStencilState.js";
import { RasterizerState } from "./RasterizerState.js";
import { VertexLayout, Topology, toGpuTopology } from "./VAO.js";

export interface GraphicsStateObjectDesc {
    vertexModule: GPUShaderModule;
    vertexEntryPoint: string;
    fragmentModule?: GPUShaderModule;
    fragmentEntryPoint?: string;
    vertexLayout?: VertexLayout | null;
    colorFormats: (GPUTextureFormat | null)[];
    depthFormat?: GPUTextureFormat;
    sampleCount?: number;
    blendState?: BlendState;
    rasterizerState?: RasterizerState;
    depthStencilState?: DepthStencilState;
    topology?: Topology;
    layout?: GPUPipelineLayout | "auto";
}

export class GraphicsStateObject {
    readonly gpuPipeline: GPURenderPipeline;

    constructor(device: Device, public readonly desc: GraphicsStateObjectDesc) {
        const blend = desc.blendState ?? BlendState.create();
        const raster = desc.rasterizerState ?? RasterizerState.create();
        const depthStencil = desc.depthStencilState ?? DepthStencilState.create();
        const topology = desc.topology ?? Topology.TriangleList;

        const targets: (GPUColorTargetState | null)[] = desc.colorFormats.map((format, i) => {
            if (!format) return null;
            const { blend: blendPart, writeMask } = blend.getGpuTargetState(i);
            return { format, blend: blendPart, writeMask };
        });

        this.gpuPipeline = device.gpuDevice.createRenderPipeline({
            layout: desc.layout ?? "auto",
            vertex: {
                module: desc.vertexModule,
                entryPoint: desc.vertexEntryPoint,
                buffers: desc.vertexLayout?.getGpuLayouts() ?? [],
            },
            ...(desc.fragmentModule
                ? { fragment: { module: desc.fragmentModule, entryPoint: desc.fragmentEntryPoint, targets } }
                : {}),
            primitive: {
                ...raster.getGpuPrimitiveState(toGpuTopology(topology)),
                stripIndexFormat: topology === Topology.TriangleStrip || topology === Topology.LineStrip ? "uint32" : undefined,
            },
            ...(desc.depthFormat
                ? { depthStencil: { ...depthStencil.getGpuDepthStencilState(desc.depthFormat), ...raster.getGpuDepthBias() } }
                : {}),
            multisample: desc.sampleCount && desc.sampleCount > 1 ? { count: desc.sampleCount } : undefined,
        });
    }
}
