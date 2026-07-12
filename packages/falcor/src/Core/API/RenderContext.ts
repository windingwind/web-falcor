/**
 * Render command context mirroring Falcor/Core/API/RenderContext.h.
 *
 * Draw entry points taking GraphicsState/ProgramVars arrive with M2; blit and
 * clears are available now. blit() uses an internal cached pipeline exactly like
 * Falcor's BlitContext (fullscreen triangle + sampled copy).
 */

import { ComputeContext } from "./ComputeContext.js";
import type { Texture } from "./Texture.js";
import type { Fbo } from "./FBO.js";
import type { Vao } from "./VAO.js";
import type { GraphicsStateObject } from "./GraphicsStateObject.js";
import { isDepthFormat } from "./Formats.js";
import { RuntimeError } from "../Error.js";

const kBlitWgsl = /* wgsl */ `
@group(0) @binding(0) var gSrc: texture_2d<f32>;
@group(0) @binding(1) var gSampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vsMain(@builtin(vertex_index) vid: u32) -> VSOut {
    // Fullscreen triangle (same trick as Falcor's FullScreenPass).
    var out: VSOut;
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    out.pos = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    return out;
}

@fragment fn psMain(in: VSOut) -> @location(0) vec4f {
    return textureSampleLevel(gSrc, gSampler, in.uv, 0.0);
}
`;

export class RenderContext extends ComputeContext {
    /** Attaches profiler timestamps to a render-pass descriptor when active. */
    private withTimestamps(desc: GPURenderPassDescriptor): GPURenderPassDescriptor {
        const tw = this.device.profilerHook?.passTimestampWrites();
        if (tw) desc.timestampWrites = tw;
        return desc;
    }

    private blitPipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
    private blitSamplers = new Map<GPUFilterMode, GPUSampler>();

    /** Mirrors RenderContext::clearRtv. */
    clearRtv(view: GPUTextureView, color: [number, number, number, number]): void {
        const pass = this.getEncoder().beginRenderPass(this.withTimestamps({
            colorAttachments: [
                { view, clearValue: { r: color[0], g: color[1], b: color[2], a: color[3] }, loadOp: "clear", storeOp: "store" },
            ],
        }));
        pass.end();
    }

    /** Mirrors RenderContext::clearDsv. */
    clearDsv(view: GPUTextureView, depth: number, stencil: number, clearDepth = true, clearStencil = false): void {
        const pass = this.getEncoder().beginRenderPass(this.withTimestamps({
            colorAttachments: [],
            depthStencilAttachment: {
                view,
                depthClearValue: depth,
                depthLoadOp: clearDepth ? "clear" : "load",
                depthStoreOp: "store",
                ...(clearStencil ? { stencilClearValue: stencil, stencilLoadOp: "clear" as const, stencilStoreOp: "store" as const } : {}),
            },
        }));
        pass.end();
    }

    /** Mirrors RenderContext::clearTexture (color textures). */
    clearTexture(texture: Texture, color: [number, number, number, number] = [0, 0, 0, 0]): void {
        if (isDepthFormat(texture.format)) {
            this.clearDsv(texture.getDSV(), color[0], 0);
            return;
        }
        for (let mip = 0; mip < texture.mipCount; mip++) this.clearRtv(texture.getRTV(mip), color);
    }

    /**
     * Mirrors RenderContext::blit: draws src into dst with optional filtering.
     * Handles format conversion via the render pipeline; complex reductions
     * (BlitContext's parity/min/max modes) come with M2 programs.
     */
    blit(src: Texture, dst: Texture, filter: GPUFilterMode = "linear", srcMip = 0, dstMip = 0): void {
        if (isDepthFormat(dst.format)) throw new RuntimeError("blit to depth target not supported (use copy)");
        let pipeline = this.blitPipelines.get(dst.gpuFormat);
        if (!pipeline) {
            const module = this.device.gpuDevice.createShaderModule({ code: kBlitWgsl });
            pipeline = this.device.gpuDevice.createRenderPipeline({
                layout: "auto",
                vertex: { module, entryPoint: "vsMain" },
                fragment: { module, entryPoint: "psMain", targets: [{ format: dst.gpuFormat }] },
                primitive: { topology: "triangle-list" },
            });
            this.blitPipelines.set(dst.gpuFormat, pipeline);
        }
        let sampler = this.blitSamplers.get(filter);
        if (!sampler) {
            sampler = this.device.gpuDevice.createSampler({ magFilter: filter, minFilter: filter });
            this.blitSamplers.set(filter, sampler);
        }
        const bindGroup = this.device.gpuDevice.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: src.getSRV(srcMip, 1) },
                { binding: 1, resource: sampler },
            ],
        });
        const pass = this.getEncoder().beginRenderPass(this.withTimestamps({
            colorAttachments: [{ view: dst.getRTV(dstMip), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" }],
        }));
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(3);
        pass.end();
    }

    /**
     * Raw draw path used until M2's GraphicsState/ProgramVars land: begins a render
     * pass from the FBO, binds the PSO, VAO buffers and bind groups, draws.
     */
    drawRaw(
        gso: GraphicsStateObject,
        vao: Vao | null,
        fbo: Fbo,
        bindGroups: (GPUBindGroup | null)[],
        vertexOrIndexCount: number,
        instanceCount = 1,
        opts: { indexed?: boolean; blendConstant?: [number, number, number, number]; stencilRef?: number } = {},
    ): void {
        const pass = this.getEncoder().beginRenderPass(this.withTimestamps(fbo.getGpuRenderPassDescriptor()));
        pass.setPipeline(gso.gpuPipeline);
        pass.setViewport(0, 0, fbo.width, fbo.height, 0, 1);
        bindGroups.forEach((bg, i) => bg && pass.setBindGroup(i, bg));
        if (opts.blendConstant) pass.setBlendConstant({ r: opts.blendConstant[0], g: opts.blendConstant[1], b: opts.blendConstant[2], a: opts.blendConstant[3] });
        if (opts.stencilRef !== undefined) pass.setStencilReference(opts.stencilRef);
        vao?.vertexBuffers.forEach((vb, i) => pass.setVertexBuffer(i, vb.gpuBuffer));
        if (opts.indexed && vao?.indexBuffer) {
            pass.setIndexBuffer(vao.indexBuffer.gpuBuffer, vao.getGpuIndexFormat());
            pass.drawIndexed(vertexOrIndexCount, instanceCount);
        } else {
            pass.draw(vertexOrIndexCount, instanceCount);
        }
        pass.end();
    }

    /** Mirrors RenderContext::resolveResource (MSAA resolve). */
    resolveResource(src: Texture, dst: Texture): void {
        if (src.sampleCount <= 1) throw new RuntimeError("resolveResource: source is not multisampled");
        const pass = this.getEncoder().beginRenderPass(this.withTimestamps({
            colorAttachments: [
                { view: src.getRTV(), resolveTarget: dst.getRTV(), loadOp: "load", storeOp: "discard" },
            ],
        }));
        pass.end();
    }
}
