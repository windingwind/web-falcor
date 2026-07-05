/**
 * Render command context mirroring Falcor/Core/API/RenderContext.h.
 *
 * Draw entry points taking GraphicsState/ProgramVars arrive with M2; blit and
 * clears are available now. blit() uses an internal cached pipeline exactly like
 * Falcor's BlitContext (fullscreen triangle + sampled copy).
 */

import { ComputeContext } from "./ComputeContext.js";
import type { Texture } from "./Texture.js";
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
    private blitPipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
    private blitSamplers = new Map<GPUFilterMode, GPUSampler>();

    /** Mirrors RenderContext::clearRtv. */
    clearRtv(view: GPUTextureView, color: [number, number, number, number]): void {
        const pass = this.getEncoder().beginRenderPass({
            colorAttachments: [
                { view, clearValue: { r: color[0], g: color[1], b: color[2], a: color[3] }, loadOp: "clear", storeOp: "store" },
            ],
        });
        pass.end();
    }

    /** Mirrors RenderContext::clearDsv. */
    clearDsv(view: GPUTextureView, depth: number, stencil: number, clearDepth = true, clearStencil = false): void {
        const pass = this.getEncoder().beginRenderPass({
            colorAttachments: [],
            depthStencilAttachment: {
                view,
                depthClearValue: depth,
                depthLoadOp: clearDepth ? "clear" : "load",
                depthStoreOp: "store",
                ...(clearStencil ? { stencilClearValue: stencil, stencilLoadOp: "clear" as const, stencilStoreOp: "store" as const } : {}),
            },
        });
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
    blit(src: Texture, dst: Texture, filter: GPUFilterMode = "linear"): void {
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
                { binding: 0, resource: src.getSRV(0, 1) },
                { binding: 1, resource: sampler },
            ],
        });
        const pass = this.getEncoder().beginRenderPass({
            colorAttachments: [{ view: dst.getRTV(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" }],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(3);
        pass.end();
    }

    /** Mirrors RenderContext::resolveResource (MSAA resolve). */
    resolveResource(src: Texture, dst: Texture): void {
        if (src.sampleCount <= 1) throw new RuntimeError("resolveResource: source is not multisampled");
        const pass = this.getEncoder().beginRenderPass({
            colorAttachments: [
                { view: src.getRTV(), resolveTarget: dst.getRTV(), loadOp: "load", storeOp: "discard" },
            ],
        });
        pass.end();
    }
}
