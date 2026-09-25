/**
 * Render command context mirroring Falcor/Core/API/RenderContext.h.
 *
 * Draw entry points taking GraphicsState/ProgramVars arrive with M2; blit and
 * clears are available now. blit() uses an internal cached pipeline exactly like
 * Falcor's BlitContext (fullscreen triangle + sampled copy).
 */

import { ComputeContext } from "./ComputeContext.js";
import type { Texture } from "./Texture.js";
import { FboAttachmentType, type Fbo } from "./FBO.js";
import type { Vao } from "./VAO.js";
import type { GraphicsStateObject } from "./GraphicsStateObject.js";
import { FormatType, ResourceFormat, getFormatChannelCount, getFormatType, getNumChannelBits, isDepthFormat } from "./Formats.js";
import { ResourceBindFlags } from "./Types.js";
import { float32ToFloat16 } from "../../Utils/Math/Float16.js";
import { Logger } from "../../Utils/Logger.js";
import { TextureReductionMode } from "./Sampler.js";
import { RuntimeError } from "../Error.js";

type BlitKind = "f32" | "u32" | "i32";

const kStandardReductions = [TextureReductionMode.Standard, TextureReductionMode.Standard, TextureReductionMode.Standard, TextureReductionMode.Standard] as const;

/**
 * Blit shader (mirrors Core/API/BlitReduction.3d.slang). Float sources sample with the
 * blit filter; min/max channels reduce over the texels the filter footprint touches
 * (what a reduction sampler returns). Integer sources can't be filtered in WebGPU:
 * they load texels, and the linear filter averages the 2x2 footprint exactly (floor).
 */
function blitWgsl(src: BlitKind, dst: BlitKind, reductions: readonly TextureReductionMode[], linear: boolean, sampleCount = 1): string {
    const vec = (k: BlitKind) => `vec4<${k}>`;
    const complex = reductions.some((r) => r !== TextureReductionMode.Standard);
    let body: string;
    if (sampleCount > 1) {
        // Native's SAMPLE_COUNT > 1 path: the average of the texel's samples, point-sampled.
        body = `
    let dims = textureDimensions(gSrc);
    let crd = vec2<u32>(vec2<f32>(dims) * in.uv);
    var res = vec4<f32>(0.0);
    for (var i = 0u; i < ${sampleCount}u; i++) { res += textureLoad(gSrc, crd, i); }
    res /= ${sampleCount}.0;`;
    } else if (src === "f32" && !complex) {
        body = `let res = textureSampleLevel(gSrc, gSampler, in.uv, 0.0);`;
    } else {
        body = `
    let dims = vec2<i32>(textureDimensions(gSrc, 0));
    let p = in.uv * vec2<f32>(dims) - 0.5;
    ${linear ? "let i0 = vec2<i32>(floor(p));" : "let i0 = vec2<i32>(floor(p + 0.5));"}
    let c00 = textureLoad(gSrc, clamp(i0, vec2<i32>(0), dims - 1), 0);
    let c10 = textureLoad(gSrc, clamp(i0 + vec2<i32>(${linear ? 1 : 0}, 0), vec2<i32>(0), dims - 1), 0);
    let c01 = textureLoad(gSrc, clamp(i0 + vec2<i32>(0, ${linear ? 1 : 0}), vec2<i32>(0), dims - 1), 0);
    let c11 = textureLoad(gSrc, clamp(i0 + vec2<i32>(${linear ? 1 : 0}), vec2<i32>(0), dims - 1), 0);`;
        if (src === "f32") {
            body += `
    let avg = textureSampleLevel(gSrc, gSampler, in.uv, 0.0);
    let mn = min(min(c00, c10), min(c01, c11));
    let mx = max(max(c00, c10), max(c01, c11));
    let res = vec4<f32>(${reductions.map((r, i) => `${r === TextureReductionMode.Min ? "mn" : r === TextureReductionMode.Max ? "mx" : "avg"}[${i}]`).join(", ")});`;
        } else if (linear) {
            // floor((a + b + c + d) / 4) without overflow.
            const avg = src === "u32" ? "(c00 >> vec4<u32>(2u)) + (c10 >> vec4<u32>(2u)) + (c01 >> vec4<u32>(2u)) + (c11 >> vec4<u32>(2u)) + (((c00 & vec4<u32>(3u)) + (c10 & vec4<u32>(3u)) + (c01 & vec4<u32>(3u)) + (c11 & vec4<u32>(3u))) >> vec4<u32>(2u))"
                : "(c00 >> vec4<u32>(2u)) + (c10 >> vec4<u32>(2u)) + (c01 >> vec4<u32>(2u)) + (c11 >> vec4<u32>(2u)) + (((c00 & vec4<i32>(3)) + (c10 & vec4<i32>(3)) + (c01 & vec4<i32>(3)) + (c11 & vec4<i32>(3))) >> vec4<u32>(2u))";
            body += `
    let res = ${avg};`;
        } else {
            body += `
    let res = c00;`;
        }
    }
    const convert = src === dst ? "res" : `${vec(dst)}(res)`;
    return /* wgsl */ `
@group(0) @binding(0) var gSrc: ${sampleCount > 1 ? "texture_multisampled_2d" : "texture_2d"}<${src}>;
${src === "f32" && sampleCount === 1 ? "@group(0) @binding(1) var gSampler: sampler;" : ""}

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

@fragment fn psMain(in: VSOut) -> @location(0) ${vec(dst)} {
    ${body}
    return ${convert};
}
`;
}

function blitKind(format: ResourceFormat): BlitKind {
    const t = getFormatType(format);
    return t === FormatType.Uint ? "u32" : t === FormatType.Sint ? "i32" : "f32";
}

/** One texel of `color` in `format` (float32/float16/unorm8 channels), or null if unsupported. */
function encodeClearTexel(format: ResourceFormat, color: [number, number, number, number]): Uint8Array | null {
    const channels = getFormatChannelCount(format);
    const bits = getNumChannelBits(format, 0);
    const type = getFormatType(format);
    const out = new Uint8Array((channels * bits) / 8);
    const view = new DataView(out.buffer);
    for (let c = 0; c < channels; c++) {
        const v = color[c] ?? 0;
        if (type === FormatType.Float && bits === 32) view.setFloat32(c * 4, v, true);
        else if (type === FormatType.Float && bits === 16) view.setUint16(c * 2, float32ToFloat16(v), true);
        else if (type === FormatType.Unorm && bits === 8) out[c] = Math.round(Math.min(Math.max(v, 0), 1) * 255);
        else if (color.every((x) => x === 0)) out.fill(0);
        else return null;
    }
    return out;
}

export class RenderContext extends ComputeContext {
    /** Attaches profiler timestamps to a render-pass descriptor when active. */
    private withTimestamps(desc: GPURenderPassDescriptor): GPURenderPassDescriptor {
        const tw = this.device.profilerHook?.passTimestampWrites();
        if (tw) desc.timestampWrites = tw;
        return desc;
    }

    private blitPipelines = new Map<string, GPURenderPipeline>();
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

    /** Mirrors RenderContext::clearFbo: every color target, then depth/stencil, as `flags` selects. */
    clearFbo(fbo: Fbo, color: [number, number, number, number], depth: number, stencil: number, flags = FboAttachmentType.All): void {
        if (flags & FboAttachmentType.Color) {
            for (let i = 0; i < fbo.getColorAttachmentCount(); i++) {
                const view = fbo.getRenderTargetView(i);
                if (view) this.clearRtv(view, color);
            }
        }
        const dsv = fbo.getDepthStencilView();
        if (dsv && flags & (FboAttachmentType.Depth | FboAttachmentType.Stencil)) {
            const clearStencil = (flags & FboAttachmentType.Stencil) !== 0 && fbo.getDepthStencilTexture()!.gpuFormat.includes("stencil");
            this.clearDsv(dsv, depth, stencil, (flags & FboAttachmentType.Depth) !== 0, clearStencil);
        }
    }

    /** Mirrors RenderContext::clearTexture (color textures). */
    clearTexture(texture: Texture, color: [number, number, number, number] = [0, 0, 0, 0]): void {
        if (isDepthFormat(texture.format)) {
            this.clearDsv(texture.getDSV(), color[0], 0);
            return;
        }
        // Native picks RTV or UAV clears by bind flags; a storage-only texture can't be a render
        // attachment in WebGPU, so it is filled by a (recorded, hence ordered) buffer copy.
        if (texture.bindFlags & ResourceBindFlags.RenderTarget) {
            for (let mip = 0; mip < texture.mipCount; mip++) this.clearRtv(texture.getRTV(mip), color);
            return;
        }
        const texel = encodeClearTexel(texture.format, color);
        if (!texel) {
            Logger.warning(`RenderContext::clearTexture() - Unsupported texture format ${ResourceFormat[texture.format]} for a non-render-target clear.`);
            return;
        }
        const layers = texture.gpuTexture.depthOrArrayLayers;
        for (let mip = 0; mip < texture.mipCount; mip++) {
            const [w, h] = [Math.max(1, texture.width >> mip), Math.max(1, texture.height >> mip)];
            const bytesPerRow = Math.ceil((w * texel.length) / 256) * 256;
            const data = new Uint8Array(bytesPerRow * h * layers);
            for (let z = 0; z < h * layers; z++) for (let x = 0; x < w; x++) data.set(texel, z * bytesPerRow + x * texel.length);
            const staging = this.device.gpuDevice.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
            new Uint8Array(staging.getMappedRange()).set(data);
            staging.unmap();
            this.getEncoder().copyBufferToTexture({ buffer: staging, bytesPerRow, rowsPerImage: h }, { texture: texture.gpuTexture, mipLevel: mip }, [w, h, layers]);
        }
    }

    /**
     * Mirrors RenderContext::blit: draws src into dst with optional filtering and,
     * like the complex blit, per-channel min/max reduction. Identical full-resource
     * blits take native's copyResource fast path.
     */
    blit(
        src: Texture,
        dst: Texture,
        filter: GPUFilterMode = "linear",
        srcMip = 0,
        dstMip = 0,
        srcLayer = 0,
        dstLayer = 0,
        reductions: readonly TextureReductionMode[] = kStandardReductions,
    ): void {
        if (isDepthFormat(dst.format)) throw new RuntimeError("blit to depth target not supported (use copy)");
        const complex = reductions.some((r) => r !== TextureReductionMode.Standard);
        const srcKind = blitKind(src.format);
        if (complex && srcKind !== "f32") throw new RuntimeError("RenderContext::blit() requires non-integer source format for complex blit");
        if (src.sampleCount > 1 && complex) throw new RuntimeError("RenderContext::blit() does not support complex blit for multisampled textures");
        if (src.sampleCount > 1 && srcKind !== "f32") throw new RuntimeError("RenderContext::blit() does not support sample count > 1 for integer source formats");
        if (
            !complex && src !== dst && src.format === dst.format && src.width === dst.width && src.height === dst.height &&
            src.mipCount === 1 && dst.mipCount === 1 && src.arraySize === 1 && dst.arraySize === 1 && src.sampleCount === dst.sampleCount
        ) {
            this.getEncoder().copyTextureToTexture({ texture: src.gpuTexture }, { texture: dst.gpuTexture }, [src.width, src.height, 1]);
            return;
        }
        const dstKind = blitKind(dst.format);
        const key = `${dst.gpuFormat}|${srcKind}|${reductions.join(",")}|${filter}|${src.sampleCount}`;
        let pipeline = this.blitPipelines.get(key);
        if (!pipeline) {
            const module = this.device.gpuDevice.createShaderModule({ code: blitWgsl(srcKind, dstKind, reductions, filter === "linear", src.sampleCount) });
            pipeline = this.device.gpuDevice.createRenderPipeline({
                layout: "auto",
                vertex: { module, entryPoint: "vsMain" },
                fragment: { module, entryPoint: "psMain", targets: [{ format: dst.gpuFormat }] },
                primitive: { topology: "triangle-list" },
            });
            this.blitPipelines.set(key, pipeline);
        }
        const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: src.getSRV(srcMip, 1, srcLayer, 1) }];
        if (srcKind === "f32" && src.sampleCount === 1) {
            let sampler = this.blitSamplers.get(filter);
            if (!sampler) {
                sampler = this.device.gpuDevice.createSampler({ magFilter: filter, minFilter: filter });
                this.blitSamplers.set(filter, sampler);
            }
            entries.push({ binding: 1, resource: sampler });
        }
        const bindGroup = this.device.gpuDevice.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const pass = this.getEncoder().beginRenderPass(this.withTimestamps({
            colorAttachments: [{ view: dst.getRTV(dstMip, dstLayer), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" }],
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
