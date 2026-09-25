/**
 * Swapchain present: fullscreen-quad blit of a render-graph output Texture to
 * the canvas' current GPUTexture. Mirrors Mogwai's final present (native blits
 * the graph output to the swapchain). Format-converts to the swapchain format
 * (typically bgra8unorm) and tonemaps nothing — the graph is expected to end in
 * a display-ready output (ToneMapper.dst / PathTracer.color etc.).
 */

import type { Device } from "./Device.js";
import type { Texture } from "./Texture.js";
import { FormatType, getFormatType, isDepthFormat } from "./Formats.js";

/** How a texture is read: filtered floats, or integer/depth texels (not samplable) converted to float. */
type SampleKind = "float" | "uint" | "sint" | "depth";

const kVertexWgsl = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VSOut;
    out.pos = vec4(p[vi], 0.0, 1.0);
    out.uv = vec2(0.5, -0.5) * p[vi] + vec2(0.5, 0.5);
    return out;
}
`;

const kFragmentWgsl: Record<SampleKind, string> = {
    float: `@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn psMain(in: VSOut) -> @location(0) vec4<f32> { return textureSampleLevel(src, samp, in.uv, 0.0); }`,
    // Integer texels as floats (native blits them to the swapchain the same way: 0 black, >= 1 saturated).
    uint: `@group(0) @binding(0) var src: texture_2d<u32>;
@fragment fn psMain(in: VSOut) -> @location(0) vec4<f32> { return vec4<f32>(textureLoad(src, vec2<i32>(in.uv * vec2<f32>(textureDimensions(src))), 0)); }`,
    sint: `@group(0) @binding(0) var src: texture_2d<i32>;
@fragment fn psMain(in: VSOut) -> @location(0) vec4<f32> { return vec4<f32>(textureLoad(src, vec2<i32>(in.uv * vec2<f32>(textureDimensions(src))), 0)); }`,
    depth: `@group(0) @binding(0) var src: texture_depth_2d;
@fragment fn psMain(in: VSOut) -> @location(0) vec4<f32> { let d = textureLoad(src, vec2<i32>(in.uv * vec2<f32>(textureDimensions(src))), 0); return vec4<f32>(d, d, d, 1.0); }`,
};

const pipelines = new Map<string, GPURenderPipeline>();
let sampler: GPUSampler | null = null;

function sampleKind(src: Texture): SampleKind {
    if (isDepthFormat(src.format)) return "depth";
    const type = getFormatType(src.format);
    return type === FormatType.Uint ? "uint" : type === FormatType.Sint ? "sint" : "float";
}

/** Blits `src` (a graph output) to `dst` (the swapchain's current texture). */
export function presentToCanvas(device: Device, src: Texture, dst: GPUTexture, format: GPUTextureFormat): void {
    const kind = sampleKind(src);
    const key = `${format}:${kind}`;
    let pipeline = pipelines.get(key);
    if (!pipeline) {
        const module = device.gpuDevice.createShaderModule({ code: kVertexWgsl + kFragmentWgsl[kind] });
        pipeline = device.gpuDevice.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vsMain" },
            fragment: { module, entryPoint: "psMain", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        pipelines.set(key, pipeline);
    }
    if (!sampler) sampler = device.gpuDevice.createSampler({ magFilter: "linear", minFilter: "linear" });

    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: src.getSRV(0, 1) }];
    if (kind === "float") entries.push({ binding: 1, resource: sampler });
    const bindGroup = device.gpuDevice.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    // Flush the render graph's pending commands so the output is ready before
    // we sample it (present records on a separate encoder).
    device.renderContext.submit();

    const encoder = device.gpuDevice.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: dst.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.gpuDevice.queue.submit([encoder.finish()]);
}
