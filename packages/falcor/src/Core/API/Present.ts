/**
 * Swapchain present: fullscreen-quad blit of a render-graph output Texture to
 * the canvas' current GPUTexture. Mirrors Mogwai's final present (native blits
 * the graph output to the swapchain). Format-converts to the swapchain format
 * (typically bgra8unorm) and tonemaps nothing — the graph is expected to end in
 * a display-ready output (ToneMapper.dst / PathTracer.color etc.).
 */

import type { Device } from "./Device.js";
import type { Texture } from "./Texture.js";

const kPresentWgsl = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VSOut;
    out.pos = vec4(p[vi], 0.0, 1.0);
    out.uv = vec2(0.5, -0.5) * p[vi] + vec2(0.5, 0.5);
    return out;
}
@fragment fn psMain(in: VSOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
`;

const pipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
let sampler: GPUSampler | null = null;

/** Blits `src` (a graph output) to `dst` (the swapchain's current texture). */
export function presentToCanvas(device: Device, src: Texture, dst: GPUTexture, format: GPUTextureFormat): void {
    let pipeline = pipelines.get(format);
    if (!pipeline) {
        const module = device.gpuDevice.createShaderModule({ code: kPresentWgsl });
        pipeline = device.gpuDevice.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vsMain" },
            fragment: { module, entryPoint: "psMain", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        pipelines.set(format, pipeline);
    }
    if (!sampler) sampler = device.gpuDevice.createSampler({ magFilter: "linear", minFilter: "linear" });

    const bindGroup = device.gpuDevice.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: src.getSRV(0, 1) },
            { binding: 1, resource: sampler },
        ],
    });

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
