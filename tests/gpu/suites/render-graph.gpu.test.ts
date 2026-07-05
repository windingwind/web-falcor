/**
 * M4 render-graph GPU tests: graph compilation + execution with real passes
 * (AccumulatePass via override shader, ToneMapper via unmodified upstream
 * pixel shader, BlitPass), mirroring the standard upstream graph topology.
 */

import { RenderGraph, ResourceFormat, ResourceBindFlags, createPass } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("RenderGraph.accumulateToneMapChain", async ({ device }) => {
    const w = 16, h = 16;
    const graph = new RenderGraph(device, "TestGraph");
    graph.onResize(w, h);

    const accumulate = createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" });
    const toneMapper = createPass(device, "ToneMapper", { operator: "Linear", clamp: true, exposureCompensation: 0 });
    graph.addPass(accumulate, "AccumulatePass");
    graph.addPass(toneMapper, "ToneMapper");
    graph.addEdge("AccumulatePass.output", "ToneMapper.src");
    graph.markOutput("ToneMapper.dst");

    // External input: constant 0.25 texture (as if a renderer produced it).
    const input = device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(w * h * 4).fill(0.25));
    graph.setInput("AccumulatePass.input", input);

    const ctx = device.renderContext;
    // Execute 4 frames; accumulation of a constant stays the constant.
    for (let i = 0; i < 4; i++) graph.execute(ctx);

    const out = graph.getOutput("ToneMapper.dst")!;
    expectEq(out.gpuFormat, "rgba8unorm-srgb", "tone mapper output format");
    const px = await ctx.readTextureSubresource(out);
    // Linear tone map of 0.25 -> sRGB-encoded byte ~ 137 (0.25^(1/2.2)-ish per sRGB EOTF).
    const srgb = (v: number) => (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
    expectClose(px[0]! / 255, srgb(0.25), 0.01, "sRGB-encoded value");
    input.destroy();
});

gpuTest("RenderGraph.accumulationConverges", async ({ device }) => {
    const w = 8, h = 8;
    const graph = new RenderGraph(device, "AccumGraph");
    graph.onResize(w, h);
    graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Accum");
    graph.markOutput("Accum.output");

    const frameA = device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(w * h * 4).fill(1.0));
    const frameB = device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(w * h * 4).fill(0.0));

    const ctx = device.renderContext;
    // Alternate 1.0 and 0.0 inputs; average over 4 frames = 0.5.
    for (let i = 0; i < 4; i++) {
        graph.setInput("Accum.input", i % 2 === 0 ? frameA : frameB);
        // setInput invalidates compilation; state (frame count) lives in the pass.
        graph.execute(ctx);
    }
    const px = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("Accum.output")!)).buffer);
    expectClose(px[0]!, 0.5, 1e-6, "4-frame average");
    frameA.destroy();
    frameB.destroy();
});

gpuTest("RenderGraph.blitTopologicalOrder", async ({ device }) => {
    // Chain of three blits: verifies topo sort and edge resolution.
    const w = 4, h = 4;
    const graph = new RenderGraph(device, "BlitChain");
    graph.onResize(w, h);
    graph.addPass(createPass(device, "BlitPass"), "B1");
    graph.addPass(createPass(device, "BlitPass"), "B2");
    graph.addPass(createPass(device, "BlitPass"), "B3");
    // Intentionally added out of order.
    graph.addEdge("B2.dst", "B3.src");
    graph.addEdge("B1.dst", "B2.src");
    graph.markOutput("B3.dst");

    const input = device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(w * h * 4).fill(0.75));
    graph.setInput("B1.src", input);
    graph.execute(device.renderContext);
    const px = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("B3.dst")!)).buffer);
    expectClose(px[0]!, 0.75, 1e-6, "value survives blit chain");
    input.destroy();
});
