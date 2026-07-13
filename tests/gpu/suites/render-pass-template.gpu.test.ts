/**
 * RenderPassTemplate: the authoring skeleton registers, reflects its fields,
 * and passes data through a graph unchanged.
 */

import { RenderGraph, createPass, initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("RenderPassTemplate.passesThrough", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const g = new RenderGraph(device, "TemplateTest");
    g.addPass(createPass(device, "ImageLoader", { filename: "test_scenes/textures/checker_tile_base_color.png", mips: false, srgb: false }), "ImageLoader");
    g.addPass(createPass(device, "RenderPassTemplate", {}), "Template");
    g.addEdge("ImageLoader.dst", "Template.src");
    g.markOutput("Template.dst");
    g.onResize(128, 128);
    await g.init();
    const ctx = device.renderContext;
    g.execute(ctx);
    const out = new Float32Array((await ctx.readTextureSubresource(g.getOutput("Template.dst")!)).buffer);
    let nonzero = 0;
    for (let i = 0; i < out.length; i += 4) if (out[i]! > 0 || out[i + 1]! > 0 || out[i + 2]! > 0) nonzero++;
    console.error(`# template: ${nonzero} nonzero px`);
    expectEq(nonzero > 1000, true, `data flows through (${nonzero})`);
});
