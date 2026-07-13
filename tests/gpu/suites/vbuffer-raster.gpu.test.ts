/**
 * VBufferRaster (non-indexed vertex-pulling raster) cross-verified against
 * the ray-traced VBufferRT over cornell_box: hit headers (instance +
 * primitive) must agree except for a small rasterization-edge tail, and
 * barycentrics must match closely where headers agree.
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("VBufferRaster.matchesVBufferRT", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    const ctx = device.renderContext;

    const render = async (passType: string) => {
        const g = new RenderGraph(device, passType);
        g.addPass(createPass(device, passType, { samplePattern: "Center" }), "Pass");
        g.markOutput("Pass.vbuffer");
        g.onResize(size, size);
        g.setScene(scene);
        await g.init();
        g.execute(ctx);
        return new Uint32Array((await ctx.readTextureSubresource(g.getOutput("Pass.vbuffer")!)).buffer);
    };

    const rt = await render("VBufferRT");
    const raster = await render("VBufferRaster");

    const f32 = (u: number) => new Float32Array(new Uint32Array([u]).buffer)[0]!;
    let headerMismatch = 0;
    let baryBad = 0;
    let hits = 0;
    for (let i = 0; i < size * size; i++) {
        const a = [rt[i * 4]!, rt[i * 4 + 1]!, rt[i * 4 + 2]!, rt[i * 4 + 3]!];
        const b = [raster[i * 4]!, raster[i * 4 + 1]!, raster[i * 4 + 2]!, raster[i * 4 + 3]!];
        if (a[0] !== b[0] || a[1] !== b[1]) {
            headerMismatch++;
            continue;
        }
        if (a[0] === 0 && a[1] === 0) continue; // both background
        hits++;
        if (Math.abs(f32(a[2]!) - f32(b[2]!)) > 2e-3 || Math.abs(f32(a[3]!) - f32(b[3]!)) > 2e-3) baryBad++;
    }
    console.error(`# vbuffer: hits=${hits} headerMismatch=${headerMismatch} baryBad=${baryBad}`);
    expectEq(hits > size * size * 0.5, true, `enough hit pixels (${hits})`);
    expectEq(headerMismatch <= size * size * 0.01, true, `header mismatches ${headerMismatch}`);
    expectEq(baryBad <= hits * 0.001, true, `barycentric outliers ${baryBad}`);
});
