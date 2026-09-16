/**
 * VBufferRaster (non-indexed vertex-pulling raster) cross-verified against
 * the ray-traced VBufferRT over cornell_box: hit headers (instance +
 * primitive) must agree except for a small rasterization-edge tail, and
 * barycentrics must match closely where headers agree.
 */

import { RenderGraph, createPass, float3, initScripting, runSceneScript } from "@web-falcor/falcor";
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

/** Optional mvec/mask channels (extra render targets, no ROVs on WebGPU) agree with VBufferRT. */
gpuTest("VBufferRaster.mvecMaskMatchVBufferRT", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    const ctx = device.renderContext;
    const startPos = scene.camera.getPosition();

    const render = async (passType: string) => {
        scene.camera.setPosition(startPos);
        const g = new RenderGraph(device, passType);
        g.addPass(createPass(device, passType, { samplePattern: "Center" }), "Pass");
        g.markOutput("Pass.mvec");
        g.markOutput("Pass.mask");
        g.onResize(size, size);
        g.setScene(scene);
        await g.init();
        // Two static frames: the camera is shared across graphs, so the first frame's
        // previous matrix may still be the other run's; the second frame has prev == cur.
        g.execute(ctx);
        g.execute(ctx);
        const read = async (name: string) => new Float32Array((await ctx.readTextureSubresource(g.getOutput(`Pass.${name}`)!)).buffer);
        const staticMvec = await read("mvec");
        const mask = await read("mask");
        // frame 1: dolly the camera so the motion vectors become non-zero.
        scene.camera.setPosition(new float3(startPos.x + 0.05, startPos.y, startPos.z));
        g.execute(ctx);
        const movedMvec = await read("mvec");
        return { staticMvec, mask, movedMvec };
    };
    const rt = await render("VBufferRT");
    const raster = await render("VBufferRaster");

    let hits = 0;
    let maskMismatch = 0;
    let staticNonZero = 0;
    let movedBad = 0;
    let movedNonZero = 0;
    for (let i = 0; i < size * size; i++) {
        if (rt.mask[i] !== raster.mask[i]) maskMismatch++;
        if (rt.mask[i] !== 1 || raster.mask[i] !== 1) continue;
        hits++;
        if (Math.abs(raster.staticMvec[i * 2]!) > 1e-5 || Math.abs(raster.staticMvec[i * 2 + 1]!) > 1e-5) staticNonZero++;
        const dx = Math.abs(rt.movedMvec[i * 2]! - raster.movedMvec[i * 2]!);
        const dy = Math.abs(rt.movedMvec[i * 2 + 1]! - raster.movedMvec[i * 2 + 1]!);
        if (dx > 2e-3 || dy > 2e-3) movedBad++;
        if (Math.abs(raster.movedMvec[i * 2]!) > 1e-4) movedNonZero++;
    }
    console.error(`# vbuffer mvec/mask: hits=${hits} maskMismatch=${maskMismatch} staticNonZero=${staticNonZero} movedBad=${movedBad} movedNonZero=${movedNonZero}`);
    expectEq(hits > size * size * 0.5, true, `enough hit pixels (${hits})`);
    expectEq(maskMismatch <= size * size * 0.01, true, `mask mismatches ${maskMismatch}`);
    expectEq(staticNonZero, 0, "static frame has zero motion vectors");
    expectEq(movedNonZero > hits * 0.9, true, `camera move produced motion (${movedNonZero}/${hits})`);
    expectEq(movedBad <= hits * 0.01, true, `mvec mismatches vs VBufferRT: ${movedBad}`);
});
