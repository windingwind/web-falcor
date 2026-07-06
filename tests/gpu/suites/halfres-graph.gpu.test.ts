/**
 * The UNMODIFIED upstream HalfRes.py graph (GBufferRaster half-res ->
 * Accumulate -> ToneMap -> SimplePostFX, all 'outputSize': 'Half' with
 * stratified camera jitter) over cornell_box.pyscene.
 *
 * ⚠ No native pixel oracle: the oracle machine's Vulkan driver lacks
 * rasterizer ordered views, so native Mogwai refuses to construct
 * GBufferRaster ("requires ROVs support"), and the upstream test scene
 * (Arcade.pyscene) additionally needs the FBX importer. This test verifies
 * the web side end-to-end: IOSize plumbing (128x128 outputs at a 256 window),
 * stratified jitter advancing per frame, and 16-frame accumulation.
 * The jitter sequence itself is pinned bit-exactly against gcc/libstdc++ in
 * packages/falcor/tests/sample-generators.test.ts.
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("HalfResGraph.halfSizeJitteredAccumulation", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/HalfRes.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    await graph!.init();
    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;

    // Frame 1: pattern generator attaches during execute (no jitter yet).
    graph!.execute(ctx);
    const normW = graph!.getOutput("GBuffer.normW")!;
    expectEq(normW.width, size / 2, "GBuffer.normW is half-res");
    expectEq(graph!.getOutput("SimplePostFX.dst")!.width, size / 2, "SimplePostFX.dst is half-res");
    expectEq(graph!.getOutput("ToneMapper.dst")!.width, size / 2, "ToneMapper.dst is half-res");
    const frame1 = new Float32Array((await ctx.readTextureSubresource(normW)).buffer);

    // Frame 2: first stratified sample applies -> geometry edges shift.
    graph!.execute(ctx);
    const frame2 = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("GBuffer.normW")!)).buffer);
    let changed = 0;
    for (let i = 0; i < frame1.length; i += 4) {
        if (Math.abs(frame1[i]! - frame2[i]!) > 1e-6) changed++;
    }
    expectEq(changed > 50, true, `jitter moved edge pixels (changed=${changed})`);

    // Frames 3..16: accumulation stays finite and averages the jittered frames.
    for (let f = 2; f < 16; f++) graph!.execute(ctx);
    const acc = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("AccumulatePass.output")!)).buffer);
    let sum = 0;
    let finite = true;
    for (let i = 0; i < acc.length; i += 4) {
        sum += acc[i]!;
        finite &&= Number.isFinite(acc[i]!);
    }
    expectEq(finite, true, "accumulated output finite");
    expectEq(sum !== 0, true, "accumulated output non-zero");
    // Accumulated normals are averages of unit-ish vectors: |mean| <= 1 + eps.
    const avg = sum / (acc.length / 4);
    expectEq(Math.abs(avg) <= 1.01, true, `plausible accumulated normW mean (${avg.toFixed(4)})`);
});
