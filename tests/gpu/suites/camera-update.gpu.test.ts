/**
 * Regression probe: per-frame camera updates must reach the GPU. Renders
 * GBufferRT.diffuseOpacity, moves the camera (position + fixed target),
 * renders again — the two frames must differ substantially.
 */

import { float3, initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;
const graphSrc = `
from falcor import *
g = RenderGraph("CamUpdate")
GBufferRT = createPass("GBufferRT")
g.addPass(GBufferRT, "GBufferRT")
g.markOutput("GBufferRT.diffuseOpacity")
try: m.addGraph(g)
except NameError: None
`;

gpuTest("Camera.perFrameUpdateReachesGpu", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const [graph] = await runGraphScript(device, graphSrc);
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;

    const p0 = scene.camera.getPosition();
    const t0 = scene.camera.getTarget();

    // Variant A: readback between frames (forces submit).
    graph!.execute(ctx);
    const frame1 = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("GBufferRT.diffuseOpacity")!)).buffer);
    scene.camera.setPosition(new float3(p0.x + 0.1875, p0.y, p0.z));
    scene.camera.setTarget(t0);
    graph!.execute(ctx);
    const frame2 = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("GBufferRT.diffuseOpacity")!)).buffer);
    let diffA = 0;
    for (let i = 0; i < size * size; i++) diffA += Math.abs(frame1[i * 4]! - frame2[i * 4]!);
    diffA /= size * size;

    // Variant B: no readback between frames (single encoder for both).
    scene.camera.setPosition(p0);
    scene.camera.setTarget(t0);
    graph!.execute(ctx);
    scene.camera.setPosition(new float3(p0.x + 0.1875, p0.y, p0.z));
    scene.camera.setTarget(t0);
    graph!.execute(ctx);
    const frame4 = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("GBufferRT.diffuseOpacity")!)).buffer);
    let diffB = 0;
    for (let i = 0; i < size * size; i++) diffB += Math.abs(frame2[i * 4]! - frame4[i * 4]!);
    diffB /= size * size;

    console.error(`# camera update: moved-vs-static diffA=${diffA.toExponential(2)} (submit between), sameMove-vs-sameMove diffB=${diffB.toExponential(2)} (single encoder)`);
    expectEq(diffA > 1e-2, true, `camera move must change the image (diffA=${diffA})`);
    expectEq(diffB < 1e-6, true, `same camera path must give identical images (diffB=${diffB})`);
});
