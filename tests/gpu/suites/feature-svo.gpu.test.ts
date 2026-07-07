/**
 * FEATURE VERIFY — SparseVoxelOctree SDF grid on GPU. SDFSVO.pyscene
 * (SDFGrid.createSVO + generateCheeseValues(128, 0) — same cheese SURFACE as
 * NDSDF/SBS/SVS, octree representation), SceneDebugger 'GeometryID', 640x360,
 * frame 64. SVO traces the whole grid as a single procedural primitive (an
 * in-shader octree walk over the `svo` buffer the CPU build produces).
 * Cross-validated against the shared CPU footprint oracle. Native can't
 * oracle SVO either (octree procedural-AABB BLAS core-dumps).
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const width = 640;
const height = 360;

gpuTest("FeatureSVOGrid.matchesAlgorithmOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = `
from falcor import *
g = RenderGraph('SceneDebugger')
SceneDebugger = createPass('SceneDebugger', {'mode': 'GeometryID'})
g.addPass(SceneDebugger, 'SceneDebugger')
g.markOutput('SceneDebugger.output')
m.addGraph(g)
`;
    const [graph] = await runGraphScript(device, source);
    const sceneSource = await (await fetch("/Falcor/tests/image_tests/scene/scenes/SDFSVO.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/tests/image_tests/scene/scenes");
    scene.camera.setAspectRatio(width / height);

    graph!.onResize(width, height);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 64; f++) graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("SceneDebugger.output")!)).buffer);
    const mask = new Uint8Array(await (await fetch("/tests/oracle/assets/ndsdf-cpu-footprint.bin")).arrayBuffer());

    const isBody = (r: number) => Math.abs(r - 0.153) < 0.05;
    let webHits = 0;
    let mismatch = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const cpuHit = (mask[idx >> 3]! & (1 << (idx & 7))) !== 0;
            const webBody = isBody(web[idx * 4]!);
            if (webBody) webHits++;
            if (webBody !== cpuHit) mismatch++;
        }
    }
    console.error(`# svo: webHits=${webHits} mismatch-vs-ndsdf-footprint=${mismatch} (of ${width * height})`);
    expectEq(webHits > 30000, true, `SVO body too small (${webHits})`);
    expectEq(mismatch <= 80, true, `SVO-vs-NDSDF footprint mismatch ${mismatch}`);
});
