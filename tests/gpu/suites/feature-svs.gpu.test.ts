/**
 * FEATURE VERIFY — SparseVoxelSet SDF grid on GPU. SDFSVS.pyscene
 * (SDFGrid.createSVS + generateCheeseValues(128, 0) — the SAME cheese SURFACE
 * as NDSDFGrid.pyscene, SparseVoxelSet representation: one AABB + one packed
 * 4x4x4 neighborhood per surface voxel), SceneDebugger 'GeometryID', 640x360,
 * frame 64. Cross-validated against the shared CPU footprint oracle
 * (ndsdf-cpu-footprint.bin) — the identical surface the web renderer already
 * reproduces pixel-exactly. Native can't oracle SVS either (Mogwai core-dumps
 * building the voxel procedural-AABB acceleration structure, same local
 * artifact class as SBS). Practical only through the SDF primitive-AABB BVH
 * (tens of thousands of voxel AABBs).
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const width = 640;
const height = 360;

gpuTest("FeatureSVSGrid.matchesAlgorithmOracle", async ({ device }) => {
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
    const sceneSource = await (await fetch("/Falcor/tests/image_tests/scene/scenes/SDFSVS.pyscene")).text();
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
            const webBody = isBody(web[(idx * 4)]!);
            if (webBody) webHits++;
            if (webBody !== cpuHit) mismatch++;
        }
    }
    console.error(`# svs: webHits=${webHits} mismatch-vs-ndsdf-footprint=${mismatch} (of ${width * height})`);
    expectEq(webHits > 30000, true, `SVS body too small (${webHits})`);
    expectEq(mismatch <= 60, true, `SVS-vs-NDSDF footprint mismatch ${mismatch}`);
});
