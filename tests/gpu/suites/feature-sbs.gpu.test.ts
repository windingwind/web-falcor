/**
 * FEATURE VERIFY — SparseBrickSet SDF grid on GPU. SDFSBS.pyscene
 * (SDFGrid.createSBS + generateCheeseValues(128, 0) — the SAME cheese
 * SURFACE as NDSDFGrid.pyscene, SparseBrickSet representation), SceneDebugger
 * mode 'GeometryID', 640x360, frame 64.
 *
 * Native cannot oracle SBS on this machine: Mogwai core-dumps building the
 * procedural-AABB acceleration structure for the brick set (same local
 * native-artifact class as the NDSDF gradient NaNs / openvdb segfault,
 * DESIGN.md 6.3). SBS is instead cross-validated against the CPU ALGORITHM
 * footprint oracle (tests/oracle/assets/ndsdf-cpu-footprint.bin): the brick
 * set encodes the identical surface, so its GeometryID body must match the
 * NDSDF footprint that the web renderer already reproduces PIXEL-EXACTLY
 * (feature-ndsdf). A small edge tolerance covers the two representations'
 * differing quantization (SBS snorm8 over the whole grid vs NDSDF's
 * narrow-band LODs) and their differing tracers (per-brick DDA vs LOD sphere
 * tracing).
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const width = 640;
const height = 360;

gpuTest("FeatureSBSGrid.matchesAlgorithmOracle", async ({ device }) => {
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
    const sceneSource = await (await fetch("/Falcor/tests/image_tests/scene/scenes/SDFSBS.pyscene")).text();
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
    console.error(`# sbs: webHits=${webHits} mismatch-vs-ndsdf-footprint=${mismatch} (of ${width * height})`);
    // Observed: webHits 34355 vs NDSDF 34356, 3 disputed pixels — the two
    // representations of the identical surface agree to silhouette-edge noise.
    expectEq(webHits > 30000, true, `SBS body too small (${webHits})`);
    expectEq(mismatch <= 50, true, `SBS-vs-NDSDF footprint mismatch ${mismatch}`);
});
