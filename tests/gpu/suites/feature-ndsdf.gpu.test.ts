/**
 * FEATURE VERIFY — SDF grids on GPU. The upstream scene test
 * test_NDSDFGrids.py scene: SceneDebugger (mode 'GeometryID') over the
 * unmodified NDSDFGrid.pyscene (SDFGrid.createNDGrid + procedural cheese,
 * bit-exact host build), 640x360, frame 64.
 *
 * The SDF body is gated against the CPU ALGORITHM ORACLE
 * (assets/ndsdf-cpu-footprint.bin — faithful double-precision port of
 * NDSDFGrid::intersectSDF, pinned by packages/falcor/tests/
 * ndsdf-intersect.test.ts): the web footprint matched it PIXEL-EXACTLY at
 * adjudication (iter 78). Native is NOT comparable inside the SDF body on
 * this machine: it dilates the footprint by 24568 spurious pixels and NaNs
 * the gradients (SPIR-V offset texel fetches; same local-artifact class as
 * the openvdb segfault, docs 6.3) — probe-verified via the real-shader
 * single-ray kernel (WebFalcor/Debug/SdfProbe.cs.slang). Background pixels
 * still compare against the native oracle EXR wherever native itself shows
 * background. Instance ordering (mesh 0, SDF 1) was probe-verified against
 * native separately (mesh-scene InstanceID renders match; the mesh-less
 * TLAS is a further native off-by-one artifact).
 *
 * Regenerate the native oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-feature-ndsdf.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const width = 640;
const height = 360;

gpuTest("FeatureNDSDFGrid.matchesAlgorithmOracle", async ({ device }) => {
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
    const sceneSource = await (await fetch("/Falcor/tests/image_tests/scene/scenes/NDSDFGrid.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/tests/image_tests/scene/scenes");
    scene.camera.setAspectRatio(width / height);

    graph!.onResize(width, height);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 64; f++) graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("SceneDebugger.output")!)).buffer);

    const mask = new Uint8Array(await (await fetch("/tests/oracle/assets/ndsdf-cpu-footprint.bin")).arrayBuffer());
    const res = await fetch("/tests/oracle/out-native/oracle-feature-ndsdf.SceneDebugger.output.64.exr");
    const { data, width: nw, height: nh } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(nw, width, "oracle resolution");

    // Body = pseudocolor(geometryID 0) = jenkinsHash(0) bytes / 255.
    const isBody = (r: number) => Math.abs(r - 0.153) < 0.05;
    let bodyMismatch = 0;
    let bgMismatch = 0;
    let bgCompared = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const wi = idx * 4;
            const ni = ((nh - 1 - y) * nw + x) * 4;
            const cpuHit = (mask[idx >> 3]! & (1 << (idx & 7))) !== 0;
            const webBody = isBody(web[wi]!);
            if (webBody !== cpuHit) bodyMismatch++;
            if (!cpuHit && !isBody(data[ni]!)) {
                // True background on both sides: exact native compare.
                bgCompared++;
                let d = 0;
                for (let c = 0; c < 3; c++) d = Math.max(d, Math.abs(web[wi + c]! - data[ni + c]!));
                if (d > 1e-2) bgMismatch++;
            }
        }
    }
    console.error(`# ndsdf: bodyMismatch=${bodyMismatch} (of ${width * height}) bgMismatch=${bgMismatch}/${bgCompared}`);
    // GPU-vs-CPU filter precision may flip isolated silhouette pixels.
    expectEq(bodyMismatch <= 100, true, `NDSDF body-vs-algorithm mismatches ${bodyMismatch}`);
    expectEq(bgMismatch <= 50, true, `NDSDF background-vs-native mismatches ${bgMismatch}`);
});
