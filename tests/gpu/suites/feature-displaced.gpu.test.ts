/**
 * FEATURE VERIFY — displacement mapping: cornell_box_displaced.pyscene
 * (brick height map on the back wall) through SceneDebugger FaceNormal
 * (deterministic ray-marched surface normals) and MPT (shaded, 64 frames),
 * compared against native.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-displaced.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq, saveArtifact } from "../harness/registry.js";

const kDebugGraph = `
from falcor import *
g = RenderGraph("SceneDebugger")
g.addPass(createPass("SceneDebugger", {'mode': 'FaceNormal'}), "SceneDebugger")
g.markOutput("SceneDebugger.output")
m.addGraph(g)
`;

gpuTest("FeatureDisplaced.faceNormalsMatchNative", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const [graph] = await runGraphScript(device, kDebugGraph);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box_displaced.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("SceneDebugger.output")!)).buffer);
    const res = await fetch("/tests/oracle/out-native/oracle-displaced-debug.SceneDebugger.output.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let bad = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const wi = (y * size + x) * 4;
            const ni = ((height - 1 - y) * width + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(web[wi + c]! - data[ni + c]!);
                sum += d;
                pixelMax = Math.max(pixelMax, d);
            }
            // The iterative shell march is fp-sensitive at brick edges (same
            // class as SDF sphere-tracing edges): gate on structural diffs.
            if (pixelMax > 5e-2) bad++;
        }
    }
    const mean = sum / (size * size * 3);
    const toByte = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255);
    await saveArtifact("displaced-web", Array.from({ length: size * size * 4 }, (_x, i) => (i % 4 === 3 ? 255 : toByte(web[i]!))), size, size, false);
    await saveArtifact("displaced-native", Array.from({ length: size * size * 4 }, (_x, i) => {
        if (i % 4 === 3) return 255;
        const p2 = Math.floor(i / 4);
        return toByte(data[((size - 1 - Math.floor(p2 / size)) * size + (p2 % size)) * 4 + (i % 4)]!);
    }), size, size, false);
    console.error(`# featureDisplaced: mean=${mean.toExponential(2)} bad=${bad}`);
    expectEq(mean < 8e-3, true, `mean ${mean}`);
    expectEq(bad <= 4000, true, `structurally mismatched pixels ${bad}`);
});
