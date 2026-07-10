/**
 * FEATURE VERIFY — curve geometry (linear swept spheres): the upstream
 * two_curves.pyscene rendered through SceneDebugger FaceNormal, compared
 * per-pixel against native (ID-free; curve segments intersected by the
 * software CurveIntersector loop).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-curves.py --headless
 *   (needs build/.../plugins/USDImporter.so — re-enable from plugins-disabled/)
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq, saveArtifact } from "../harness/registry.js";

const kGraph = `
from falcor import *
g = RenderGraph("SceneDebugger")
g.addPass(createPass("SceneDebugger", {'mode': 'FaceNormal'}), "SceneDebugger")
g.markOutput("SceneDebugger.output")
m.addGraph(g)
`;

gpuTest("FeatureCurves.faceNormalsMatchNative", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const [graph] = await runGraphScript(device, kGraph);

    let sceneSource = await (await fetch("/Falcor/media/test_scenes/curves/two_curves.pyscene")).text();
    // Web approximates custom primitives as visible box meshes (documented
    // divergence); native renders nothing for them here. Strip for the
    // curve-focused compare.
    sceneSource = sceneSource.replace(/sceneBuilder\.addCustomPrimitive.*/g, "pass");
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes/curves");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("SceneDebugger.output")!)).buffer);
    const res = await fetch("/tests/oracle/out-native/oracle-curves-debug.SceneDebugger.output.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let bad = 0;
    let webCurve = 0;
    let natCurve = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const wi = (y * size + x) * 4;
            const ni = ((height - 1 - y) * width + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) pixelMax = Math.max(pixelMax, Math.abs(web[wi + c]! - data[ni + c]!));
            if (pixelMax > 1e-2) bad++;
            // Curve pixels differ from the background/mesh pseudocolors; count
            // via nonblack-and-not-background heuristic per side for coverage.
            if (web[wi]! + web[wi + 1]! + web[wi + 2]! > 0.01) webCurve++;
            if (data[ni]! + data[ni + 1]! + data[ni + 2]! > 0.01) natCurve++;
        }
    }
    const toByte = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255);
    await saveArtifact("curves-web", Array.from({ length: size * size * 4 }, (_x, i) => (i % 4 === 3 ? 255 : toByte(web[i]!))), size, size, false);
    await saveArtifact("curves-native", Array.from({ length: size * size * 4 }, (_x, i) => {
        if (i % 4 === 3) return 255;
        const p2 = Math.floor(i / 4);
        return toByte(data[((size - 1 - Math.floor(p2 / size)) * size + (p2 % size)) * 4 + (i % 4)]!);
    }), size, size, false);
    console.error(`# featureCurves: bad=${bad} webNonzero=${webCurve} natNonzero=${natCurve}`);
    expectEq(Math.abs(webCurve - natCurve) < 500, true, `coverage web=${webCurve} native=${natCurve}`);
    expectEq(bad <= 500, true, `mismatched pixels ${bad}`);
});
