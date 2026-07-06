/**
 * OVERALL VERIFY — upstream image test: the UNMODIFIED SceneDebugger.py graph
 * (FaceNormal visualization over primary inline rays) over cornell_box,
 * diffed against the native EXR capture.
 *
 * Regenerate: Mogwai --script tests/oracle/render-native-imgtest-scenedebugger.py --headless
 *
 * (Historical: first landed all-black — an undersized pixelData buffer
 * (124 B C-packed vs 176 B WGSL std430, float3 pads to 16 B) invalidated
 * the whole command buffer. Structured-buffer sizes must use WGSL layout.)
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("ImageTestSceneDebugger.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/SceneDebugger.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("SceneDebugger.output")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-imgtest-scenedebugger.SceneDebugger.output.0.exr");
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
            if (pixelMax > 1e-3) bad++;
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# imgtestSceneDebugger: mean=${mean.toExponential(2)} bad=${bad}`);
    expectEq(mean < 1e-4, true, `mean ${mean}`);
    expectEq(bad <= 50, true, `bad ${bad}`);
});
