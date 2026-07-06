/**
 * OVERALL VERIFY — upstream image test replicated 1:1: the UNMODIFIED
 * tests/image_tests/renderpasses/graphs/MinimalPathTracer.py graph script
 * (VBufferRT -> MinimalPathTracer -> AccumulatePass -> ToneMapper) over the
 * UNMODIFIED cornell_box.pyscene, compared against native Mogwai running the
 * same two files.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-mpt.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("ImageTestMPT.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");

    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);

    const ctx = device.renderContext;
    graph!.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-imgtest-mpt.ToneMapper.dst.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let badPixels = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const webIdx = (y * size + x) * 4;
            const natIdx = ((height - 1 - y) * width + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(web[webIdx + c]! - data[natIdx + c]!);
                sum += d;
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 0.05) badPixels++;
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# imgtestMPT: meanAbs=${mean.toExponential(2)} bad=${badPixels}`);
    expectEq(mean < 5e-3, true, `mean abs diff ${mean}`);
    expectEq(badPixels < size * 4, true, `bad pixels ${badPixels}`);
});
