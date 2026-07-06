/**
 * OVERALL VERIFY — upstream image test: the UNMODIFIED VBufferRTInline.py
 * graph (useTraceRayInline, the web-default path) over cornell_box.pyscene;
 * depth/mvec/viewW channels compared against native EXR captures.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-vbufferinline.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("ImageTestVBufferInline.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");

    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/VBufferRTInline.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    // Compare float channels (vbuffer is packed uints; hit equality is
    // covered indirectly by every PT oracle already).
    // mvec tolerance is loose: native's frame-0 prev-camera state differs from
    // prev==current on web, adding ~1e-3 offsets on a static scene.
    for (const [channel, components, tol, badTol] of [
        ["depth", 1, 1e-3, 200],
        ["mvec", 2, 5e-2, 65536],
        ["viewW", 3, 1e-3, 200],
    ] as const) {
        const tex = graph!.getOutput(`VBufferRT.${channel}`)!;
        const web = new Float32Array((await ctx.readTextureSubresource(tex)).buffer);
        const webComponents = web.length / (size * size); // padded per format

        const res = await fetch(`/tests/oracle/out-native/oracle-imgtest-vbufferinline.VBufferRT.${channel}.0.exr`);
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        expectEq(width, size, `${channel} oracle resolution`);

        let sum = 0;
        let badPixels = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * webComponents;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pixelMax = 0;
                for (let c = 0; c < components; c++) {
                    const d = Math.abs(web[wi + c]! - data[ni + c]!);
                    sum += d;
                    pixelMax = Math.max(pixelMax, d);
                }
                if (pixelMax > 1e-3) badPixels++;
            }
        }
        const mean = sum / (size * size * components);
        console.error(`# imgtestVBufferInline.${channel}: mean=${mean.toExponential(2)} bad=${badPixels}`);
        expectEq(mean < tol, true, `${channel} mean ${mean}`);
        expectEq(badPixels <= badTol, true, `${channel} bad pixels ${badPixels}`);
    }
});
