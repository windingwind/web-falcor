/**
 * OVERALL VERIFY — PathTracer fixed samplesPerPixel > 1: the upstream
 * PathTracer.py graph with samplesPerPixel=4 over cornell_box.pyscene,
 * single frame, diffed against the native capture (color + tonemapped).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-ptspp4.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("ImageTestPTSpp4.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    let source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/PathTracer.py")).text();
    // Native oracle applies updatePass samplesPerPixel=4; the web pass takes
    // props at construction, so patch them into the unmodified graph script.
    expectEq(source.includes("{'samplesPerPixel': 1}"), true, "graph script has the spp prop");
    source = source.replace("{'samplesPerPixel': 1}", "{'samplesPerPixel': 4}");
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    // Radiance: stochastic-content policy (bad-pixel gate at 0.05); the 4x
    // average tightens the tail vs the 1-spp test.
    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("PathTracer.color")!)).buffer);
    const res = await fetch("/tests/oracle/out-native/oracle-imgtest-ptspp4.PathTracer.color.0.exr");
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
            if (pixelMax > 0.05) bad++;
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# imgtestPTSpp4.color: mean=${mean.toExponential(2)} bad=${bad}`);
    expectEq(mean < 1e-3, true, `color mean ${mean}`);
    expectEq(bad <= 300, true, `color bad pixels ${bad}`);

    // Final tonemapped frame (sRGB bytes).
    const webPng = new Uint8Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    const blob = await (await fetch("/tests/oracle/out-native/oracle-imgtest-ptspp4.ToneMapper.dst.0.png")).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    const canvas = new OffscreenCanvas(size, size);
    const c2d = canvas.getContext("2d", { willReadFrequently: true })!;
    c2d.drawImage(bitmap, 0, 0);
    const nat = c2d.getImageData(0, 0, size, size).data;
    let mse = 0;
    for (let i = 0; i < size * size; i++) {
        for (let ch = 0; ch < 3; ch++) {
            const d = (webPng[i * 4 + ch]! - nat[i * 4 + ch]!) / 255;
            mse += d * d;
        }
    }
    mse /= size * size * 3;
    console.error(`# imgtestPTSpp4.tonemapped: mse=${mse.toExponential(2)}`);
    expectEq(mse < 5e-4, true, `tonemapped MSE ${mse}`);
});
