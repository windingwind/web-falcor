/**
 * FEATURE VERIFY — TAA pass vs native. Mirrors the upstream TAA.py image
 * test 1:1 except GBufferRaster -> GBufferRT (native driver lacks ROV, so
 * the raster graph cannot produce an oracle; GBufferRT channels are verified
 * elsewhere). 8 static frames with Halton camera jitter: per-frame varying
 * input exercises the UNMODIFIED TAA kernel (YCgCo neighborhood clamp,
 * Catmull-Rom history fetch, anti-flicker). Scripted per-frame camera moves
 * are deliberately avoided — native Mogwai's camera controller composes them
 * statefully, so they are not reproducible cross-implementation.
 *
 * Known divergence (documented, DESIGN §9): native allocates the
 * Unknown-format colorOut as BGRA8UnormSrgb (swapchain default), so its
 * history feedback quantizes to sRGB bytes each frame; the web allocator
 * defaults to RGBA32Float (full-float history). Flat-albedo regions are
 * unaffected (clamp box is degenerate); jittered edges may drift a few LSB.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-feature-taa.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("FeatureTAA.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/graphs/taa-feature.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let k = 0; k < 8; k++) graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("TAA.colorOut")!)).buffer);

    const blob = await (await fetch("/tests/oracle/out-native/oracle-feature-taa.TAA.colorOut.0.png")).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    const canvas = new OffscreenCanvas(size, size);
    const c2d = canvas.getContext("2d", { willReadFrequently: true })!;
    c2d.drawImage(bitmap, 0, 0);
    const nat = c2d.getImageData(0, 0, size, size).data;

    const toSrgbByte = (v: number) => {
        const c = Math.min(Math.max(v, 0), 1);
        const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
        return Math.round(s * 255);
    };
    let mse = 0;
    let maxDiff = 0;
    let maxAt = [0, 0];
    let bad = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const i = y * size + x;
            let pixelMax = 0;
            for (let ch = 0; ch < 3; ch++) {
                const d = Math.abs(toSrgbByte(web[i * 4 + ch]!) - nat[i * 4 + ch]!);
                mse += (d / 255) * (d / 255);
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > maxDiff) {
                maxDiff = pixelMax;
                maxAt = [x, y];
            }
            if (pixelMax > 3) bad++;
        }
    }
    mse /= size * size * 3;
    console.error(`# oracle-feature-taa.TAA.colorOut.0.png: mse=${mse.toExponential(2)} maxByteDiff=${maxDiff} at=${maxAt} bad(>3LSB)=${bad}`);
    expectEq(mse < 2e-5, true, `TAA colorOut MSE ${mse}`);
    expectEq(bad <= 300, true, `TAA colorOut pixels >3 LSB: ${bad}`);
});
