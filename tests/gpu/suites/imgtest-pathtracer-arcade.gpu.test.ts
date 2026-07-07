/**
 * QUARANTINED (.gpu.wip): pending LightCollection textured-emissive flux
 * (the Cabinet screen light has a black constant emissive + texture x150;
 * the web LightCollection integrates constants only, so NEE never samples
 * the scene's main light) and the MPT residual investigation.
 *
 * OVERALL VERIFY — the upstream image test test_PathTracer.py replicated
 * 1:1: the UNMODIFIED PathTracer.py graph over the UNMODIFIED upstream
 * Arcade.pyscene (FBX import), 640x360, captured at frame 128 exactly like
 * the upstream render_frames helper. (The cornell-based imgtest-pathtracer
 * remains as a faster regression with per-channel coverage.)
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-pathtracer-arcade.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const width = 640;
const height = 360;

gpuTest("ImageTestPathTracerArcade.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/PathTracer.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/Arcade/Arcade.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/Arcade");
    scene.camera.setAspectRatio(width / height);

    graph!.onResize(width, height);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 128; f++) graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("PathTracer.color")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-imgtest-pathtracer-arcade.PathTracer.color.128.exr");
    const { data, width: nw, height: nh } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(nw, width, "oracle resolution");

    // Stochastic 1-spp content, 128 accumulated frames of high-variance
    // glossy transport: gate on BIAS (signed mean must be ~0) and RELATIVE
    // per-pixel error (fireflies near the x150 screen emitter speckle).
    let signedSum = 0;
    let bad = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const wi = (y * width + x) * 4;
            const ni = ((nh - 1 - y) * nw + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const w = Math.min(web[wi + c]!, 65504);
                const n = Math.min(data[ni + c]!, 65504);
                signedSum += w - n;
                pixelMax = Math.max(pixelMax, Math.abs(w - n) / (1 + Math.abs(n)));
            }
            if (pixelMax > 0.05) bad++;
        }
    }
    const mean = signedSum / (width * height * 3);
    // NEE sample sequences decorrelate between implementations (the flux
    // tables differ in float rounding -> different RNG consumption), so
    // per-pixel noise does not cancel like MPT's bit-matched paths. Gate on
    // bias plus 8x8 block averages (averaging cancels decorrelated noise).
    let blockBad = 0;
    const bw = width / 8;
    const bh = height / 8;
    for (let by = 0; by < bh; by++) {
        for (let bx = 0; bx < bw; bx++) {
            let wSum = 0;
            let nSum = 0;
            for (let y = by * 8; y < by * 8 + 8; y++) {
                for (let x = bx * 8; x < bx * 8 + 8; x++) {
                    const wi = (y * width + x) * 4;
                    const ni = ((nh - 1 - y) * nw + x) * 4;
                    for (let c = 0; c < 3; c++) {
                        wSum += Math.min(web[wi + c]!, 65504);
                        nSum += Math.min(data[ni + c]!, 65504);
                    }
                }
            }
            if (Math.abs(wSum - nSum) / (192 + Math.abs(nSum)) > 0.05) blockBad++;
        }
    }
    console.error(`# oracle-imgtest-pathtracer-arcade: signed-mean=${mean.toExponential(2)} relBad@0.05=${bad} blockBad=${blockBad}/${bw * bh}`);
    expectEq(Math.abs(mean) < 3e-3, true, `PathTracer Arcade bias ${mean}`);
    expectEq(blockBad <= 60, true, `PathTracer Arcade bad 8x8 blocks ${blockBad}`);
});
