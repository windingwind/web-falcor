/**
 * OVERALL VERIFY — the upstream image test test_RTXDI.py replicated 1:1:
 * the UNMODIFIED RTXDI.py graph (VBufferRT -> RTXDIPass -> ToneMapper) over
 * the UNMODIFIED upstream Arcade.pyscene, 640x360, captured at frames
 * 1/16/64 exactly like the upstream render_frames helper. The comparison
 * runs on RTXDIPass.color (linear pre-tonemap radiance; marked identically
 * on the native side).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-rtxdi-arcade.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const width = 640;
const height = 360;

gpuTest("ImageTestRTXDIArcade.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/RTXDI.py")).text();
    const [graph] = await runGraphScript(device, source + '\nRTXDI.markOutput("RTXDIPass.color")\n');

    const sceneSource = await (await fetch("/Falcor/media/Arcade/Arcade.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/Arcade");
    scene.camera.setAspectRatio(width / height);

    graph!.onResize(width, height);
    graph!.setScene(scene);
    const ctx = device.renderContext;

    let frame = 0;
    for (const captureFrame of [1, 16, 64]) {
        while (frame < captureFrame) {
            graph!.execute(ctx);
            frame++;
        }
        const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("RTXDIPass.color")!)).buffer);

        const res = await fetch(`/tests/oracle/out-native/oracle-imgtest-rtxdi-arcade.RTXDIPass.color.${captureFrame}.exr`);
        const { data, width: nw, height: nh } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        expectEq(nw, width, "oracle resolution");

        // ReSTIR light-sample sequences decorrelate between implementations
        // (the local-light PDF texture is R32Float on web vs R16Float native,
        // so RIS tile picks diverge and spatiotemporal reuse propagates the
        // divergence). Like the PT-Arcade test, gate on BIAS (signed mean)
        // plus 8x8 BLOCK AVERAGES (averaging cancels decorrelated noise);
        // fireflies near the x150 screen emitter rule out per-pixel gates.
        let signedSum = 0;
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
                            const w = Math.min(web[wi + c]!, 65504);
                            const n = Math.min(data[ni + c]!, 65504);
                            signedSum += w - n;
                            wSum += w;
                            nSum += n;
                        }
                    }
                }
                if (Math.abs(wSum - nSum) / (192 + Math.abs(nSum)) > 0.05) blockBad++;
            }
        }
        const mean = signedSum / (width * height * 3);
        console.error(`# oracle-imgtest-rtxdi-arcade frame ${captureFrame}: signed-mean=${mean.toExponential(2)} blockBad=${blockBad}/${bw * bh}`);
        // Observed: |bias| < 6e-4, blockBad <= 5 across all three frames.
        expectEq(Math.abs(mean) < 2e-3, true, `RTXDI Arcade frame ${captureFrame} bias ${mean}`);
        expectEq(blockBad <= 30, true, `RTXDI Arcade frame ${captureFrame} bad 8x8 blocks ${blockBad}`);
    }
});
