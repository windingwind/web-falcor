/**
 * OVERALL VERIFY — upstream image tests for the small utility passes: the
 * UNMODIFIED CompositePass.py / CrossFadePass.py / GaussianBlur.py graphs
 * (scene-less ImageLoader chains) diffed per-pixel against native EXR captures.
 *
 * Regenerate oracles with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-<name>.py --headless
 *
 * Tolerances: PNG/HDR sources decode losslessly (tight); the jpg input
 * (sorsele3/posz.jpg) legitimately differs by a few LSB between the browser's
 * and FreeImage's IDCT, so jpg-fed graphs get a looser bad-pixel gate.
 */

import { initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

async function runUpstreamGraph(device: import("@web-falcor/falcor").Device, graphFile: string, output: string): Promise<Float32Array> {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch(`/Falcor/tests/image_tests/renderpasses/graphs/${graphFile}`)).text();
    const [graph] = await runGraphScript(device, source);
    await graph!.init();
    graph!.onResize(size, size);
    const ctx = device.renderContext;
    graph!.execute(ctx);
    return new Float32Array((await ctx.readTextureSubresource(graph!.getOutput(output)!)).buffer);
}

async function compareToOracle(web: Float32Array, oracle: string, meanTol: number, badPixelTol: number, badPixelThreshold = 1e-3): Promise<void> {
    const res = await fetch(`/tests/oracle/out-native/${oracle}`);
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    const webComponents = web.length / (size * size);
    let sum = 0;
    let badPixels = 0;
    let maxDiff = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const wi = (y * size + x) * webComponents;
            const ni = ((height - 1 - y) * width + x) * 4; // EXR rows are bottom-up
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(web[wi + c]! - data[ni + c]!);
                sum += d;
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > badPixelThreshold) badPixels++;
            maxDiff = Math.max(maxDiff, pixelMax);
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# ${oracle}: mean=${mean.toExponential(2)} bad=${badPixels} max=${maxDiff.toExponential(2)}`);
    expectEq(mean < meanTol, true, `mean ${mean}`);
    expectEq(badPixels <= badPixelTol, true, `bad pixels ${badPixels}`);
}

gpuTest("ImageTestComposite.matchesNativeOracle", async ({ device }) => {
    const web = await runUpstreamGraph(device, "CompositePass.py", "Composite.out");
    // Residual is entirely input A's jpg decode (browser vs FreeImage IDCT /
    // chroma upsampling): the png-fed pixels contribute zero error (identical
    // stats to the CrossFade test, which drops input B). Threshold 0.02 linear
    // ≈ 2-3 sRGB LSB.
    await compareToOracle(web, "oracle-imgtest-composite.Composite.out.0.exr", 5e-3, 700, 0.02);
});

gpuTest("ImageTestCrossFade.matchesNativeOracle", async ({ device }) => {
    const web = await runUpstreamGraph(device, "CrossFadePass.py", "CrossFade.out");
    // Frame 0 auto-fade mix=0 -> out = A (jpg); same decoder residual as above.
    await compareToOracle(web, "oracle-imgtest-crossfade.CrossFade.out.0.exr", 5e-3, 700, 0.02);
});

gpuTest("ImageTestGaussianBlur.matchesNativeOracle", async ({ device }) => {
    const web = await runUpstreamGraph(device, "GaussianBlur.py", "GaussianBlur.dst");
    // hdr input decodes bit-exactly; separable blur is pure float math.
    await compareToOracle(web, "oracle-imgtest-gaussianblur.GaussianBlur.dst.0.exr", 1e-4, 50);
});
