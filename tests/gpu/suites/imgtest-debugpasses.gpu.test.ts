/**
 * OVERALL VERIFY — upstream image tests for the DebugPasses family: the
 * UNMODIFIED SideBySide.py / SplitScreen.py / ColorMapPass.py graphs
 * (scene-less ImageLoader chains) diffed against native PNG captures.
 *
 * Native allocates the graph-default (swapchain sRGB) format for these
 * outputs and captures PNG; the web allocator defaults to RGBA32Float, so the
 * comparison re-encodes web linear floats to sRGB bytes (same policy as the
 * ToneMapping image test; MSE threshold per upstream ImageCompare).
 *
 * Regenerate oracles with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-<name>.py --headless
 */

import { initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
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

async function compareToPngOracle(web: Float32Array, oracle: string, mseTol: number): Promise<void> {
    const blob = await (await fetch(`/tests/oracle/out-native/${oracle}`)).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    expectEq(bitmap.width, size, "oracle resolution");
    const canvas = new OffscreenCanvas(size, size);
    const c2d = canvas.getContext("2d", { willReadFrequently: true })!;
    c2d.drawImage(bitmap, 0, 0);
    const nat = c2d.getImageData(0, 0, size, size).data; // sRGB bytes, top-down

    const toSrgbByte = (v: number) => {
        const c = Math.min(Math.max(v, 0), 1);
        const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
        return Math.round(s * 255);
    };
    let mse = 0;
    let maxDiff = 0;
    for (let i = 0; i < size * size; i++) {
        for (let ch = 0; ch < 3; ch++) {
            const d = (toSrgbByte(web[i * 4 + ch]!) - nat[i * 4 + ch]!) / 255;
            mse += d * d;
            maxDiff = Math.max(maxDiff, Math.abs(d) * 255);
        }
    }
    mse /= size * size * 3;
    console.error(`# ${oracle}: mse=${mse.toExponential(2)} maxByteDiff=${maxDiff}`);
    expectEq(mse < mseTol, true, `sRGB MSE ${mse}`);
}

gpuTest("ImageTestSideBySide.matchesNativeOracle", async ({ device }) => {
    const web = await runUpstreamGraph(device, "SideBySide.py", "SideBySidePass.output");
    // Both inputs are the jpg (raw + sRGB): decoder residual dominates.
    await compareToPngOracle(web, "oracle-imgtest-sidebyside.SideBySidePass.output.0.png", 5e-4);
});

gpuTest("ImageTestSplitScreen.matchesNativeOracle", async ({ device }) => {
    const web = await runUpstreamGraph(device, "SplitScreen.py", "SplitScreenPass.output");
    await compareToPngOracle(web, "oracle-imgtest-splitscreen.SplitScreenPass.output.0.png", 5e-4);
});

gpuTest("ImageTestColorMap.matchesNativeOracle", async ({ device }) => {
    // hdr input; frame 0 renders with the static [0,1] range (the auto-range
    // reduction result is consumed one frame later, like native).
    const web = await runUpstreamGraph(device, "ColorMapPass.py", "ColorMap.output");
    await compareToPngOracle(web, "oracle-imgtest-colormap.ColorMap.output.0.png", 2e-4);
});
