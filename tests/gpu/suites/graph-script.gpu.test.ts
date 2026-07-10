/**
 * M4 exit test: the UNMODIFIED upstream graph script
 * tests/image_tests/renderpasses/graphs/ToneMapping.py executes in the browser
 * via Pyodide (falcor bridge), loads its .hdr asset, and renders.
 */

import { initScripting, runGraphScript, Properties } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("GraphScript.upstreamToneMappingPy", async ({ device }) => {
    // Fetch the untouched upstream script and run it.
    const scriptUrl = "/Falcor/tests/image_tests/renderpasses/graphs/ToneMapping.py";
    const source = await (await fetch(scriptUrl)).text();
    expectEq(source.includes("createPass(\"ToneMapper\")"), true, "fetched upstream script");

    await initScripting("/node_modules/pyodide");
    const graphs = await runGraphScript(device, source);
    expectEq(graphs.length, 1, "one graph registered via m.addGraph");
    const graph = graphs[0]!;
    expectEq(graph.name, "ToneMapper", "graph name from script");

    // Load assets (ImageLoader fetches the .hdr), compile + run.
    await graph.init();
    graph.onResize(256, 128);
    const ctx = device.renderContext;
    graph.execute(ctx);

    // Bisect: the loaded HDR must be non-zero after ImageLoader's blit.
    const loaded = graph.getOutput("ImageLoader.dst")!;
    const hdrPx = new Float32Array((await ctx.readTextureSubresource(loaded)).buffer);
    let hdrSum = 0;
    for (let i = 0; i < hdrPx.length; i += 4) hdrSum += hdrPx[i]!;
    expectEq(hdrSum > 0, true, `ImageLoader output non-zero (sum=${hdrSum.toFixed(2)})`);

    const out = graph.getOutput("BlitPass.dst");
    expectEq(out !== undefined, true, "marked output exists");

    // BlitPass.dst is a graph-default RGBA32Float target; the blit samples the
    // sRGB tone-mapped texture (decoding to linear). The env map is a bright sky:
    // average must be clearly non-zero and bounded.
    const px = new Float32Array((await ctx.readTextureSubresource(out!)).buffer);
    let sum = 0;
    let finite = true;
    for (let i = 0; i < px.length; i += 4) {
        sum += px[i]!;
        finite &&= Number.isFinite(px[i]!);
    }
    const avg = sum / (px.length / 4);
    expectEq(finite, true, "all values finite");
    expectEq(avg > 0.02 && avg <= 1.0, true, `plausible tonemapped luminance (avg=${avg.toFixed(4)})`);
});

gpuTest("GraphScript.toneMappingMatchesNativePng", async ({ device }) => {
    // OVERALL VERIFY: same unmodified graph, diffed against the native Mogwai
    // frame capture (PNG, top-down, sRGB bytes).
    // Regenerate: Mogwai --script tests/oracle/render-native-imgtest-tonemap.py --headless
    const size = 256;
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/ToneMapping.py")).text();
    await initScripting("/node_modules/pyodide");
    const [graph] = await runGraphScript(device, source);
    await graph!.init();
    graph!.onResize(size, size);
    const ctx = device.renderContext;
    graph!.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("BlitPass.dst")!)).buffer);

    const blob = await (await fetch("/tests/oracle/out-native/oracle-imgtest-tonemap.BlitPass.dst.0.png")).blob();
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
    // Heavy minification (2048x1024 HDR -> 256x256) makes bit-exact bilinear
    // phases unrealistic; adopt upstream ImageCompare policy (MSE threshold in
    // encoded sRGB) plus a mean-byte guard.
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
    console.error(`# imgtestTonemap: mse=${mse.toExponential(2)} maxByteDiff=${maxDiff}`);
    expectEq(mse < 2e-4, true, `sRGB MSE ${mse}`); // upstream image tests use 1e-4..1e-3 per-test
});

gpuTest("GraphScript.toneMappingAutoExposureMatchesNativePng", async ({ device }) => {
    // The test_ToneMapping.py 'autoExposure.True' variant: same graph with the
    // log-luminance mip-chain exposure enabled.
    // Regenerate: Mogwai --script tests/oracle/render-native-imgtest-tonemap-autoexp.py --headless
    const size = 256;
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/ToneMapping.py")).text();
    await initScripting("/node_modules/pyodide");
    const [graph] = await runGraphScript(device, source);
    graph!.getPass("ToneMapping")!.setProperties(new Properties({ autoExposure: true }));
    await graph!.init();
    graph!.onResize(size, size);
    const ctx = device.renderContext;
    graph!.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("BlitPass.dst")!)).buffer);

    const blob = await (await fetch("/tests/oracle/out-native/oracle-imgtest-tonemap-autoexp.BlitPass.dst.0.png")).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    expectEq(bitmap.width, size, "oracle resolution");
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
    let signedSum = 0;
    for (let i = 0; i < size * size; i++) {
        for (let ch = 0; ch < 3; ch++) {
            const d = (toSrgbByte(web[i * 4 + ch]!) - nat[i * 4 + ch]!) / 255;
            mse += d * d;
            signedSum += d;
            maxDiff = Math.max(maxDiff, Math.abs(d) * 255);
        }
    }
    mse /= size * size * 3;
    const meanSigned = (signedSum / (size * size * 3)) * 255;
    console.error(`# imgtestTonemapAutoExp: mse=${mse.toExponential(2)} maxByteDiff=${maxDiff} meanSignedByte=${meanSigned.toFixed(3)}`);
    // The avg-luminance divisor couples the base test's minification residual
    // into a global exposure scale, so the gate is looser than the manual test.
    expectEq(mse < 4e-4, true, `sRGB MSE ${mse}`);
    expectEq(Math.abs(meanSigned) < 1.0, true, `no global exposure bias (meanSignedByte=${meanSigned.toFixed(3)})`);
});
