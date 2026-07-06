/**
 * FEATURE VERIFY — SVGF denoiser vs native. Mirrors the upstream SVGF.py
 * image test with GBufferRT in place of ROV-blocked GBufferRaster and mvec
 * standing in for the raster-only pnFwidth channel (see
 * tests/oracle/graphs/svgf-feature.py for the equivalence argument).
 * 4 frames over sphere_array (cornell's axis-aligned walls hit normalize(0)
 * UB in GBufferRT's linearZ-derivative helper on half the image — native
 * itself writes z=Inf there; on spheres the degenerate set is ~a point per
 * sphere): temporal reprojection accumulates history across the path
 * tracer's per-frame noise; the a-trous chain filters spatially.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-feature-svgf.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

function halfToFloat(h: number): number {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

gpuTest("FeatureSVGF.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/graphs/svgf-feature.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/sphere_array.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    for (const out of ["PathTracer.color", "GBufferRT.linearZ", "GBufferRT.guideNormalW", "GBufferRT.emissive"]) graph!.markOutput(out);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 4; f++) graph!.execute(ctx);

    // Input bisection vs native EXRs (frame 4 state).
    for (const [ref, oracle, comps] of [
        ["PathTracer.color", "oracle-feature-svgf.PathTracer.color.0.exr", 3],
        ["GBufferRT.linearZ", "oracle-feature-svgf.GBufferRT.linearZ.0.exr", 2],
        ["GBufferRT.guideNormalW", "oracle-feature-svgf.GBufferRT.guideNormalW.0.exr", 3],
        ["GBufferRT.emissive", "oracle-feature-svgf.GBufferRT.emissive.0.exr", 3],
    ] as const) {
        const tex = graph!.getOutput(ref);
        if (!tex) { console.error(`# input ${ref}: NOT ALLOCATED`); continue; }
        const webIn = new Float32Array((await ctx.readTextureSubresource(tex)).buffer);
        const r2 = await fetch(encodeURI(`/tests/oracle/out-native/${oracle}`));
        const exr = parseExr(await r2.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        const webComps = webIn.length / (size * size);
        let s2 = 0;
        let bad2 = 0;
        let mx = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * webComps;
                const ni = ((exr.height - 1 - y) * exr.width + x) * 4;
                let pm = 0;
                for (let c = 0; c < comps; c++) {
                    const w = Math.min(webIn[wi + c]!, 65504);
                    const n = Math.min(exr.data[ni + c]!, 65504);
                    const d = Math.abs(w - n);
                    s2 += d;
                    pm = Math.max(pm, d);
                }
                if (pm > 0.05) bad2++;
                mx = Math.max(mx, pm);
            }
        }
        console.error(`# input ${ref}: mean=${(s2 / (size * size * comps)).toExponential(2)} bad@0.05=${bad2} max=${mx.toExponential(2)}`);
    }

    const raw = new Uint16Array((await ctx.readTextureSubresource(graph!.getOutput("SVGFPass.Filtered image")!)).buffer);
    const web = Float32Array.from(raw, halfToFloat);

    const res = await fetch(encodeURI("/tests/oracle/out-native/oracle-feature-svgf.SVGFPass.Filtered image.0.exr"));
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let bad = 0;
    let maxD = 0;
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
            maxD = Math.max(maxD, pixelMax);
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# oracle-feature-svgf: mean=${mean.toExponential(2)} bad@0.05=${bad} max=${maxD.toExponential(2)}`);
    expectEq(mean < 1e-3, true, `SVGF mean ${mean}`);
    expectEq(bad <= 300, true, `SVGF bad pixels ${bad}`);
});
