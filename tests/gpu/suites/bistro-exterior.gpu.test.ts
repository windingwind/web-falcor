/**
 * BistroExterior.pyscene — the media drop's 120 MB FBX with 405 textures, mostly 2048²
 * BC1/BC3/BC5 — against a native GBufferRT capture from the scene's own camera: the
 * Assimp import at native's flags, the scene build, and material textures sampled at
 * full resolution in their own BC formats (per-format/size texture arrays).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-bistro-exterior.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("BistroExterior.matchesNativeOracle", async ({ device }) => {
    const [w, h] = [320, 180];
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/media/Bistro_v5_2/BistroExterior.pyscene")).text();
    const t0 = performance.now();
    const scene = await runSceneScript(device, source, "/Falcor/media/Bistro_v5_2");
    const loadS = (performance.now() - t0) / 1000;
    scene.camera.setAspectRatio(w / h);

    const graph = new RenderGraph(device, "Bistro");
    graph.addPass(createPass(device, "GBufferRT", { useTraceRayInline: true, samplePattern: "Center" }), "GBufferRT");
    const channels = ["posW", "normW", "texC", "diffuseOpacity", "specRough"] as const;
    graph.markOutput("GBufferRT.mask");
    for (const c of channels) graph.markOutput(`GBufferRT.${c}`);
    graph.onResize(w, h);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    // Mogwai updates the (animated) scene at the clock's time before rendering a frame.
    scene.animate(0);
    graph.execute(ctx);

    const native = async (c: string) => (parseExr(await (await fetch(`/tests/oracle/out-native/bistro-exterior.GBufferRT.${c}.0.exr`)).arrayBuffer(), 1015) as { data: Float32Array }).data;
    const read = async (c: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(`GBufferRT.${c}`)!)).buffer);
    const [natMask, webMask] = [await native("mask"), await read("mask")];
    let hits = 0;
    let maskMismatch = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const nat = natMask[((h - 1 - y) * w + x) * 4] !== 0;
            if (nat) hits++;
            if (nat !== (webMask[y * w + x] !== 0)) maskMismatch++;
        }
    }
    // The FBX camera: native hits must reproject onto the web pixel centers.
    let meanOffset = 0;
    let medianOffset = 0;
    {
        const vp = scene.camera.getViewProjMatrix();
        const nat = await native("posW");
        let sx = 0, sy = 0, n = 0;
        const hist: number[] = [];
        for (let y = 0; y < h; y += 3) for (let x = 0; x < w; x += 3) {
            const ni = ((h - 1 - y) * w + x) * 4;
            if (natMask[ni] === 0) continue;
            const p = [nat[ni]!, nat[ni + 1]!, nat[ni + 2]!, 1];
            const r = [0, 1, 2, 3].map((i) => vp.get(i, 0) * p[0]! + vp.get(i, 1) * p[1]! + vp.get(i, 2) * p[2]! + vp.get(i, 3));
            const px = ((r[0]! / r[3]!) * 0.5 + 0.5) * w - (x + 0.5), py = (0.5 - (r[1]! / r[3]!) * 0.5) * h - (y + 0.5);
            if (Math.abs(px) > 3 || Math.abs(py) > 3) continue;
            sx += px; sy += py; n++; hist.push(Math.hypot(px, py));
        }
        hist.sort((a, b) => a - b);
        meanOffset = Math.hypot(sx / n, sy / n);
        medianOffset = hist[hist.length >> 1]!;
    }
    // Rays still land ~0.1 px apart (fp32 at Bistro's scale), aliasing Mip0 samples of 2048² textures
    // and foliage UVs; a pixel is bad only if no web pixel in its 3x3 neighbourhood agrees.
    const summary: string[] = [`load ${loadS.toFixed(1)}s`, `hits ${hits}`, `mask mismatches ${maskMismatch}`, `reprojection mean ${meanOffset.toFixed(4)}px median ${medianOffset.toFixed(4)}px`];
    const stats: Record<string, { mean: number; bad: number; badNeighbourhood: number }> = {};
    for (const c of channels) {
        const [nat, web] = [await native(c), await read(c)];
        const comps = web.length / (w * h);
        const tol = c === "posW" ? 1e-2 : c === "texC" ? 2e-3 : 2e-2;
        const diff = (wx: number, wy: number, ni: number) => {
            const wi = (wy * w + wx) * comps;
            let d = 0;
            for (let k = 0; k < (c === "texC" ? 2 : 3); k++) d = Math.max(d, Math.abs(web[wi + k]! - nat[ni + k]!) / (c === "posW" || c === "texC" ? Math.max(1, Math.abs(nat[ni + k]!)) : 1));
            return d;
        };
        let sum = 0;
        let n = 0;
        let bad = 0;
        let badNeighbourhood = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const ni = ((h - 1 - y) * w + x) * 4;
                if (natMask[ni] === 0 || webMask[y * w + x] === 0) continue;
                const d = diff(x, y, ni);
                sum += d;
                n++;
                if (d <= tol) continue;
                bad++;
                let best = d;
                for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
                    const [nx, ny] = [x + dx, y + dy];
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h && webMask[ny * w + nx] !== 0) best = Math.min(best, diff(nx, ny, ni));
                }
                if (best > tol) badNeighbourhood++;
            }
        }
        stats[c] = { mean: sum / Math.max(n, 1), bad, badNeighbourhood };
        summary.push(`${c} mean=${stats[c]!.mean.toExponential(2)} bad=${bad} (3x3 ${badNeighbourhood})`);
    }
    console.error(`# bistroExterior: ${summary.join(", ")}`);
    expectEq(hits > w * h * 0.5, true, `the scene is hit (${hits})`);
    expectEq(maskMismatch <= w * h * 0.002, true, `hit masks match (${maskMismatch})`);
    expectEq(meanOffset < 0.05 && medianOffset < 0.25, true, `camera reprojection ${meanOffset}/${medianOffset}px`);
    for (const [c, limit] of [["posW", 0.01], ["specRough", 0.01], ["normW", 0.05], ["diffuseOpacity", 0.05]] as const) {
        expectEq(stats[c]!.badNeighbourhood <= w * h * limit, true, `${c} bad in 3x3 ${stats[c]!.badNeighbourhood}`);
    }
    expectEq(stats.diffuseOpacity!.mean < 2e-2, true, `diffuse mean ${stats.diffuseOpacity!.mean}`);
});
