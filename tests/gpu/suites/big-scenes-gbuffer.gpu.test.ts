/**
 * The production scenes besides BistroExterior (bistro-exterior.gpu.test.ts) through GBufferRT from each scene's own
 * camera at t = 0, against native captures. Opt-in (each loads for minutes): EXTRA_QUERY=bigScenes=1.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-big-gbuffer.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq, saveArtifact, SkipError } from "../harness/registry.js";

const kBigScenes = new URLSearchParams(location.search).get("bigScenes") === "1";
const kScenes: Record<string, string> = {
    BistroInterior_Wine: "Bistro_v5_2/BistroInterior_Wine.pyscene",
    EmeraldSquare_Day: "EmeraldSquare_v4_1/EmeraldSquare_Day.pyscene",
    SunTemple: "SunTemple_v4/SunTemple/SunTemple.pyscene",
    ZeroDay: "ZeroDay_v1/ZeroDay.pyscene",
};
const kChannels = ["posW", "normW", "texC", "diffuseOpacity", "specRough", "emissive"] as const;

for (const [name, path] of Object.entries(kScenes)) {
    gpuTest(`BigScenes.gbufferMatchesNative.${name}`, async ({ device }) => {
        if (!kBigScenes) throw new SkipError("run with EXTRA_QUERY=bigScenes=1");
        const [w, h] = [320, 180];
        await initScripting("/node_modules/pyodide");
        const dir = `/Falcor/media/${path.slice(0, path.lastIndexOf("/"))}`;
        const scene = await runSceneScript(device, await (await fetch(`/Falcor/media/${path}`)).text(), dir);
        scene.camera.setAspectRatio(w / h);
        const graph = new RenderGraph(device, name);
        graph.addPass(createPass(device, "GBufferRT", { useTraceRayInline: true, samplePattern: "Center" }), "GBufferRT");
        graph.markOutput("GBufferRT.mask");
        for (const c of kChannels) graph.markOutput(`GBufferRT.${c}`);
        graph.onResize(w, h);
        graph.setScene(scene);
        await graph.init();
        const ctx = device.renderContext;
        scene.animate(0);
        graph.execute(ctx);

        const native = async (c: string) => (parseExr(await (await fetch(`/tests/oracle/out-native/big-gbuffer/${name}.GBufferRT.${c}.0.exr`)).arrayBuffer(), 1015) as { data: Float32Array }).data;
        const read = async (c: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(`GBufferRT.${c}`)!)).buffer);
        const [natMask, webMask] = [await native("mask"), await read("mask")];
        let hits = 0;
        let maskMismatch = 0;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const nat = natMask[((h - 1 - y) * w + x) * 4] !== 0;
            if (nat) hits++;
            if (nat !== (webMask[y * w + x] !== 0)) maskMismatch++;
        }
        // Native hits reprojected through the web camera land on the web pixel centers.
        let meanOffset = 0, medianOffset = 0;
        {
            const vp = scene.camera.getViewProjMatrix();
            const nat = await native("posW");
            let sx = 0, sy = 0, n = 0;
            const hist: number[] = [];
            for (let y = 0; y < h; y += 3) for (let x = 0; x < w; x += 3) {
                const ni = ((h - 1 - y) * w + x) * 4;
                if (natMask[ni] === 0) continue;
                const r = [0, 1, 2, 3].map((i) => vp.get(i, 0) * nat[ni]! + vp.get(i, 1) * nat[ni + 1]! + vp.get(i, 2) * nat[ni + 2]! + vp.get(i, 3));
                const px = ((r[0]! / r[3]!) * 0.5 + 0.5) * w - (x + 0.5), py = (0.5 - (r[1]! / r[3]!) * 0.5) * h - (y + 0.5);
                if (Math.abs(px) > 3 || Math.abs(py) > 3) continue;
                sx += px; sy += py; n++; hist.push(Math.hypot(px, py));
            }
            hist.sort((a, b) => a - b);
            meanOffset = Math.hypot(sx / n, sy / n);
            medianOffset = hist[hist.length >> 1]!;
        }
        // A pixel is bad only if no web pixel in its 3x3 neighbourhood agrees (rays land ~0.1 px apart in fp32).
        const summary: string[] = [`hits ${hits}`, `mask mismatches ${maskMismatch}`, `reprojection mean ${meanOffset.toFixed(4)}px median ${medianOffset.toFixed(4)}px`];
        const stats: Record<string, { mean: number; badNeighbourhood: number }> = {};
        for (const c of kChannels) {
            const [nat, web] = [await native(c), await read(c)];
            const comps = web.length / (w * h);
            const relative = c === "posW" || c === "texC";
            const tol = c === "posW" ? 1e-2 : c === "texC" ? 2e-3 : 2e-2;
            const diff = (wx: number, wy: number, ni: number) => {
                const wi = (wy * w + wx) * comps;
                let d = 0;
                for (let k = 0; k < (c === "texC" ? 2 : 3); k++) d = Math.max(d, Math.abs(web[wi + k]! - nat[ni + k]!) / (relative ? Math.max(1, Math.abs(nat[ni + k]!)) : 1));
                return d;
            };
            let sum = 0, n = 0, badNeighbourhood = 0;
            const badMap = new Uint8Array(w * h * 4).fill(255);
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
                const ni = ((h - 1 - y) * w + x) * 4;
                if (natMask[ni] === 0 || webMask[y * w + x] === 0) continue;
                const d = diff(x, y, ni);
                sum += d;
                n++;
                if (d <= tol) continue;
                let best = d;
                for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
                    const [nx, ny] = [x + dx, y + dy];
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h && webMask[ny * w + nx] !== 0) best = Math.min(best, diff(nx, ny, ni));
                }
                if (best > tol) (badNeighbourhood++, badMap.set([255, 0, 0, 255], (y * w + x) * 4));
            }
            if (c === "posW") await saveArtifact(`big-gbuffer-${name}-posW-bad`, badMap, w, h, false);
            if (c === "diffuseOpacity") {
                const toByte = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) ** (1 / 2.2) * 255);
                await saveArtifact(`big-gbuffer-${name}-diffuse-web`, Array.from({ length: w * h * 4 }, (_x, i) => (i % 4 === 3 ? 255 : toByte(web[i]!))), w, h, false);
                await saveArtifact(`big-gbuffer-${name}-diffuse-native`, Array.from({ length: w * h * 4 }, (_x, i) => {
                    const [px, c4] = [i >> 2, i & 3];
                    const [x, y] = [px % w, Math.floor(px / w)];
                    return c4 === 3 ? 255 : toByte(nat[((h - 1 - y) * w + x) * 4 + c4]!);
                }), w, h, false);
            }
            stats[c] = { mean: sum / Math.max(n, 1), badNeighbourhood };
            summary.push(`${c} mean=${stats[c]!.mean.toExponential(2)} bad3x3=${badNeighbourhood}`);
        }
        console.error(`# big-gbuffer ${name}: ${summary.join(", ")}`);
        scene.destroy();
        expectEq(hits > w * h * 0.3, true, `the scene is hit (${hits})`);
        expectEq(maskMismatch <= w * h * 0.002, true, `hit masks match (${maskMismatch})`);
        expectEq(meanOffset < 0.02 && medianOffset < 0.25, true, `camera reprojection ${meanOffset}/${medianOffset}px`);
        for (const [c, limit] of [["posW", 0.01], ["specRough", 0.01], ["normW", 0.05], ["diffuseOpacity", 0.05], ["texC", 0.02], ["emissive", 0.01]] as const) {
            expectEq(stats[c]!.badNeighbourhood <= w * h * limit, true, `${c} bad in 3x3 ${stats[c]!.badNeighbourhood}`);
        }
    });
}
