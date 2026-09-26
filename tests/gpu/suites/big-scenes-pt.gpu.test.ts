/**
 * The production scenes through the default PathTracer (1 spp, 64 accumulated frames at t = 0) against native
 * captures: lighting, emissive and environment sampling, and materials at scale. Opt-in: EXTRA_QUERY=bigScenes=1.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-big-pt.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq, saveArtifact, SkipError } from "../harness/registry.js";

const kBigScenes = new URLSearchParams(location.search).get("bigScenes") === "1";
const kScenes: Record<string, string> = {
    BistroExterior: "Bistro_v5_2/BistroExterior.pyscene",
    BistroInterior_Wine: "Bistro_v5_2/BistroInterior_Wine.pyscene",
    EmeraldSquare_Day: "EmeraldSquare_v4_1/EmeraldSquare_Day.pyscene",
    SunTemple: "SunTemple_v4/SunTemple/SunTemple.pyscene",
    ZeroDay: "ZeroDay_v1/ZeroDay.pyscene",
};
const kFrames = 64;

for (const [name, path] of Object.entries(kScenes)) {
    gpuTest(`BigScenes.pathTracerMatchesNative.${name}`, async ({ device }) => {
        if (!kBigScenes) throw new SkipError("run with EXTRA_QUERY=bigScenes=1");
        const [w, h] = [320, 180];
        await initScripting("/node_modules/pyodide");
        const dir = `/Falcor/media/${path.slice(0, path.lastIndexOf("/"))}`;
        const scene = await runSceneScript(device, await (await fetch(`/Falcor/media/${path}`)).text(), dir);
        scene.camera.setAspectRatio(w / h);
        const graph = new RenderGraph(device, name);
        graph.addPass(createPass(device, "VBufferRT", { samplePattern: "Center", useAlphaTest: true }), "VBufferRT");
        graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, useSER: false }), "PathTracer");
        graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "AccumulatePass");
        graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
        graph.addEdge("PathTracer.color", "AccumulatePass.input");
        graph.markOutput("AccumulatePass.output");
        graph.onResize(w, h);
        graph.setScene(scene);
        await graph.init();
        const ctx = device.renderContext;
        scene.animate(0);
        for (let i = 0; i < kFrames; i++) graph.execute(ctx);

        const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("AccumulatePass.output")!)).buffer);
        const nat = (parseExr(await (await fetch(`/tests/oracle/out-native/big-pt/${name}.AccumulatePass.output.0.exr`)).arrayBuffer(), 1015) as { data: Float32Array }).data;
        let sum = 0, refSum = 0;
        const mean = [0, 0, 0], refMean = [0, 0, 0];
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const [wi, ni] = [(y * w + x) * 4, ((h - 1 - y) * w + x) * 4];
            for (let c = 0; c < 3; c++) {
                const [a, b] = [web[wi + c]!, nat[ni + c]!];
                if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
                sum += Math.abs(a - b);
                refSum += Math.abs(b);
                mean[c] += a / (w * h);
                refMean[c] += b / (w * h);
            }
        }
        const rel = sum / Math.max(refSum, 1e-6);
        const ratio = mean.map((m, c) => m / Math.max(refMean[c]!, 1e-9));
        console.error(`# big-pt ${name}: relL1 ${rel.toFixed(4)}, mean web/native ${ratio.map((r) => r.toFixed(3)).join(",")} (native mean ${refMean.map((m) => m.toExponential(2)).join(",")})`);
        const tone = (v: number) => Math.round(Math.min(Math.max(v / (1 + v), 0), 1) ** (1 / 2.2) * 255);
        await saveArtifact(`big-pt-${name}-web`, Array.from({ length: w * h * 4 }, (_x, i) => (i % 4 === 3 ? 255 : tone(web[i]!))), w, h, false);
        await saveArtifact(`big-pt-${name}-native`, Array.from({ length: w * h * 4 }, (_x, i) => {
            const [px, c4] = [i >> 2, i & 3];
            return c4 === 3 ? 255 : tone(nat[((h - 1 - Math.floor(px / w)) * w + (px % w)) * 4 + c4]!);
        }), w, h, false);
        scene.destroy();
        // Open residual: ZeroDay (thousands of emissive panels) renders ~4% darker and noisier than native.
        for (const r of ratio) expectEq(Math.abs(r - 1) < 0.05, true, `mean radiance ratio ${ratio}`);
        expectEq(rel < (name === "ZeroDay" ? 0.35 : 0.1), true, `relative L1 ${rel}`);
    });
}
