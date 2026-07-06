/**
 * Volume-absorption isolation: unit cube, IoR=1 (no refraction/Fresnel),
 * specular transmission 1 -> center transmittance = exp(-sigma * 1.0).
 * Diffed per-pixel against the native oracle.
 *
 * Regenerate: Mogwai --script tests/oracle/render-native-absorb.py --headless
 */

import { RenderGraph, createPass, runSceneScript, initScripting } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Absorb.matchesNativeOracle", async ({ device }) => {
    const size = 128;
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/tests/oracle/assets/oracle-absorb.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/tests/oracle/assets");
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, "Absorb");
    graph.addPass(createPass(device, "VBufferRT", { samplePattern: "Center" }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1 }), "PathTracer");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.markOutput("PathTracer.color");
    graph.onResize(size, size);
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("PathTracer.color")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-absorb.PathTracer.color.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    const webPx = (x: number, y: number) => [web[(y * size + x) * 4]!, web[(y * size + x) * 4 + 1]!, web[(y * size + x) * 4 + 2]!];
    const natPx = (x: number, y: number) => {
        const i = ((height - 1 - y) * width + x) * 4;
        return [data[i]!, data[i + 1]!, data[i + 2]!];
    };
    console.error(`# absorb center web=${webPx(64, 64).map((v) => v.toFixed(5))} nat=${natPx(64, 64).map((v) => v.toFixed(5))}`);
    console.error(`# absorb bg web=${webPx(5, 5).map((v) => v.toFixed(5))} nat=${natPx(5, 5).map((v) => v.toFixed(5))}`);

    let sum = 0;
    let bad = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const w = webPx(x, y);
            const n = natPx(x, y);
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(w[c]! - n[c]!);
                sum += d;
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 1e-3) bad++;
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# absorb: mean=${mean.toExponential(2)} bad=${bad}`);
    expectEq(mean < 1e-3, true, `mean ${mean}`);
    expectEq(bad <= 150, true, `bad ${bad}`);
});
