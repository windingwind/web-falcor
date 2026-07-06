/**
 * M7: delta reflection — perfect mirror (metallic=1, roughness=0) reflecting
 * colored geometry vs the native PathTracer.
 * Regenerate: Mogwai --script tests/oracle/render-native-mirror.py --headless
 */
import { RenderGraph, createPass, float3, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("MirrorDelta.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/oracle-mirror.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, "IsoMirrorGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, emissiveSampler: "Uniform" }), "PathTracer");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.markOutput("PathTracer.color");
    graph.setScene(scene);
    const ctx = device.renderContext;
    graph.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("PathTracer.color")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-mirror.PathTracer.color.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");
    let sum = 0;
    let badPixels = 0;
    const samples: string[] = [];
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
            if (pixelMax > 0.05) {
                badPixels++;
                if (samples.length < 4 && pixelMax > 0.2) samples.push(`(${x},${y}) web=(${web[wi]?.toFixed(2)},${web[wi+1]?.toFixed(2)},${web[wi+2]?.toFixed(2)}) nat=(${data[ni]?.toFixed(2)},${data[ni+1]?.toFixed(2)},${data[ni+2]?.toFixed(2)})`);
            }
        }
    }
    console.error(`# mirrorPT: meanAbs=${(sum / (size * size * 3)).toExponential(2)} bad=${badPixels}`);
    expectEq(sum / (size * size * 3) < 2e-3, true, "mean");
    expectEq(badPixels < 128, true, `bad pixels ${badPixels}`);
});
