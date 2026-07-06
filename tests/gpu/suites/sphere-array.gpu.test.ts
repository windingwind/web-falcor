/**
 * OVERALL VERIFY: the UNMODIFIED upstream sphere_array.pyscene — 64 spheres
 * (131k triangles), roughness/metallic grid, HDR environment light with
 * importance sampling — through the Pyodide bridge and full PathTracer.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-spherearray.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("SphereArray.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/media/test_scenes/sphere_array.pyscene")).text();
    const scene = await runSceneScript(device, source, "/Falcor/media/test_scenes");

    // Camera comes from the pyscene itself; only the aspect mirrors the 256^2 framebuffer.
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, "SphereArrayGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, emissiveSampler: "Uniform" }), "PathTracer");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.markOutput("PathTracer.color");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("PathTracer.color")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-spherearray.PathTracer.color.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let refSum = 0;
    let badPixels = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const webIdx = (y * size + x) * 4;
            const natIdx = ((height - 1 - y) * width + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(web[webIdx + c]! - data[natIdx + c]!);
                sum += d;
                refSum += Math.abs(data[natIdx + c]!);
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 0.05) badPixels++;
        }
    }
    const mean = sum / (size * size * 3);
    const rel = sum / Math.max(refSum, 1e-6);
    console.error(`# sphereArrayPT: meanAbs=${mean.toExponential(2)} rel=${rel.toExponential(2)} bad=${badPixels}`);
    expectEq(mean < 5e-3, true, `radiance mean abs diff ${mean}`);
    expectEq(badPixels < size * 8, true, `bad pixels ${badPixels}`);
});
