/**
 * M7 EXIT TEST — full PathTracer (GeneratePaths + TracePass megakernel: NEE,
 * MIS, LightBVH emissive sampler: CPU-built BVH with SAOH splits, cone
 * bounds and 64-bit traversal bitmasks) vs the native hardware-DXR PathTracer on the emissive two-quad scene.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-fullpt-lightbvh.py --headless
 */

import { GltfImporter, LightType, RenderGraph, createPass, float3, initScripting } from "@web-falcor/falcor";
import { runMogwaiScript, runMogwaiSource } from "../../../packages/mogwai/src/ScriptRunner.js";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("FullPathTracerLightBVH.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    const scene = await GltfImporter.importFromUrl(device, "/tests/oracle/assets/quad-emissive.gltf", [
        { type: LightType.Point, posW: new float3(0.5, 0.5, 1.5), intensity: new float3(3, 3, 3) },
    ]);
    expectEq(scene.useEmissiveLights, true, "emissive materials detected");
    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, "FullPTLightBVHGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, emissiveSampler: "LightBVH" }), "PathTracer");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.markOutput("PathTracer.color");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("PathTracer.color")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-fullpt-lightbvh.PathTracer.color.0.exr");
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
    console.error(`# fullPTLightBVH: meanAbs=${mean.toExponential(2)} rel=${rel.toExponential(2)} bad=${badPixels}`);
    expectEq(mean < 5e-3, true, `radiance mean abs diff ${mean}`);
    expectEq(badPixels < size * 4, true, `bad pixels ${badPixels}`);
});

gpuTest("FullPathTracerLightBVH.optionsMatchNativeOracle", async ({ device }) => {
    // NON-DEFAULT sampler + builder options (Equal splits, 1 tri/leaf, no
    // bounding cone, BoxToAverage bound) — same dict as the native oracle, so
    // traversal probabilities and the RNG stream must bit-match again.
    //
    // Regenerate the oracle with:
    //   xvfb-run -a Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-lightbvh-options.py --headless
    const size = 256;
    const scene = await GltfImporter.importFromUrl(device, "/tests/oracle/assets/quad-emissive.gltf", [
        { type: LightType.Point, posW: new float3(0.5, 0.5, 1.5), intensity: new float3(3, 3, 3) },
    ]);
    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, "FullPTLightBVHOptionsGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(
        createPass(device, "PathTracer", {
            samplesPerPixel: 1,
            emissiveSampler: "LightBVH",
            lightBVHOptions: {
                useBoundingCone: false,
                solidAngleBoundMethod: "BoxToAverage",
                buildOptions: { maxTriangleCountPerLeaf: 1, splitHeuristicSelection: "Equal", useLeafCreationCost: false },
            },
        }),
        "PathTracer",
    );
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.markOutput("PathTracer.color");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("PathTracer.color")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-lightbvh-options.PathTracer.color.0.exr");
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
    console.error(`# fullPTLightBVHOptions: meanAbs=${mean.toExponential(2)} rel=${rel.toExponential(2)} bad=${badPixels}`);
    expectEq(mean < 5e-3, true, `radiance mean abs diff ${mean}`);
    expectEq(badPixels < size * 4, true, `bad pixels ${badPixels}`);
});

gpuTest("FullPathTracerLightBVH.binnedSAHMatchesNativeOracle", async ({ device }) => {
    // tests/oracle/render-native-lightbvh-sah.py run unchanged: a BinnedSAH LightBVH over ~1.4k emissive
    // triangles. The same script with BinnedSAOH must differ more from native (the heuristic shapes the tree).
    //   Mogwai --script tests/oracle/render-native-lightbvh-sah.py --headless
    await initScripting("/node_modules/pyodide");
    const nat = parseExr(await (await fetch("/tests/oracle/out-native/oracle-lightbvh-sah.PathTracer.color.0.exr")).arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    const script = "/tests/oracle/render-native-lightbvh-sah.py";
    const source = await (await fetch(script)).text();
    const meanDiff = async (src: string) => {
        const { frameCapture } = await runMogwaiSource(device, src, "/tests/oracle", { fileName: "render-native-lightbvh-sah.py" });
        const exr = frameCapture.captured.find((f) => f.name.includes("PathTracer.color"))!;
        const web = parseExr(exr.bytes.slice().buffer, 1015) as { data: Float32Array };
        let sum = 0;
        for (let i = 0; i < nat.width * nat.height; i++) for (let c = 0; c < 3; c++) sum += Math.abs(web.data[i * 4 + c]! - nat.data[i * 4 + c]!);
        return sum / (nat.width * nat.height * 3);
    };
    const sah = await meanDiff(source);
    const saoh = await meanDiff(source.replace("'BinnedSAH'", "'BinnedSAOH'"));
    console.error(`# lightBVH BinnedSAH vs native: meanAbs ${sah.toExponential(2)}; with BinnedSAOH instead: ${saoh.toExponential(2)}`);
    expectEq(sah < 5e-3, true, `BinnedSAH radiance mean abs diff ${sah}`);
    expectEq(saoh > sah * 2, true, `BinnedSAOH differs more (${saoh} vs ${sah})`);
});

gpuTest("EmissiveUniformSampler.matchesNativeOracle", async ({ device }) => {
    // tests/oracle/render-native-emissive-uniform.py run unchanged: the Uniform sampler picks from the active
    // (flux > 0) triangles, so the spheres' degenerate pole triangles must be culled as natively.
    await initScripting("/node_modules/pyodide");
    const nat = parseExr(await (await fetch("/tests/oracle/out-native/oracle-emissive-uniform.PathTracer.color.0.exr")).arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    const { frameCapture } = await runMogwaiScript(device, "/tests/oracle/render-native-emissive-uniform.py");
    const web = parseExr(frameCapture.captured.find((f) => f.name.includes("PathTracer.color"))!.bytes.slice().buffer, 1015) as { data: Float32Array };
    let sum = 0;
    let differing = 0;
    for (let i = 0; i < nat.width * nat.height; i++) {
        let d = 0;
        for (let c = 0; c < 3; c++) {
            const x = Math.abs(web.data[i * 4 + c]! - nat.data[i * 4 + c]!);
            sum += x;
            d = Math.max(d, x);
        }
        if (d > 1e-3) differing++;
    }
    const mean = sum / (nat.width * nat.height * 3);
    console.error(`# uniform emissive sampler vs native: meanAbs ${mean.toExponential(2)}, differing pixels ${differing}`);
    expectEq(mean < 1e-4, true, `radiance mean abs diff ${mean}`);
    expectEq(differing < nat.width * nat.height * 0.002, true, `differing pixels ${differing}`);
});
