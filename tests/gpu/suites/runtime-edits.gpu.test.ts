/**
 * FEATURE VERIFY — runtime scene edits: after the scene loads, light
 * intensity and material baseColor/roughness are changed through the Scene
 * API (getLight/updateLights, getMaterial/updateMaterial), then one frame
 * renders through the upstream MinimalPathTracer.py graph and is diffed
 * against native Mogwai applying the same edits via python.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-runtime-edits.py --headless
 */

import { initScripting, runGraphScript, runSceneScript, float3, float4 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("RuntimeEdits.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const sceneSource = await (await fetch("/tests/oracle/assets/oracle-dof.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/tests/oracle/assets");
    scene.camera.setAspectRatio(1.0);

    // Runtime edits (same values as the native oracle script).
    const light = scene.getLight(0);
    light.intensity = new float3(30.0, 18.0, 6.0);
    scene.updateLights();
    const mat = scene.getMaterial("Mid");
    mat.basic.baseColor = new float4(0.9, 0.4, 0.1, 1.0);
    const spec = mat.basic.specular ?? new float4(0, 0, 0, 0);
    mat.basic.specular = new float4(spec.x, 0.1, spec.z, spec.w); // roughness = G
    scene.updateMaterial(mat);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    const res = await fetch("/tests/oracle/out-native/oracle-runtime-edits.ToneMapper.dst.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let bad = 0;
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
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# runtimeEdits: mean=${mean.toExponential(2)} bad=${bad}`);
    expectEq(mean < 1e-3, true, `mean ${mean}`);
    expectEq(bad <= 300, true, `bad pixels ${bad}`);
});

gpuTest("RuntimeEdits.emissiveRebuildMatchesNative", async ({ device }) => {
    // Emissive edit after load rebuilds the NEE flux tables: cornell light
    // dimmed to emissiveFactor 0.4, PathTracer 64 frames vs native.
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/PathTracer.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    const mat = scene.getMaterial("Light");
    mat.basic.emissiveFactor = 0.4;
    scene.updateMaterial(mat);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 64; f++) graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("PathTracer.color")!)).buffer);
    const res = await fetch("/tests/oracle/out-native/oracle-emissive-edit.PathTracer.color.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let signed = 0;
    let natSum = 0;
    const kBlock = 8;
    let badBlocks = 0;
    for (let by = 0; by < size / kBlock; by++) {
        for (let bx = 0; bx < size / kBlock; bx++) {
            let w = 0;
            let n = 0;
            for (let y = by * kBlock; y < (by + 1) * kBlock; y++) {
                for (let x = bx * kBlock; x < (bx + 1) * kBlock; x++) {
                    const wi = (y * size + x) * 4;
                    const ni = ((height - 1 - y) * width + x) * 4;
                    for (let c = 0; c < 3; c++) {
                        w += Math.min(web[wi + c]!, 16);
                        n += Math.min(data[ni + c]!, 16);
                        signed += Math.min(web[wi + c]!, 16) - Math.min(data[ni + c]!, 16);
                        natSum += Math.min(data[ni + c]!, 16);
                    }
                }
            }
            if (Math.abs(w - n) / Math.max(n, 0.5) > 0.05) badBlocks++;
        }
    }
    const bias = signed / Math.max(natSum, 1e-6);
    console.error(`# emissiveEdit: bias=${bias.toExponential(2)} badBlocks=${badBlocks}/1024`);
    expectEq(Math.abs(bias) < 1e-2, true, `signed bias ${bias}`);
    expectEq(badBlocks <= 60, true, `bad blocks ${badBlocks}`);
});
