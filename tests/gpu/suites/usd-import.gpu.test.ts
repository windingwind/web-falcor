/**
 * FEATURE VERIFY — USD import (tinyusdz): oracle-usd.usda (meshes + xform
 * hierarchy + UsdPreviewSurface materials) imported on the web and rendered
 * through the upstream MinimalPathTracer.py graph vs the native USDImporter
 * capture. The camera and SphereLight are hand-set from the .usda values —
 * the tinyusdz RenderScene API does not expose lights/cameras yet (docs §8.4).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-usd.py --headless
 *   (needs build/.../plugins/USDImporter.so — re-enable from plugins-disabled/)
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq, saveArtifact } from "../harness/registry.js";

// Camera/light mirror the .usda prims (not yet exposed by tinyusdz's
// RenderScene API; values match the native import — probed).
const kScene = `
sceneBuilder.importScene('oracle-usd.usda')

camera = Camera()
camera.position = float3(3.5, 2.5, 4.5)
camera.target = float3(2.923902, 2.154341, 3.759303)
camera.up = float3(-0.212214, 0.938360, -0.272846)
camera.focalLength = 35.0
sceneBuilder.addCamera(camera)

light = SphereLight('Light')
light.intensity = float3(15.0, 15.0, 15.0)
light.scaling = 0.3
light.position = float3(1.5, 2.5, 2.0)
sceneBuilder.addLight(light)
`;

gpuTest("UsdImport.matchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");

    // Deterministic geometry/camera parity first: VBufferRT depth vs native.
    {
        const vbufferSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/VBufferRT.py")).text();
        const [vgraph] = await runGraphScript(device, vbufferSource);
        const vscene = await runSceneScript(device, kScene, "/tests/oracle/assets");
        vscene.camera.setAspectRatio(1.0);
        vscene.camera.setDepthRange(1, 1000000); // USD clippingRange default (native probe)
        vscene.camera.setFrameHeight(15.2908); // USD vertical-aperture default (native probe)
        vgraph!.onResize(size, size);
        vgraph!.setScene(vscene);
        console.error(`# usdImport.bounds: ${JSON.stringify(vscene.worldBounds)}`);
        const vctx = device.renderContext;
        vgraph!.execute(vctx);
        // Depth is unusable as an oracle channel here: the USD clippingRange
        // default (1..1e6) pushes NDC depth against 1.0 where the fp16 EXR
        // capture cannot resolve it. mask = geometry parity, viewW = camera.
        for (const [channel, components, tol, badTol] of [
            ["mask", 1, 1e-4, 60],
            ["viewW", 3, 1e-3, 60],
        ] as const) {
            const tex = vgraph!.getOutput(`VBufferRT.${channel}`)!;
            const web = new Float32Array((await vctx.readTextureSubresource(tex)).buffer);
            const webComponents = web.length / (size * size);
            const res = await fetch(`/tests/oracle/out-native/oracle-usd-vbuffer.VBufferRT.${channel}.0.exr`);
            const nat = (parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array }).data;
            let dSum = 0;
            let dBad = 0;
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const wi = (y * size + x) * webComponents;
                    const ni = ((size - 1 - y) * size + x) * 4;
                    let pixelMax = 0;
                    for (let c = 0; c < components; c++) {
                        const d = Math.abs(web[wi + c]! - nat[ni + c]!);
                        dSum += d;
                        pixelMax = Math.max(pixelMax, d);
                    }
                    if (pixelMax > 1e-3) dBad++;
                }
            }
            const mean = dSum / (size * size * components);
            console.error(`# usdImport.${channel}: mean=${mean.toExponential(2)} bad=${dBad}`);
            expectEq(mean < tol, true, `${channel} mean ${mean}`);
            expectEq(dBad <= badTol, true, `${channel} bad ${dBad}`);
        }
    }
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const scene = await runSceneScript(device, kScene, "/tests/oracle/assets");
    scene.camera.setAspectRatio(1.0);
    scene.camera.setDepthRange(1, 1000000);
    scene.camera.setFrameHeight(15.2908);
    expectEq(scene.stats.instances >= 2, true, `meshes imported (${scene.stats.instances})`);
    expectEq(scene.getMaterial(0) !== undefined, true, "materials imported");

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 64; f++) graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    const res = await fetch("/tests/oracle/out-native/oracle-usd.ToneMapper.dst.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    // Stochastic sphere-light NEE decorrelates across implementations (the
    // PT-class policy): gate on signed mean bias + 8x8 block averages.
    let signed = 0;
    let natSum = 0;
    const kBlock = 8;
    const blocks = size / kBlock;
    let badBlocks = 0;
    let webLit = 0;
    let natLit = 0;
    for (let by = 0; by < blocks; by++) {
        for (let bx = 0; bx < blocks; bx++) {
            let webAvg = 0;
            let natAvg = 0;
            for (let y = by * kBlock; y < (by + 1) * kBlock; y++) {
                for (let x = bx * kBlock; x < (bx + 1) * kBlock; x++) {
                    const wi = (y * size + x) * 4;
                    const ni = ((height - 1 - y) * width + x) * 4;
                    for (let c = 0; c < 3; c++) {
                        webAvg += web[wi + c]!;
                        natAvg += data[ni + c]!;
                        signed += web[wi + c]! - data[ni + c]!;
                        natSum += data[ni + c]!;
                    }
                    if (web[wi]! + web[wi + 1]! + web[wi + 2]! > 0.01) webLit++;
                    if (data[ni]! + data[ni + 1]! + data[ni + 2]! > 0.01) natLit++;
                }
            }
            webAvg /= kBlock * kBlock * 3;
            natAvg /= kBlock * kBlock * 3;
            if (Math.abs(webAvg - natAvg) > 0.05 * Math.max(natAvg, 0.02)) badBlocks++;
        }
    }
    const toByte = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255);
    await saveArtifact("usd-web", Array.from({ length: size * size * 4 }, (_x, i) => toByte(web[i]!)), size, size, false);
    await saveArtifact("usd-native", Array.from({ length: size * size * 4 }, (_x, i) => {
        const p = Math.floor(i / 4);
        const y = Math.floor(p / size);
        const ni = ((height - 1 - y) * width + (p % size)) * 4 + (i % 4);
        return toByte(data[ni]!);
    }), size, size, false);

    // Region diagnostics: split the bias by native brightness bands.
    const bands = [
        ["floor-bright", (nv: number, y: number) => y > 120 && nv > 0.3],
        ["floor-dark", (nv: number, y: number) => y > 120 && nv <= 0.3],
        ["box", (_nv: number, y: number) => y > 40 && y <= 120],
    ] as const;
    for (const [label, pred] of bands) {
        let w = 0;
        let n = 0;
        let cnt = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * 4;
                const ni = ((height - 1 - y) * width + x) * 4;
                const nv = data[ni]! + data[ni + 1]! + data[ni + 2]!;
                if (!pred(nv, y)) continue;
                w += web[wi]! + web[wi + 1]! + web[wi + 2]!;
                n += nv;
                cnt++;
            }
        }
        console.error(`# usdImport.band ${label}: web/nat=${(w / Math.max(n, 1e-6)).toFixed(3)} px=${cnt}`);
    }

    const bias = signed / Math.max(natSum, 1e-6);
    console.error(`# usdImport: bias=${bias.toExponential(2)} badBlocks=${badBlocks}/${blocks * blocks} webLit=${webLit} natLit=${natLit}`);
    expectEq(Math.abs(bias) < 2e-2, true, `signed bias ${bias}`);
    expectEq(badBlocks <= 60, true, `bad 8x8 blocks ${badBlocks}`);
});
