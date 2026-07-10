/**
 * FEATURE VERIFY — camera depth of field (thin lens): the UNMODIFIED upstream
 * VBufferRT.py graph over a scene with apertureRadius 0.05 / focalDistance
 * 3.05. The thin-lens aperture samples come from the bit-matched
 * SampleGenerator, so depth/viewW compare per-pixel against native EXRs.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-dof.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("FeatureDoF.vbufferMatchesNativeOracle", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");

    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/VBufferRT.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const sceneSource = await (await fetch("/tests/oracle/assets/oracle-dof.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/tests/oracle/assets");
    scene.camera.setAspectRatio(1.0);
    expectEq(scene.camera.getApertureRadius() > 0, true, "aperture radius set from pyscene");

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    for (const [channel, components, tol, badTol] of [
        ["depth", 1, 1e-3, 200],
        ["viewW", 3, 1e-3, 200],
    ] as const) {
        const tex = graph!.getOutput(`VBufferRT.${channel}`)!;
        const web = new Float32Array((await ctx.readTextureSubresource(tex)).buffer);
        const webComponents = web.length / (size * size);

        const res = await fetch(`/tests/oracle/out-native/oracle-dof.VBufferRT.${channel}.0.exr`);
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        expectEq(width, size, `${channel} oracle resolution`);

        let sum = 0;
        let badPixels = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * webComponents;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pixelMax = 0;
                for (let c = 0; c < components; c++) {
                    const d = Math.abs(web[wi + c]! - data[ni + c]!);
                    sum += d;
                    pixelMax = Math.max(pixelMax, d);
                }
                if (pixelMax > 1e-3) badPixels++;
            }
        }
        const mean = sum / (size * size * components);
        console.error(`# featureDoF.${channel}: mean=${mean.toExponential(2)} bad=${badPixels}`);
        expectEq(mean < tol, true, `${channel} mean ${mean}`);
        expectEq(badPixels <= badTol, true, `${channel} bad pixels ${badPixels}`);
    }

    // Structural check: thin-lens dirs must differ from the pinhole dirs the
    // same camera produces with aperture 0 (guards against DoF silently off).
    const viewWTex = graph!.getOutput("VBufferRT.viewW")!;
    const dof = new Float32Array((await ctx.readTextureSubresource(viewWTex)).buffer);
    scene.camera.setApertureRadius(0);
    graph!.execute(ctx);
    const pinhole = new Float32Array((await ctx.readTextureSubresource(viewWTex)).buffer);
    let changed = 0;
    for (let i = 0; i < dof.length; i += 4) {
        if (Math.abs(dof[i]! - pinhole[i]!) > 1e-5) changed++;
    }
    console.error(`# featureDoF.structural: ${changed}/${size * size} px differ from pinhole`);
    expectEq(changed > size * size * 0.5, true, `thin-lens rays differ from pinhole (${changed})`);
});
