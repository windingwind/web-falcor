/**
 * FEATURE VERIFY — grid volumes on GPU vs native. The upstream
 * SceneDebugger.py graph over the smoke volume scene: SceneDebugger's
 * handleVolumes ray-marches grid-volume transmittance (500 steps through
 * the NanoVDB grid via PNanoVDB accessors) over every pixel — the upstream
 * GPU consumer of Grid.slang in this Falcor drop (no shipped pass renders
 * volumes in light transport).
 *
 * Web loads the UNMODIFIED smoke.pyscene (parses the original OpenVDB
 * smoke.vdb in-browser); native loads the byte-identical pre-converted
 * smoke.nvdb (native openvdb is broken on this machine, DESIGN §6.3).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-feature-smoke-debugger.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("FeatureSmokeDebugger.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/SceneDebugger.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/smoke.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("SceneDebugger.output")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-feature-smoke-debugger.SceneDebugger.output.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let bad = 0;
    let maxD = 0;
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
            if (pixelMax > 1e-2) bad++;
            maxD = Math.max(maxD, pixelMax);
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# oracle-feature-smoke-debugger: mean=${mean.toExponential(2)} bad@1e-2=${bad} max=${maxD.toExponential(2)}`);
    expectEq(mean < 1e-3, true, `smoke transmittance mean ${mean}`);
    expectEq(bad <= 100, true, `smoke transmittance bad pixels ${bad}`);
});
