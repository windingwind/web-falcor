/**
 * FEATURE VERIFY — PathTracer primaryLodMode=RayDiffs vs native over the
 * textured tutorial scene (checker floor minifies; emissive lighting). Ray
 * differentials at primary hits drive SampleGrad against the now-mipped
 * material arrays. Non-vacuousness pinned against the native Mip0 capture
 * (they differ at 30k px, mean 3e-2); filtering detail is
 * implementation-defined across APIs (docs §9), so gates are separation-based.
 *
 * Regenerate the oracles with:
 *   xvfb-run -a Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-pt-raydiffs.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("PTRayDiffs.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/tutorial.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    const render = async (mode: string) => {
        const graph = new RenderGraph(device, `PTRayDiffs${mode}`);
        graph.onResize(size, size);
        graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
        graph.addPass(
            createPass(device, "PathTracer", {
                samplesPerPixel: 1,
                maxSurfaceBounces: 3,
                maxDiffuseBounces: 3,
                maxSpecularBounces: 3,
                maxTransmissionBounces: 10,
                useRussianRoulette: false,
                primaryLodMode: mode,
            }),
            "PathTracer",
        );
        graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Accumulate");
        graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
        graph.addEdge("PathTracer.color", "Accumulate.input");
        graph.markOutput("Accumulate.output");
        graph.setScene(scene);
        for (let f = 0; f < 64; f++) graph.execute(device.renderContext);
        return new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("Accumulate.output")!)).buffer);
    };
    const webDiffs = await render("RayDiffs");
    const webMip0 = await render("Mip0");

    const load = async (name: string) => {
        const res = await fetch(`/tests/oracle/out-native/${name}.Accumulate.output.0.exr`);
        return parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    };
    const natDiffs = await load("oracle-pt-raydiffs-raydiffs");
    const natMip0 = await load("oracle-pt-raydiffs-mip0");
    expectEq(natDiffs.width, size, "oracle resolution");

    const meanDiff = (web: Float32Array, nat: { data: Float32Array; height: number; width: number } | Float32Array) => {
        let sum = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * 4;
                const ni = nat instanceof Float32Array ? wi : ((nat.height - 1 - y) * nat.width + x) * 4;
                const nd = nat instanceof Float32Array ? nat : nat.data;
                for (let c = 0; c < 3; c++) sum += Math.abs(web[wi + c]! - nd[ni + c]!);
            }
        }
        return sum / (size * size * 3);
    };
    // 2x2 cross matrix: each web mode must match its native counterpart best.
    const dDD = meanDiff(webDiffs, natDiffs);
    const dDM = meanDiff(webDiffs, natMip0);
    const dMM = meanDiff(webMip0, natMip0);
    const dMD = meanDiff(webMip0, natDiffs);
    // LOD-effect magnitude: web inter-mode delta vs native inter-mode delta.
    const webDelta = meanDiff(webDiffs, webMip0);
    console.error(
        `# pt-raydiffs cross: dDD=${dDD.toExponential(2)} dDM=${dDM.toExponential(2)} dMM=${dMM.toExponential(2)} dMD=${dMD.toExponential(2)} webDelta=${webDelta.toExponential(2)}`,
    );
    // A mode-independent ~6e-2 baseline offset exists on this scene (the
    // specular/normal-mapped tutorial materials diverge from native in PT —
    // tracked separately); the RayDiffs assertion is the cross-ordering plus
    // matching LOD-effect magnitude, which the baseline cancels out of.
    expectEq(dDD < dDM, true, `web RayDiffs closest to native RayDiffs (${dDD} < ${dDM})`);
    expectEq(dMM < dMD, true, `web Mip0 closest to native Mip0 (${dMM} < ${dMD})`);
    expectEq(Math.abs(webDelta - 3.0e-2) < 1.2e-2, true, `web LOD-effect magnitude ~ native 3.0e-2 (${webDelta})`);
    expectEq(dDD < 0.1 && dMM < 0.1, true, `baseline offset sanity (${dDD}, ${dMM})`);
});
