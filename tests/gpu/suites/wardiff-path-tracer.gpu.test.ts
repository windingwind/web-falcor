/**
 * WARDiffPathTracer in ForwardDiffDebug mode through the upstream graph scripts
 * (WARDiffPathTracerTranslationFwd.py, WARDiffPathTracerMaterialFwd.py) over
 * bunny_war_diff_pt.pyscene with the upstream test's builder flags. The accumulated primal and
 * gradient images after 64 frames are compared with native captures.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-wardiff.py --headless
 */

import { SceneBuilderFlags, initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 128;
const frames = 64;

gpuTest("WARDiffPathTracer.forwardDiffMatchesNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/bunny_war_diff_pt.pyscene")).text();
    const flags = SceneBuilderFlags.DontMergeMaterials | SceneBuilderFlags.RTDontMergeDynamic | SceneBuilderFlags.DontOptimizeMaterials;
    const native = async (file: string) => (parseExr(await (await fetch(`/tests/oracle/out-native/${file}`)).arrayBuffer(), 1015) as { data: Float32Array }).data;

    for (const name of ["WARDiffPathTracerTranslationFwd", "WARDiffPathTracerMaterialFwd"]) {
        const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes", { flags });
        const [graph] = await runGraphScript(device, await (await fetch(`/Falcor/tests/image_tests/renderpasses/graphs/${name}.py`)).text());
        scene.camera.setAspectRatio(1);
        graph!.onResize(size, size);
        graph!.setScene(scene);
        await graph!.init();
        const t0 = performance.now();
        for (let f = 0; f < frames; f++) graph!.execute(ctx);
        const read = async (output: string) => new Float32Array((await ctx.readTextureSubresource(graph!.getOutput(output)!)).buffer);
        const summary: string[] = [];
        for (const output of ["AccumulatePassPrimal.output", "AccumulatePassDiff.output"]) {
            const [web, nat] = [await read(output), await native(`wardiff-${name}.${output}.${frames}.exr`)];
            // Native EXR captures are bottom-up. Per pixel, bounced paths decorrelate (float rounding flips
            // sampling decisions, as with the PathTracer oracles), so the gate is on 8x8 block means.
            const block = 8;
            const blocks = size / block;
            const [webB, natB] = [new Float64Array(blocks * blocks * 3), new Float64Array(blocks * blocks * 3)];
            let pixelAbs = 0, pixelNat = 0, signedWeb = 0, signedNat = 0;
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const wi = (y * size + x) * 4, ni = ((size - 1 - y) * size + x) * 4;
                    const bi = (Math.floor(y / block) * blocks + Math.floor(x / block)) * 3;
                    for (let k = 0; k < 3; k++) {
                        pixelAbs += Math.abs(web[wi + k]! - nat[ni + k]!);
                        pixelNat += Math.abs(nat[ni + k]!);
                        signedWeb += web[wi + k]!;
                        signedNat += nat[ni + k]!;
                        webB[bi + k] += web[wi + k]!;
                        natB[bi + k] += nat[ni + k]!;
                    }
                }
            }
            let blockAbs = 0, blockNat = 0;
            for (let i = 0; i < webB.length; i++) {
                blockAbs += Math.abs(webB[i]! - natB[i]!);
                blockNat += Math.abs(natB[i]!);
            }
            const rel = blockAbs / Math.max(blockNat, 1e-12);
            const sumRel = Math.abs(signedWeb - signedNat) / Math.max(Math.abs(signedNat), 1e-12);
            summary.push(`${output} block relL1 ${rel.toExponential(2)}, pixel relL1 ${(pixelAbs / pixelNat).toExponential(2)}, sum rel ${sumRel.toExponential(2)}`);
            expectEq(pixelNat > 0, true, `${name} ${output}: native image is non-zero`);
            expectEq(rel < 0.05, true, `${name} ${output}: 8x8 block relative L1 ${rel}`);
            expectEq(sumRel < 0.01, true, `${name} ${output}: image sum relative error ${sumRel}`);
        }
        console.error(`# wardiff ${name}: ${(performance.now() - t0).toFixed(0)}ms, ${summary.join(", ")}`);
    }
});
