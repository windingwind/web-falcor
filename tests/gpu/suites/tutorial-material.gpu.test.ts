/**
 * FEATURE VERIFY — tutorial.pyscene material channels vs native (GBufferRT
 * normW / specRough / guideNormalW / emissive). Adjudicates the ~6% PT
 * radiance deficit found by the texLOD cross-oracle: whichever channel
 * diverges localizes the pyscene loadTexture slot-semantics bug.
 *
 * Regenerate the oracle with:
 *   xvfb-run -a Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-tutorial-gbuffer.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("TutorialMaterial.gbufferChannelsMatchNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const graphSource = `
from falcor import *
g = RenderGraph('TutorialGBuffer')
g.addPass(createPass('GBufferRT', {'samplePattern': 'Center'}), 'GBufferRT')
g.markOutput('GBufferRT.normW')
g.markOutput('GBufferRT.specRough')
g.markOutput('GBufferRT.guideNormalW')
g.markOutput('GBufferRT.emissive')
try: m.addGraph(g)
except NameError: None
`;
    const [graph] = await runGraphScript(device, graphSource);
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/tutorial.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const results: [string, number, number][] = [];
    for (const channel of ["normW", "specRough", "guideNormalW", "emissive"]) {
        const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput(`GBufferRT.${channel}`)!)).buffer);
        const res = await fetch(`/tests/oracle/out-native/oracle-tutorial-gbuffer.GBufferRT.${channel}.0.exr`);
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        let sum = 0;
        let bad = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * 4;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pm = 0;
                for (let c = 0; c < 3; c++) {
                    const d = Math.abs(web[wi + c]! - data[ni + c]!);
                    sum += d;
                    pm = Math.max(pm, d);
                }
                if (pm > 1e-2) bad++;
            }
        }
        const mean = sum / (size * size * 3);
        // Spatial histogram of bad pixels (3x3 grid) to localize divergence.
        const grid = new Array(9).fill(0);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * 4;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pm = 0;
                for (let c = 0; c < 3; c++) pm = Math.max(pm, Math.abs(web[wi + c]! - data[ni + c]!));
                if (pm > 1e-2) grid[Math.floor((y * 3) / size) * 3 + Math.floor((x * 3) / size)]++;
            }
        }
        // Worst pixel dump: web vs native values.
        let worst = 0;
        let wx = 0;
        let wy = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * 4;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pm = 0;
                for (let c = 0; c < 3; c++) pm = Math.max(pm, Math.abs(web[wi + c]! - data[ni + c]!));
                if (pm > worst) { worst = pm; wx = x; wy = y; }
            }
        }
        const wi = (wy * size + wx) * 4;
        const ni = ((height - 1 - wy) * width + wx) * 4;
        console.error(`# tutorial-material ${channel}: mean=${mean.toExponential(2)} bad@1e-2=${bad} grid=[${grid.join(",")}] worst@(${wx},${wy}) web=(${web[wi]!.toFixed(3)},${web[wi+1]!.toFixed(3)},${web[wi+2]!.toFixed(3)}) nat=(${data[ni]!.toFixed(3)},${data[ni+1]!.toFixed(3)},${data[ni+2]!.toFixed(3)})`);
        results.push([channel, mean, bad]);
    }
    // Normal channels tolerate face-boundary noise (flat-shaded 5k-face mesh:
    // sub-ulp hit differences flip which face a pixel lands on).
    for (const [channel, mean, bad] of results) {
        const isNormal = channel === "normW" || channel === "guideNormalW";
        expectEq((mean as number) < (isNormal ? 5e-3 : 5e-4), true, `${channel} mean ${mean}`);
        expectEq((bad as number) <= (isNormal ? 500 : 100), true, `${channel} bad px ${bad}`);
    }
});
