/**
 * OVERALL VERIFY — upstream image tests: the UNMODIFIED GBufferRTTexGrads.py
 * (texGrads channel, default Mip0 LOD) and MVecRT.py (mvec with stratified
 * camera jitter, 4 frames) graphs over cornell_box.pyscene, diffed against
 * native EXR captures. (Their upstream test scenes need loadTexture-in-pyscene
 * and animated-glTF support respectively — pending; the graphs are untouched.)
 *
 * Regenerate oracles with:
 *   Mogwai --script tests/oracle/render-native-imgtest-texgrads.py --headless
 *   Mogwai --script tests/oracle/render-native-imgtest-mvecrt.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

function halfToFloat(h: number): number {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

async function runGraph(device: import("@web-falcor/falcor").Device, graphFile: string, frames: number) {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch(`/Falcor/tests/image_tests/renderpasses/graphs/${graphFile}`)).text();
    const [graph] = await runGraphScript(device, source);
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    for (let f = 0; f < frames; f++) graph!.execute(device.renderContext);
    return graph!;
}

async function compare(web: Float32Array, oracle: string, components: number, tol: number, badTol: number) {
    const res = await fetch(`/tests/oracle/out-native/${oracle}`);
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, `${oracle} resolution`);
    const webComponents = web.length / (size * size);
    let sum = 0;
    let bad = 0;
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
            if (pixelMax > 1e-3) bad++;
        }
    }
    const mean = sum / (size * size * components);
    console.error(`# ${oracle}: mean=${mean.toExponential(2)} bad=${bad}`);
    expectEq(mean < tol, true, `${oracle} mean ${mean}`);
    expectEq(bad <= badTol, true, `${oracle} bad ${bad}`);
}

gpuTest("ImageTestTexGrads.matchesNativeOracle", async ({ device }) => {
    const graph = await runGraph(device, "GBufferRTTexGrads.py", 1);
    const raw = await device.renderContext.readTextureSubresource(graph.getOutput("GBufferRT.texGrads")!);
    const web = Float32Array.from(new Uint16Array(raw.buffer), halfToFloat);
    await compare(web, "oracle-imgtest-texgrads.GBufferRT.texGrads.0.exr", 3, 1e-4, 50);
});

gpuTest("ImageTestMVecRT.matchesNativeOracle", async ({ device }) => {
    // Frame 4 with stratified jitter: static scene, so mvec is the per-frame
    // jitter offset — bit-exact jitter sequences must line up on both sides.
    const graph = await runGraph(device, "MVecRT.py", 4);
    const web = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("GBufferRT.mvec")!)).buffer);
    await compare(web, "oracle-imgtest-mvecrt.GBufferRT.mvec.0.exr", 2, 1e-4, 100);
});
