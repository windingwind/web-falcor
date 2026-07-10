/**
 * FEATURE VERIFY — object motion vectors for animated geometry via the
 * upstream VBufferRT.py graph, two frames 0.1s apart.
 *
 * - animated_cubes (rigid -> prevWorldMatrices): web mvec vs native mvec,
 *   sampled INSIDE the clip key range (6.3s..6.4s) so no AnimationBehavior
 *   extrapolation (unimplemented on web) is involved.
 * - cesium_man (skinned -> prevVertices): native writes ZERO mvecs for this
 *   skinned content (probed: depth animates, mvec stays 0 even mid-walk), so
 *   geometry parity is checked vs native depth and the web mvec is verified
 *   by reprojection: warping frame-1 moving pixels by mvec must land on the
 *   frame-0 body.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-mvec-anim.py --headless
 */

import { initScripting, runGraphScript, runSceneScript, type Device, type RenderGraph, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

async function setup(device: Device, sceneDir: string, sceneFile: string): Promise<{ graph: RenderGraph; scene: Scene }> {
    await initScripting("/node_modules/pyodide");
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/VBufferRT.py")).text();
    const [graph] = await runGraphScript(device, graphSource);
    const sceneSource = await (await fetch(`/Falcor/media/test_scenes/${sceneDir}/${sceneFile}`)).text();
    const scene = await runSceneScript(device, sceneSource, `/Falcor/media/test_scenes/${sceneDir}`);
    scene.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    return { graph: graph!, scene };
}

async function loadOracle(oracle: string, channel: string, frame: number): Promise<Float32Array> {
    const res = await fetch(`/tests/oracle/out-native/${oracle}.VBufferRT.${channel}.${frame}.exr`);
    const { data, width } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number };
    expectEq(width, size, `${oracle} ${channel} resolution`);
    return data;
}

gpuTest("FeatureMvecAnim.rigidMatchesNative", async ({ device }) => {
    const { graph, scene } = await setup(device, "animated_cubes", "animated_cubes.pyscene");
    const ctx = device.renderContext;
    scene.animate(6.3);
    graph.execute(ctx);
    scene.animate(6.4);
    graph.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("VBufferRT.mvec")!)).buffer);
    const webComponents = web.length / (size * size);
    const nat = await loadOracle("oracle-mvec-animated_cubes", "mvec", 64);

    let sum = 0;
    let bad = 0;
    let webMotion = 0;
    let natMotion = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const wi = (y * size + x) * webComponents;
            const ni = ((size - 1 - y) * size + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 2; c++) {
                const d = Math.abs(web[wi + c]! - nat[ni + c]!);
                sum += d;
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 1e-3) bad++;
            if (Math.abs(web[wi]!) + Math.abs(web[wi + 1]!) > 1e-4) webMotion++;
            if (Math.abs(nat[ni]!) + Math.abs(nat[ni + 1]!) > 1e-4) natMotion++;
        }
    }
    const mean = sum / (size * size * 2);
    console.error(`# mvecAnim.rigid: mean=${mean.toExponential(2)} bad=${bad} webMotion=${webMotion} natMotion=${natMotion}`);
    // Moving-silhouette pixels hit different geometry across the software/
    // hardware RT divide; gate on the mean + a bounded bad-pixel tail.
    expectEq(webMotion > natMotion * 0.9 && webMotion < natMotion * 1.1, true, `motion coverage web=${webMotion} native=${natMotion}`);
    expectEq(mean < 5e-4, true, `mvec mean ${mean}`);
    expectEq(bad <= 800, true, `mvec bad pixels ${bad}`);
});

gpuTest("FeatureMvecAnim.skinnedReprojects", async ({ device }) => {
    const { graph, scene } = await setup(device, "cesium_man", "CesiumMan.pyscene");
    const ctx = device.renderContext;
    scene.animate(0);
    graph.execute(ctx);
    const depth0 = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("VBufferRT.depth")!)).buffer);
    const mask0 = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("VBufferRT.mask")!)).buffer);
    scene.animate(0.1);
    graph.execute(ctx);
    const depth1 = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("VBufferRT.depth")!)).buffer);
    const mvec = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("VBufferRT.mvec")!)).buffer);
    const mask1 = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("VBufferRT.mask")!)).buffer);
    const mvecComponents = mvec.length / (size * size);

    // Geometry parity vs native at t=0 (import/bind pose) and t=0.1 (animated pose).
    for (const [frame, webDepth] of [[0, depth0], [1, depth1]] as const) {
        const natDepth = await loadOracle("oracle-mvec-cesium_man", "depth", frame);
        let dSum = 0;
        let dBad = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const d = Math.abs(webDepth[y * size + x]! - natDepth[((size - 1 - y) * size + x) * 4]!);
                dSum += d;
                if (d > 1e-3) dBad++;
            }
        }
        console.error(`# mvecAnim.skinned depth.${frame}: mean=${(dSum / (size * size)).toExponential(2)} bad=${dBad}`);
        // Web CPU skinning vs native GPU skinning drifts on silhouette pixels
        // (present already at t=0); the mean gate keeps the pose aligned.
        expectEq(dSum / (size * size) < 1e-3, true, `depth.${frame} mean ${dSum / (size * size)}`);
        expectEq(dBad <= 2500, true, `depth.${frame} bad ${dBad}`);
    }

    // Reprojection: moving body pixels warped by mvec must land on the
    // frame-0 body (mask hit) at a similar depth.
    let moving = 0;
    let landed = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const i = y * size + x;
            const mx = mvec[i * mvecComponents]!;
            const my = mvec[i * mvecComponents + 1]!;
            if (mask1[i]! === 0 || Math.abs(mx) + Math.abs(my) < 1e-4) continue;
            moving++;
            const px = Math.round(x + mx * size);
            const py = Math.round(y + my * size);
            if (px < 0 || py < 0 || px >= size || py >= size) continue;
            const j = py * size + px;
            if (mask0[j]! !== 0 && Math.abs(depth0[j]! - depth1[i]!) < 0.05) landed++;
        }
    }
    console.error(`# mvecAnim.skinned reproject: moving=${moving} landed=${landed}`);
    expectEq(moving > 1000, true, `skinned motion present (${moving})`);
    expectEq(landed / Math.max(moving, 1) > 0.9, true, `reprojection hit rate ${(landed / Math.max(moving, 1)).toFixed(3)}`);
});
