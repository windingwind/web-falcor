/**
 * FEATURE VERIFY — per-clip pre/post-infinity AnimationBehavior vs native.
 * animated_cubes.pyscene assigns Cycle/Cycle/Linear/Oscillate to clips 1..4
 * via sceneBuilder.animations (green cube stays Constant); keys span
 * 6.25..11.25s, so t=3.0/5.5 sample the pre-infinity region where every
 * cube's pose depends on its behavior. Depth is compared per frame against
 * the native VBufferRT capture.
 *
 * Regenerate the oracle with:
 *   xvfb-run -a Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-anim-behaviors.py --headless
 */

import { float16ToFloat32, float32ToFloat16, initScripting, runGraphScript, runSceneScript, type RenderGraph, type Scene, type Device } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

async function setup(device: Device): Promise<{ graph: RenderGraph; scene: Scene }> {
    await initScripting("/node_modules/pyodide");
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/VBufferRT.py")).text();
    const [graph] = await runGraphScript(device, graphSource);
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/animated_cubes/animated_cubes.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes/animated_cubes");
    scene.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    return { graph: graph!, scene };
}

gpuTest("AnimBehaviors.preInfinityMatchesNative", async ({ device }) => {
    const { graph, scene } = await setup(device);
    const ctx = device.renderContext;

    for (const [time, frame] of [
        [3.0, 30],
        [5.5, 55],
    ] as const) {
        scene.animate(time);
        graph.execute(ctx);
        const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("VBufferRT.depth")!)).buffer);
        const comps = web.length / (size * size);

        const res = await fetch(`/tests/oracle/out-native/oracle-anim-behaviors.VBufferRT.depth.${frame}.exr`);
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        expectEq(width, size, "oracle resolution");

        let bad = 0;
        let sum = 0;
        let webHits = 0;
        let natHits = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const w = web[(y * size + x) * comps]!;
                const n = data[((height - 1 - y) * width + x) * 4]!;
                const d = Math.abs(w - n);
                sum += d;
                if (d > 1e-2) bad++;
                if (w < 1) webHits++;
                if (n < 1) natHits++;
            }
        }
        const mean = sum / (size * size);
        const silhouette = await depthBad(device, graph, `/tests/oracle/out-native/oracle-anim-behaviors.VBufferRT.depth.${frame}.exr`);
        console.error(`# anim-behaviors t=${time}: mean=${mean.toExponential(2)} bad@1e-2=${bad} webHits=${webHits} natHits=${natHits} silhouette diff ${silhouette}`);
        // The half-rounded depth tells poses apart (a frame's motion moves 300-400 pixels; the floor is ~140).
        expectEq(silhouette <= 200, true, `silhouette pixels ${silhouette}`);
        // Both sides must show the behavior-displaced cubes (a Constant-only web
        // render parks all cubes at the first key -> thousands of bad pixels).
        expectEq(webHits > 1000, true, `cubes visible (webHits ${webHits})`);
        expectEq(Math.abs(webHits - natHits) < 300, true, `coverage matches (web ${webHits} vs native ${natHits})`);
        expectEq(bad <= 300, true, `depth bad pixels ${bad}`);
        expectEq(mean < 1e-3, true, `depth mean ${mean}`);
    }
});

/**
 * Pixels whose depth differs from a native capture. The captures store depth as half floats (steps
 * of 2^-11 near 1), so the web depth is rounded to half first; any remaining difference is a moved
 * silhouette (renders a frame apart differ in 300-400 pixels).
 */
async function depthBad(device: Device, graph: RenderGraph, oracle: string): Promise<number> {
    const web = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("VBufferRT.depth")!)).buffer);
    const comps = web.length / (size * size);
    const { data, width, height } = parseExr(await (await fetch(oracle)).arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    let bad = 0;
    for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) if (Math.abs(float16ToFloat32(float32ToFloat16(web[(y * size + x) * comps]!)) - data[((height - 1 - y) * width + x) * 4]!) > 1e-4) bad++;
    return bad;
}

// Scene.loopAnimations = False (test_AnimationBehavior's last section): past the keys' end each
// clip follows its own post-infinity behavior instead of wrapping (tests/oracle/render-native-anim-loop.py).
gpuTest("AnimBehaviors.loopAnimationsFalseMatchesNative", async ({ device }) => {
    const { graph, scene } = await setup(device);
    for (const [time, frame] of [
        [12.5, 125],
        [14, 140],
    ] as const) {
        const oracle = `/tests/oracle/out-native/oracle-anim-loop.VBufferRT.depth.${frame}.exr`;
        scene.loopAnimations = true;
        scene.animate(time);
        graph.execute(device.renderContext);
        const looped = await depthBad(device, graph, oracle);
        scene.loopAnimations = false;
        scene.animate(time);
        graph.execute(device.renderContext);
        const bad = await depthBad(device, graph, oracle);
        console.error(`# anim-loop t=${time}: silhouette diff unlooped ${bad}, looped ${looped}`);
        expectEq(looped > 300, true, `looping changes the poses (${looped} px differ)`);
        expectEq(bad <= 200, true, `silhouette pixels ${bad}`);
    }
});
