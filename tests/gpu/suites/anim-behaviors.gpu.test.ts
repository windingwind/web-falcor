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

import { initScripting, runGraphScript, runSceneScript, type RenderGraph, type Scene, type Device } from "@web-falcor/falcor";
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
        console.error(`# anim-behaviors t=${time}: mean=${mean.toExponential(2)} bad@1e-2=${bad} webHits=${webHits} natHits=${natHits}`);
        // Both sides must show the behavior-displaced cubes (a Constant-only web
        // render parks all cubes at the first key -> thousands of bad pixels).
        expectEq(webHits > 1000, true, `cubes visible (webHits ${webHits})`);
        expectEq(Math.abs(webHits - natHits) < 300, true, `coverage matches (web ${webHits} vs native ${natHits})`);
        expectEq(bad <= 300, true, `depth bad pixels ${bad}`);
        expectEq(mean < 1e-3, true, `depth mean ${mean}`);
    }
});
