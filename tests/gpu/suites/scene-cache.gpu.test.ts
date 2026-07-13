/**
 * SceneCache phase 1: importing cornell_box with cache enabled stores the
 * scene description in OPFS; the next load skips script + import entirely
 * and must render BYTE-IDENTICALLY through the MinimalPathTracer graph.
 */

import { clearSceneCache, initScripting, runGraphScript, runSceneScript, wasSceneLoadedFromCache } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 128;

gpuTest("SceneCache.cachedSceneRendersIdentically", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    await clearSceneCache();
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();
    const ctx = device.renderContext;

    const render = async (scene: Awaited<ReturnType<typeof runSceneScript>>) => {
        const [graph] = await runGraphScript(device, graphSource);
        scene.camera.setAspectRatio(1.0);
        graph!.onResize(size, size);
        graph!.setScene(scene);
        graph!.execute(ctx);
        return new Uint8Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    };

    const scene1 = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes", { cache: true });
    expectEq(wasSceneLoadedFromCache(), false, "first load imports");
    const img1 = await render(scene1);

    const t0 = performance.now();
    const scene2 = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes", { cache: true });
    const cachedMs = performance.now() - t0;
    expectEq(wasSceneLoadedFromCache(), true, "second load hits the cache");
    const img2 = await render(scene2);

    let diff = 0;
    for (let i = 0; i < img1.length; i++) if (img1[i] !== img2[i]) diff++;
    console.error(`# scene-cache: cached load ${cachedMs.toFixed(1)}ms, ${diff} differing bytes of ${img1.length}`);
    expectEq(diff, 0, "cached scene renders byte-identically");

    await clearSceneCache();
});
