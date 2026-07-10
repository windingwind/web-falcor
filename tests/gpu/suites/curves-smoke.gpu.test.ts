/**
 * Curves phase 3a smoke: the upstream two_curves.pyscene (spheres + quad +
 * USD BasisCurves) builds — curve buffers, instances, and the Curve
 * geometry-type define compile through the MPT graph. Curves are not yet
 * intersected (phase 3b: segment-AABB BVH + CurveIntersector); the render
 * shows the triangle geometry.
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("CurvesSmoke.sceneBuildsAndRenders", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/curves/two_curves.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes/curves");
    scene.camera.setAspectRatio(1.0);
    // 3 spheres + quad + usda tri0 + 2 custom-primitive boxes + 2 curves.
    expectEq(scene.stats.instances, 9, `instances (${scene.stats.instances})`);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    let lit = 0;
    let finite = true;
    for (let i = 0; i < size * size; i++) {
        if (web[i * 4]! + web[i * 4 + 1]! + web[i * 4 + 2]! > 0.01) lit++;
        finite &&= Number.isFinite(web[i * 4]!);
    }
    console.error(`# curvesSmoke: lit=${lit}/${size * size}`);
    expectEq(finite, true, "finite output");
    expectEq(lit > 20000, true, `scene renders (${lit} lit px)`);
});
