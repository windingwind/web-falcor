/**
 * PixelStats getStats() aggregate: GPU per-region sums of the packed stats
 * buffer must match CPU sums of the per-pixel rayCount/pathLength outputs
 * exactly (both derive from the same buffer, so equality is bit-exact; the
 * per-pixel outputs themselves are oracle-verified in imgtest-pathtracer).
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 128;

gpuTest("PixelStats.aggregateMatchesPerPixelSums", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/PathTracer.py")).text();
    const [graph] = await runGraphScript(device, source);
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);
    ctx.submit();

    const pt = graph!.getPass("PathTracer") as unknown as { getPixelStats(): { valid: boolean; totalRays: number; visibilityRays: number; closestHitRays: number; avgPathLength: number; pathVertices: number } };
    let stats = pt.getPixelStats();
    for (let i = 0; i < 50 && !stats.valid; i++) {
        await new Promise((r) => setTimeout(r, 50));
        stats = pt.getPixelStats();
    }
    expectEq(stats.valid, true, "stats resolved");

    const sumU32 = async (ref: string) => {
        const data = new Uint32Array((await ctx.readTextureSubresource(graph!.getOutput(ref)!)).buffer);
        return data.reduce((a, v) => a + v, 0);
    };
    const rayCountSum = await sumU32("PathTracer.rayCount");
    const pathLengthSum = await sumU32("PathTracer.pathLength");
    const n = size * size;

    console.error(`# stats: totalRays=${stats.totalRays} (vis ${stats.visibilityRays} + hit ${stats.closestHitRays}) rayCountSum=${rayCountSum} pathLen=${Math.round(stats.avgPathLength * n)}/${pathLengthSum} pathVertices=${stats.pathVertices}`);
    expectEq(stats.totalRays > 0, true, "rays traced");
    expectEq(stats.totalRays, stats.visibilityRays + stats.closestHitRays, "totalRays = visibility + closestHit");
    expectEq(stats.totalRays, rayCountSum, "aggregate matches rayCount texture sum");
    expectEq(Math.round(stats.avgPathLength * n), pathLengthSum, "aggregate matches pathLength texture sum");
    expectEq(stats.pathVertices > 0, true, "path vertices counted");
});
