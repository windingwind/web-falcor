/**
 * SDF grids loaded from `.sdfg` files (SDFGrid::loadValuesFromFile): the corner
 * values come from disk instead of the procedural generator, which is the
 * content path native scenes use (and a prerequisite for SDF editing, §8.4).
 *
 * The fixture is an analytic sphere written by `node scripts/gen-assets.mjs sdf`,
 * so the rendered surface has a closed form: every hit must sit on the sphere.
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose, SkipError } from "../harness/registry.js";

const size = 128;

gpuTest("SDFFromFile.loadedGridRendersTheAnalyticSphere", async ({ device }) => {
    if (!(await fetch("/Falcor/media/sdf/sdf-sphere-64.sdfg", { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/sdf/sdf-sphere-64.sdfg missing (node scripts/gen-assets.mjs sdf)");
    }
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/tests/oracle/assets/sdf-from-file.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media");
    scene.camera.setAspectRatio(1.0);
    expectEq(scene.sdfGrids.length, 1, "one SDF grid");

    const graph = new RenderGraph(device, "SDFFile");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBufferRT");
    graph.markOutput("GBufferRT.posW");
    graph.markOutput("GBufferRT.faceNormalW");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);

    const posW = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRT.posW")!)).buffer);
    const normalW = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRT.faceNormalW")!)).buffer);

    // Every hit must lie on the sphere of radius 0.4, with the normal pointing out.
    let hits = 0;
    let worstRadius = 0;
    let worstNormal = 0;
    for (let i = 0; i < size * size; i++) {
        if (posW[i * 4 + 3] === 0) continue; // background
        hits++;
        const p = [posW[i * 4]!, posW[i * 4 + 1]!, posW[i * 4 + 2]!];
        const r = Math.hypot(p[0]!, p[1]!, p[2]!);
        worstRadius = Math.max(worstRadius, Math.abs(r - 0.4));
        // The outward normal of a sphere is the normalized position.
        const dot = (p[0]! * normalW[i * 4]! + p[1]! * normalW[i * 4 + 1]! + p[2]! * normalW[i * 4 + 2]!) / Math.max(r, 1e-6);
        worstNormal = Math.max(worstNormal, Math.abs(dot - 1));
    }
    console.error(`# SDF from file: ${hits}/${size * size} hits, worst |r - 0.4| = ${worstRadius.toExponential(2)}, worst normal deviation ${worstNormal.toExponential(2)}`);
    // A 64^3 grid resolves the sphere to about one voxel (1/64 = 0.016).
    expectEq(hits > 1000, true, `the grid is visible (${hits} hits)`); // radius 0.4 at distance 2 covers ~10% of the frame
    expectClose(worstRadius, 0, 0.02, "hits lie on the analytic sphere");
    expectEq(worstNormal < 0.15, true, `normals point outward (${worstNormal})`);
});
