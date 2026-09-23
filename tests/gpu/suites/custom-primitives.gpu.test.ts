/**
 * Custom primitives (SceneBuilder/Scene::addCustomPrimitive): user IDs plus
 * AABBs, for passes that supply their own intersection code. None of the
 * shipped passes do — native PathTracer and MinimalPathTracer warn that they
 * do not support them — so they must not occlude anything. The web used to
 * draw them as box meshes.
 */

import { GeometryType, RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("CustomPrimitives.areSceneDataNotGeometry", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/custom-primitive.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setAspectRatio(1);

    expectEq(scene.getCustomPrimitiveCount(), 1, "one custom primitive");
    expectEq(scene.getCustomPrimitive(0).userID, 7, "its user ID");
    expectEq(scene.getCustomPrimitiveAABB(0).max.join(), "0.5,0.5,1.5", "its AABB");
    expectEq(scene.hasGeometryType(GeometryType.Custom), true, "the scene reports custom geometry");
    expectEq(scene.stats.materials, 1, "no invented material for it");

    // Runtime edits, as the native Scene API allows.
    const index = scene.addCustomPrimitive(8, { min: [2, 2, 2], max: [3, 3, 3] });
    scene.updateCustomPrimitive(index, { min: [2, 2, 2], max: [4, 4, 4] });
    expectEq(scene.getCustomPrimitiveAABB(index).max.join(), "4,4,4", "updated AABB");
    scene.removeCustomPrimitives(index, index + 1);
    expectEq(scene.getCustomPrimitiveCount(), 1, "removed again");

    // Rays pass through the primitive's box and hit the quad behind it.
    const size = 64;
    const graph = new RenderGraph(device, "Custom");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBuffer");
    graph.markOutput("GBuffer.posW");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    graph.execute(device.renderContext);
    const posW = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("GBuffer.posW")!)).buffer);
    let hits = 0;
    let inBox = 0;
    for (let i = 0; i < size * size; i++) {
        if (posW[i * 4 + 3] === 0) continue;
        hits++;
        if (posW[i * 4 + 2]! > 0.25) inBox++; // anything off the z = 0 quad
    }
    const centre = ((size / 2) * size + size / 2) * 4;
    console.error(`# custom primitive: ${hits} hits, ${inBox} off the quad; centre pixel z = ${posW[centre + 2]}`);
    expectEq(hits > size * size * 0.3, true, "the quad fills the view");
    expectEq(inBox, 0, "nothing is hit at the custom primitive's box");
    expectEq(Math.abs(posW[centre + 2]!) < 1e-4, true, "the centre ray reaches the quad");
});
