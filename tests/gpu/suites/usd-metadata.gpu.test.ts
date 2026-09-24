/**
 * Scene::Metadata from Omniverse render settings (usd-metadata.usda's
 * customLayerData.renderSettings), applied by the ToneMapper on setScene like
 * native. The expected property list is native's (probed with Mogwai after
 * loading the same file).
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("UsdMetadata.toneMapperTakesSceneMetadata", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(device, "sceneBuilder.importScene('usd-metadata.usda')", "/tests/oracle/assets");
    const m = scene.metadata;
    expectEq(JSON.stringify([m.fNumber, m.filmISO, m.shutterSpeed, m.samplesPerPixel]), JSON.stringify([2.8, 400, 100, 4]), "render settings");
    // Create's bounces minus 2 (primary visibility and NEE), specular at least diffuse.
    expectEq(JSON.stringify([m.maxDiffuseBounces, m.maxSpecularBounces, m.maxTransmissionBounces, m.maxVolumeBounces]), JSON.stringify([5, 5, 5, 5]), "bounce mapping");

    const graph = new RenderGraph(device, "TM");
    graph.addPass(createPass(device, "GBufferRT", { useTraceRayInline: true }), "GBufferRT");
    const tm = createPass(device, "ToneMapper", { autoExposure: false });
    graph.addPass(tm, "ToneMapper");
    graph.addEdge("GBufferRT.diffuseOpacity", "ToneMapper.src");
    graph.markOutput("ToneMapper.dst");
    graph.onResize(64, 64);
    graph.setScene(scene);
    await graph.init();
    graph.execute(device.renderContext);
    const props = tm.getProperties().toJSON();
    const native = { outputSize: "Default", useSceneMetadata: true, exposureCompensation: 0, autoExposure: false, filmSpeed: 400, whiteBalance: false, whitePoint: 6500, operator: "Aces", clamp: true, whiteMaxLuminance: 1, whiteScale: 11.2, fNumber: 2.8, shutter: 100, exposureMode: "AperturePriority" };
    expectEq(Object.keys(props).join(","), Object.keys(native).join(","), "native's property keys and order");
    for (const [k, v] of Object.entries(native)) {
        if (typeof v === "number") expectClose(props[k] as number, v, 1e-5, k);
        else expectEq(props[k], v, k);
    }

    // useSceneMetadata = false keeps the pass's own settings.
    const own = createPass(device, "ToneMapper", { useSceneMetadata: false, fNumber: 1.4 });
    own.setScene(scene);
    expectClose(own.getProperties().toJSON().fNumber as number, 1.4, 1e-6, "metadata ignored when disabled");
});
