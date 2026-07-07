/**
 * MOGWAI VIEWER smoke test — the interactive viewer's render loop + present
 * path, headless. Loads the viewer's default content (MinimalPathTracer.py +
 * cornell_box.pyscene), executes the graph, and presents the marked output
 * through presentToCanvas to a bgra8unorm target (standing in for the
 * swapchain). Verifies the presented frame is non-trivial (the box is
 * visible) — i.e. the graph->present->display pipeline the browser app runs
 * works end to end. (The DOM/canvas wiring itself isn't headless-testable;
 * the render loop it drives is verified here and by the per-pass oracles.)
 */

import {
    initScripting,
    runSceneScript,
    presentToCanvas,
    RenderGraph,
    createPass,
    Texture,
    ResourceType,
    ResourceFormat,
    ResourceBindFlags,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const width = 256;
const height = 256;

gpuTest("MogwaiViewer.rendersAndPresentsDefaultContent", async ({ device }) => {
    await initScripting("/node_modules/pyodide");

    // Viewer default content: the verified cornell path-tracer graph.
    const sceneSrc = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSrc, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1);

    const graph = new RenderGraph(device, "Default");
    graph.onResize(width, height);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, emissiveSampler: "LightBVH" }), "PathTracer");
    graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Accumulate");
    graph.addPass(createPass(device, "ToneMapper", { autoExposure: false }), "ToneMapper");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.addEdge("PathTracer.color", "Accumulate.input");
    graph.addEdge("Accumulate.output", "ToneMapper.src");
    graph.markOutput("ToneMapper.dst");
    graph.setScene(scene);

    const outputName = graph.getOutputNames()[0];
    expectEq(outputName, "ToneMapper.dst", "graph exposes the marked output");

    // Accumulate several frames for a clean image.
    for (let f = 0; f < 16; f++) graph.execute(device.renderContext);
    const outTex = graph.getOutput(outputName!)!;

    // Present to a bgra8unorm target (the swapchain format) and read it back.
    const present = new Texture(device, {
        type: ResourceType.Texture2D,
        width,
        height,
        mipLevels: 1,
        format: ResourceFormat.BGRA8Unorm,
        bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        name: "MogwaiViewer::presentTarget",
    });
    presentToCanvas(device, outTex, present.gpuTexture, "bgra8unorm");

    const pixels = await device.renderContext.readTextureSubresource(present);
    // Non-trivial frame: a fair fraction of pixels are lit (the cornell box).
    let lit = 0;
    for (let i = 0; i < width * height; i++) {
        const b = pixels[i * 4]!;
        const g = pixels[i * 4 + 1]!;
        const r = pixels[i * 4 + 2]!;
        if (r + g + b > 30) lit++;
    }
    let nonzero = 0;
    for (let i = 0; i < width * height; i++) {
        if (pixels[i * 4]! + pixels[i * 4 + 1]! + pixels[i * 4 + 2]! > 0) nonzero++;
    }
    console.error(`# mogwai-viewer: output=${outputName} lit=${lit} nonzero=${nonzero} /${width * height}`);
    // The present pipeline delivered a structured, non-trivial frame (the box).
    expectEq(nonzero > 20000, true, `presented frame near-empty (${nonzero} nonzero px)`);
    expectEq(lit > 800, true, `presented frame too dark (${lit} lit px)`);
});
