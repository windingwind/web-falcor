/**
 * Scene::RenderSettings parity: the master light-usage switches gate the
 * scene's use*Lights flags, and the graph recompiles its passes when they change
 * (native IScene::UpdateFlags::RenderSettingsChanged). Cornell's only light is
 * emissive: with "Use emissive" off the MinimalPathTracer keeps the directly
 * visible emitter (native adds primary-hit emission unconditionally) but loses
 * all lighting, so the mean drops sharply.
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("RenderSettings.emissiveSwitchRecompilesPasses", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    const ctx = device.renderContext;
    const size = 64;

    const g = new RenderGraph(device, "MPT");
    g.addPass(createPass(device, "VBufferRT", { samplePattern: "Center" }), "VBufferRT");
    g.addPass(createPass(device, "MinimalPathTracer", { maxBounces: 1 }), "MPT");
    g.addEdge("VBufferRT.vbuffer", "MPT.vbuffer");
    g.addEdge("VBufferRT.viewW", "MPT.viewW");
    g.markOutput("MPT.color");
    g.onResize(size, size);
    g.setScene(scene);
    await g.init();

    const mean = async () => {
        g.execute(ctx);
        const px = new Float32Array((await ctx.readTextureSubresource(g.getOutput("MPT.color")!)).buffer);
        let sum = 0;
        for (let i = 0; i < size * size; i++) sum += px[i * 4]! + px[i * 4 + 1]! + px[i * 4 + 2]!;
        return sum / (size * size * 3);
    };

    expectEq(scene.useEmissiveLights, true, "emissive lights on by default");
    const lit = await mean();
    expectEq(lit > 0.01, true, `lit Cornell (mean ${lit.toExponential(2)})`);

    // Native: scene.renderSettings.useEmissiveLights = False (python) / "Use emissive" checkbox.
    scene.renderSettings.useEmissiveLights = false;
    expectEq(scene.useEmissiveLights, false, "switch gates useEmissiveLights");
    const dark = await mean(); // graph detects the change and re-sets the scene -> programs rebuilt
    console.error(`# render settings: lit=${lit.toExponential(3)} emissiveOff=${dark.toExponential(3)}`);
    expectEq(dark > 0 && dark < lit * 0.5, true, `emissive lighting removed, emitter still visible (mean ${dark} vs ${lit})`);

    scene.renderSettings.useEmissiveLights = true;
    const relit = await mean();
    expectEq(Math.abs(relit - lit) < 1e-6, true, `switch back restores the image (${relit} vs ${lit})`);
});
