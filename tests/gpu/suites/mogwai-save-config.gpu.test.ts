/**
 * Mogwai's Save Config (Renderer::saveConfig): the viewer state written as a Mogwai script
 * and replayed through the script runner restores the graph, scene, camera and captures.
 */

import { Clock, float3, initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { saveConfig } from "../../../packages/mogwai/src/SaveConfig.js";
import { FrameCaptureExtension } from "../../../packages/mogwai/src/FrameCapture.js";
import { runMogwaiSource } from "../../../packages/mogwai/src/ScriptRunner.js";

gpuTest("MogwaiSaveConfig.roundTrip", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();
    const [graph] = await runGraphScript(device, graphSource);
    const scenePath = "/Falcor/media/test_scenes/cornell_box.pyscene";
    const scene = await runSceneScript(device, await (await fetch(scenePath)).text(), "/Falcor/media/test_scenes");
    scene.camera.setPosition(new float3(0.25, 0.5, 2.5));
    scene.camera.setTarget(new float3(0, 0.25, 0));
    scene.renderSettings.useEnvLight = false;
    const clock = new Clock().setFramerate(30).setExitFrame(9);
    const fc = new FrameCaptureExtension(device, () => graph!, () => graph!, () => clock.getFrame());
    fc.baseFilename = "cfg";
    fc.addFrames(graph!, [2, 5]);

    const script = saveConfig({ graphs: [graph!], scene, scenePath, width: 320, height: 240, showUI: true, clock, frameCapture: fc });
    console.error(`# save-config:\n${script.split("\n").filter((l) => l.startsWith("m.")).join("\n")}`);
    const replay = await runMogwaiSource(device, script, "/Falcor/media/test_scenes", { download: false });

    expectEq(replay.graphs.length, 1, "one graph");
    expectEq(replay.graphs[0]!.getPasses().map((p) => p.name).join(), graph!.getPasses().map((p) => p.name).join(), "graph passes");
    expectEq(replay.graphs[0]!.getOutputNames().join(), graph!.getOutputNames().join(), "graph outputs");
    const cam = replay.scene!.camera;
    expectEq(cam.getPosition().toArray().join(), "0.25,0.5,2.5", "camera position");
    expectEq(cam.getTarget().toArray().join(), "0,0.25,0", "camera target");
    expectEq(replay.scene!.renderSettings.useEnvLight, false, "render settings");
    expectEq(replay.frameCapture.baseFilename, "cfg", "capture base filename");
    expectEq(replay.frameCapture.print(replay.graphs[0]!), "\tframes = [2, 5]", "capture frames");
});
