/**
 * Mogwai Renderer python API in scripts (docs/usage/scripting.md): getGraph(name),
 * activeGraph, removeGraph(graph or name), unloadScene().
 */

import { initScripting } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { runMogwaiSource } from "../../../packages/mogwai/src/ScriptRunner.js";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Mogwai.rendererGraphApi", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const script = [
        "from falcor import *",
        "a = RenderGraph('A')",
        "a.addPass(createPass('ToneMapper'), 'Tone')",
        "m.addGraph(a)",
        "b = RenderGraph('B')",
        "b.addPass(createPass('ToneMapper'), 'Tone')",
        "m.addGraph(b)",
        "assert m.activeGraph.name == 'A'  # Renderer::addGraph keeps the first graph active",
        "assert m.getGraph('A').name == 'A'",
        "assert m.getGraph('missing') is None",
        "m.loadScene('test_scenes/cornell_box.pyscene')",
        "m.removeGraph('A')",
        "assert m.getGraph('A') is None",
        "m.unloadScene()",
    ].join("\n");
    const replay = await runMogwaiSource(device, script, "/Falcor/media/test_scenes", { download: false });
    expectEq(replay.graphs.map((g) => g.name).join(), "B", "removeGraph by name");
    expectEq(replay.scene, null, "unloadScene");
});
