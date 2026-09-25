/**
 * Unmodified native image-test scripts (Falcor/tests/image_tests) run through
 * the Mogwai script runner: record, then replay scene loads, frames and
 * captures. Native Mogwai can't run on this host, so the check is structural
 * plus self-consistency: native file names, one capture per configuration,
 * and configurations that must differ produce different images.
 */

import { initScripting } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { runMogwaiScript, runMogwaiSource } from "../../../packages/mogwai/src/ScriptRunner.js";
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i]);

gpuTest("MogwaiScriptRunner.nativeColorMapImageTest", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const { frameCapture } = await runMogwaiScript(device, "/Falcor/tests/image_tests/renderpasses/test_ColorMapPass.py");
    const names = frameCapture.captured.map((f) => f.name);
    console.error(`# script runner: ${names.length} captures: ${names.join(" ")}`);
    const expected = [
        "default.ColorMap.output.1",
        ...["Grey", "Jet", "Viridis", "Plasma", "Magma", "Inferno"].map((c) => `colorMap.ColorMap.${c}.ColorMap.output.1`),
        ...[0, 1, 2, 3].map((c) => `colorMap.channel.${c}.ColorMap.output.1`),
        ...["0.maxValue.1", "1.maxValue.0", "0.25.maxValue.0.75"].map((v) => `minValue.${v}.ColorMap.output.1`),
        "colorMap.autoRange.ColorMap.output.1",
        "colorMap.autoRange.ColorMap.output.2",
    ];
    expectEq(names.join(), expected.map((n) => `${n}.png`).join(), "native capture sequence, naming and (sRGB target) PNG format");
    const byName = (n: string) => frameCapture.captured.find((f) => f.name.startsWith(n))!.bytes;
    expectEq(same(byName("colorMap.ColorMap.Grey."), byName("colorMap.ColorMap.Jet.")), false, "Grey and Jet differ");
    expectEq(same(byName("minValue.0.maxValue.1."), byName("minValue.1.maxValue.0.")), false, "reversed range differs");
    expectEq(same(byName("colorMap.channel.0."), byName("colorMap.channel.1.")), false, "channels differ");
});

// Regressions the runner found: scene switches (PathTracer kept the previous scene's light
// samplers), getDictionary/updatePass round trips, m.scene edits, and scene-local helper imports.
for (const [script, captures] of [["PathTracerReload", 20], ["SimplePostFX", 5], ["VBufferRT", 36], ["PathTracerMaterials", 20]] as const) {
    gpuTest(`MogwaiScriptRunner.native_${script}`, async ({ device }) => {
        await initScripting("/node_modules/pyodide");
        const { frameCapture } = await runMogwaiScript(device, `/Falcor/tests/image_tests/renderpasses/test_${script}.py`);
        expectEq(frameCapture.captured.length, captures, `test_${script}.py capture count`);
    });
}

// Every portable native image test (~13 min): run with --filter NativeImageTestScript.
// Not portable: DLSS/OptiX (vendor SDKs), SDFEditor (interactive pass), WARDiff (compiler-blocked).
const kAllScripts = [
    ..."scene/AnimationBehavior scene/CameraAnimation scene/Displacement scene/NDSDFGrids scene/RtProgram scene/SceneCache scene/SDFSBS scene/SDFSVO scene/SDFSVS scene/TriangleWinding scene/USDPreviewSurface scene/Volumes".split(" "),
    ..."renderscripts/BSDFViewer renderscripts/MinimalPathTracer renderscripts/PathTracer renderscripts/RTXDI renderscripts/SceneDebugger".split(" "),
    ..."BSDFViewer ColorMapPass CompositePass CrossFadePass FLIPPass GaussianBlur GBufferRasterAlpha GBufferRaster GBufferRTInline GBufferRT GBufferRTTexGrads HalfRes MinimalPathTracer ModulateIllumination MVecRaster MVecRT PathTracerAdaptive PathTracerDielectrics PathTracerMaterials PathTracer PathTracerReload RTXDI SideBySide SimplePostFX Skinning SplitScreen SVGF TAA TextureLOD ToneMapping VBufferRasterAlpha VBufferRaster VBufferRTInline VBufferRT".split(" ").map((s) => `renderpasses/${s}`),
];
const selected = new URLSearchParams(location.search).get("filter")?.includes("NativeImageTestScript") ?? false;
for (const script of kAllScripts) {
    gpuTest(`NativeImageTestScript.${script}`, async ({ device }) => {
        if (!selected) throw new SkipError("run with --filter NativeImageTestScript");
        await initScripting("/node_modules/pyodide");
        const [dir, name] = script.split("/");
        const { frameCapture } = await runMogwaiScript(device, `/Falcor/tests/image_tests/${dir}/test_${name}.py`);
        expectEq(frameCapture.captured.length > 0, true, `${script}.py captured ${frameCapture.captured.length} files`);
    });
}

gpuTest("MogwaiScriptRunner.activeGraphAndScript", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = [
        "from falcor import *",
        "a, b, c = RenderGraph('A'), RenderGraph('B'), RenderGraph('C')",
        "m.addGraph(a)",
        "m.addGraph(b)",
        "assert m.activeGraph.name == 'A'  # Renderer::addGraph keeps the active graph",
        "m.setActiveGraph(c)  # adds it",
        "assert m.activeGraph.name == 'C' and m.getGraph('C') is not None",
        "m.removeGraph(a)  # active index steps down, still C",
        "assert m.activeGraph.name == 'C'",
        "m.settings.clearOptions()",
        "open('helper.py', 'w').write('m.resizeSwapChain(64, 32)\\nhelper_ran = True\\n')",
        "m.script('helper.py')",
        "assert helper_ran",
    ].join("\n");
    const { graphs, activeGraph } = await runMogwaiSource(device, source, "/tests/gpu/assets");
    expectEq(graphs.map((g) => g.name).join(), "B,C", "graphs after removeGraph");
    expectEq(activeGraph?.name, "C", "active graph replayed");
});

gpuTest("MogwaiScriptRunner.rendererCallbacks", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = [
        "from falcor import *",
        "m.addGraph(RenderGraph('A'))",
        "m.loadScene('test_scenes/cornell_box.pyscene')",
        "m.renderFrame()  # before the callback is set",
        "times = []",
        "def cb(scene, t):",
        "    times.append(t)",
        "    scene.camera.nearPlane = 0.25",
        "m.sceneUpdateCallback = cb",
        "m.clock.pause()",
        "m.timingCapture.captureFrameTime('times.txt')",
        "for i in range(4): m.renderFrame()",
        "m.timingCapture.captureFrameTime('')",
        "m.renderFrame()",
    ].join("\n");
    const { scene, timingCapture } = await runMogwaiSource(device, source, "/Falcor/media/test_scenes");
    // TimingCapture: one time per frame from the second frame on, none after the capture stops.
    const times = timingCapture.files.get("times.txt") ?? [];
    expectEq(times.length === 4 && times.every((t) => t >= 0 && t < 10), true, `frame times in seconds (${times.join(", ")})`);
    expectEq(scene?.camera.getNearPlane(), 0.25, "callback ran with the scene");
    // Python state lives on in the interpreter: the callback ran once per frame after it was set.
    const { runConsoleCommand } = await import("@web-falcor/falcor");
    expectEq(runConsoleCommand(device, "len(times)", { scene: null, graph: null }), "5", "one call per frame after the callback was set");

    // The console's m keeps callbacks in the viewer's holder.
    const callbacks = { sceneUpdateCallback: null, keyCallback: null } as import("@web-falcor/falcor").MogwaiCallbacks;
    runConsoleCommand(device, "m.keyCallback = lambda pressed, key: pressed and key == 65", { scene: null, graph: null, callbacks });
    expectEq([callbacks.keyCallback?.(true, 65), callbacks.keyCallback?.(true, 66)].join(), "true,false", "keyCallback stored and callable");
    expectEq(runConsoleCommand(device, "m.keyCallback is not None", { scene: null, graph: null, callbacks }), "True", "read back in a later command");
});
