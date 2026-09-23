/**
 * Unmodified native image-test scripts (Falcor/tests/image_tests) run through
 * the Mogwai script runner: record, then replay scene loads, frames and
 * captures. Native Mogwai can't run on this host, so the check is structural
 * plus self-consistency: native file names, one capture per configuration,
 * and configurations that must differ produce different images.
 */

import { initScripting } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { runMogwaiScript } from "../../../packages/mogwai/src/ScriptRunner.js";
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
