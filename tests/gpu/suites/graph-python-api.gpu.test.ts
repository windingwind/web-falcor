/**
 * Native RenderGraph/RenderPass python bindings in plain graph scripts: the snake_case methods
 * (create_pass, add_edge, mark_output, ...), g["pass"], a settable name, pass.properties /
 * getDictionary / set_properties, and m.getSettings().
 */

import { getGlobalSettings, initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Scripting.graphPythonBindings", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const [g] = await runGraphScript(
        device,
        [
            "from falcor import *",
            "g = RenderGraph('First')",
            "g.name = 'Renamed'",
            "g.create_pass('Acc', 'AccumulatePass', {'enabled': False})",
            "g.createPass('Tone', 'ToneMapper', {})",
            "g.add_edge('Acc.output', 'Tone.src')",
            "g.mark_output('Tone.dst')",
            "assert g['Acc'].properties['enabled'] == False",
            "g.get_pass('Acc').set_properties({'enabled': True})",
            "assert g['Acc'].getDictionary()['enabled'] == True",
            "g.update_pass('Tone', {'exposureCompensation': 1.5})",
            "m.getSettings().addOptions({'graphPythonBindingsTest': 7})",
            "m.addOptions({'graphPythonBindingsTestM': 3})",
            "m.addFilteredAttributes({'x': 1}); m.clearFilteredAttributes()",
            "m.addGraph(g)",
        ].join("\n"),
    );
    expectEq(g!.name, "Renamed", "settable name");
    expectEq(g!.getPass("Acc")?.getProperties().toJSON()["enabled"], true, "set_properties reached the pass");
    expectEq(g!.getPass("Tone")?.getProperties().toJSON()["exposureCompensation"], 1.5, "update_pass");
    expectEq(g!.getOutputNames().join(), "Tone.dst", "mark_output");
    expectEq(getGlobalSettings().getOption("graphPythonBindingsTest", 0), 7, "m.getSettings().addOptions");
    expectEq(getGlobalSettings().getOption("graphPythonBindingsTestM", 0), 3, "m.addOptions");
});

// Pre-string/dict render scripts, as tools/fix_render_script.py would rewrite them.
gpuTest("Scripting.legacyRenderScriptNames", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const [g] = await runGraphScript(
        device,
        [
            "from falcor import *",
            "g = RenderGraph('Legacy')",
            "g.create_pass('Tone', 'ToneMapper', {'operator': ToneMapOp.Reinhard})",
            "g.create_pass('GBuf', 'GBufferRaster', {'cull': CullMode.CullFront, 'samplePattern': SamplePattern.Halton})",
            "g.create_pass('PT', 'PathTracer', {'RTXDIOptions': RTXDIOptions(mode='SpatiotemporalResampling', presampledTileCount=64)})",
            "assert ToneMapOp.Aces == 'Aces' and PathTracerParams(samplesPerPixel=2) == {'samplesPerPixel': 2}",
            "g.mark_output('Tone.dst')",
            "m.addGraph(g)",
        ].join("\n"),
    );
    expectEq(g!.getPass("Tone")?.getProperties().toJSON()["operator"], "Reinhard", "ToneMapOp enum value");
    expectEq(g!.getPass("GBuf")?.getProperties().toJSON()["cull"], "Front", "CullMode.CullFront");
    expectEq(g!.getPass("GBuf")?.getProperties().toJSON()["samplePattern"], "Halton", "SamplePattern enum value");
    expectEq(JSON.stringify((g!.getPass("PT")?.getProperties().toJSON()["RTXDIOptions"] as Record<string, unknown>)?.["presampledTileCount"]), "64", "RTXDIOptions struct");
});

// Pass python properties go through native's setters: clamps, derived values, enum names.
gpuTest("Scripting.passPythonProperties", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const [g] = await runGraphScript(
        device,
        [
            "from falcor import *",
            "g = RenderGraph('Props')",
            "g.create_pass('Acc', 'AccumulatePass', {})",
            "g.create_pass('Tone', 'ToneMapper', {})",
            "g.create_pass('TAA', 'TAA', {})",
            "g.create_pass('Blur', 'GaussianBlur', {})",
            "g.create_pass('Blit', 'BlitPass', {})",
            "g.create_pass('FX', 'SimplePostFX', {})",
            "g.create_pass('PT', 'PathTracer', {'fixedSeed': 7})",
            "assert g['Tone'].operator == 'Aces' and g['Blit'].filter == 'Linear' and g['PT'].useFixedSeed and g['PT'].fixedSeed == 7",
            "g['Acc'].enabled = False",
            "g['Tone'].fNumber = 4.0",
            "g['Tone'].exposureCompensation = 100.0",
            "g['Tone'].operator = 'Reinhard'",
            "g['Tone'].exposureMode = 'ShutterPriority'",
            "g['TAA'].sigma = 3.5",
            "g['Blur'].kernelWidth = 8",
            "g['Blit'].filter = 'Point'",
            "g['FX'].colorScale = float3(0.25, 0.5, 0.75)",
            "g['PT'].useFixedSeed = False",
            "g.mark_output('Tone.dst')",
            "m.addGraph(g)",
        ].join("\n"),
    );
    const props = (name: string) => g!.getPass(name)!.getProperties().toJSON() as Record<string, unknown>;
    expectEq(props("Acc")["enabled"], false, "AccumulatePass.enabled");
    const tone = props("Tone");
    expectEq(tone["fNumber"], 4, "ToneMapper.fNumber");
    expectEq(tone["exposureCompensation"], 12, "exposureCompensation clamped to native's max");
    expectEq(tone["operator"], "Reinhard", "ToneMapper.operator by name");
    expectEq(tone["exposureMode"], "ShutterPriority", "ToneMapper.exposureMode by name");
    expectEq((g!.getPass("Tone") as unknown as { exposureValue: number }).exposureValue, 4, "exposureValue resynced (log2(1 * 4 * 4))");
    expectEq(props("TAA")["colorBoxSigma"], 3.5, "TAA.sigma is colorBoxSigma");
    expectEq(props("Blur")["kernelWidth"], 9, "GaussianBlur.kernelWidth forced odd");
    expectEq(props("Blit")["filter"], "Point", "BlitPass.filter");
    expectEq(JSON.stringify(props("FX")["colorScale"]), "[0.25,0.5,0.75]", "SimplePostFX.colorScale from float3");
    expectEq(props("PT")["fixedSeed"], undefined, "PathTracer fixedSeed dropped with useFixedSeed off");
});
