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
            "m.addGraph(g)",
        ].join("\n"),
    );
    expectEq(g!.name, "Renamed", "settable name");
    expectEq(g!.getPass("Acc")?.getProperties().toJSON()["enabled"], true, "set_properties reached the pass");
    expectEq(g!.getPass("Tone")?.getProperties().toJSON()["exposureCompensation"], 1.5, "update_pass");
    expectEq(g!.getOutputNames().join(), "Tone.dst", "mark_output");
    expectEq(getGlobalSettings().getOption("graphPythonBindingsTest", 0), 7, "m.getSettings().addOptions");
});
