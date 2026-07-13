/**
 * Settings scripting surface: python graph scripts configure the global
 * Settings via m.settings (native Mogwai getSettings() binding), including
 * PyProxy dict conversion and filtered attributes.
 */

import { getGlobalSettings, initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("SettingsScripting.pythonBinding", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    getGlobalSettings().clearOptions();
    getGlobalSettings().clearFilteredAttributes();

    const script = `
from falcor import *
m.settings.addOptions({"usdImporter": {"loadMaterialX": False}, "MyPass": {"quality": 3}})
m.settings.addFilteredAttributes({"regex": "Fur.*", "attributes": {"curves": {"lodMode": 1}}})
g = RenderGraph("SettingsTest")
g.addPass(createPass("ToneMapper", {}), "ToneMapper")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
`;
    await runGraphScript(device, script);

    const s = getGlobalSettings();
    expectEq(s.getOption("usdImporter:loadMaterialX", true), false, "nested option flattened");
    expectEq(s.getOption("MyPass:quality", 0), 3, "int option");
    expectEq(s.getAttribute("FurShape", "curves:lodMode", 0), 1, "filtered attribute");
    expectEq(s.getAttribute("Rock", "curves:lodMode", 0), 0, "non-matching shape");
});
