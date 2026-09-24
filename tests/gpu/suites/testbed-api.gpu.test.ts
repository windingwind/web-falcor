/**
 * Testbed python API beyond frames: load_scene_from_string, get_import_paths/dicts,
 * load_render_graph, window, window_size_change_callback, keyboard_event_callback
 * (tests/gpu/fixtures/testbed-api.py; the key event is dispatched here, as the window would).
 */

import { KeyboardEventType, ModifierFlags, initScripting, runTestbedScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Testbed.pythonApi", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const result = await runTestbedScript(device, "/tests/gpu/assets/testbed-api.py", { maxFrames: 10 });
    const testbed = result.testbeds[0]!;
    expectEq(testbed.renderGraph?.getPass("ToneMapping") !== undefined, true, "load_render_graph set the graph");
    expectEq(testbed.showUI, true, "UI shown before the key");
    testbed.handleKeyboardEvent({ type: KeyboardEventType.KeyPressed, key: "E", mods: ModifierFlags.None, codepoint: 0 });
    expectEq(testbed.showUI, false, "the python keyboard callback ran");
});
