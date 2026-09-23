/** Sample launcher: `?sample=<Name>` runs one of Falcor's samples on the web SampleApp, `?testbed=<script>` a Python Testbed script. */

import "@web-falcor/render-passes";
import { Device, initProgramSystem, initScripting, runTestbedScript, type SampleApp, type SampleAppConfig } from "@web-falcor/falcor";
import { SampleAppTemplate } from "./SampleAppTemplate.js";
import { ShaderToy } from "./ShaderToy.js";
import { Visualization2D } from "./Visualization2D.js";
import { MultiSampling } from "./MultiSampling.js";
import { HelloDXR } from "./HelloDXR.js";

type SampleClass = (new (config: SampleAppConfig) => SampleApp) & { config: SampleAppConfig };
export const kSamples: Record<string, SampleClass> = { SampleAppTemplate, ShaderToy, Visualization2D, MultiSampling, HelloDXR };

const params = new URLSearchParams(location.search);
const testbedScript = params.get("testbed");
const name = params.get("sample") ?? (testbedScript ? "" : "ShaderToy");
const picker = document.getElementById("picker")!;
for (const s of Object.keys(kSamples)) {
    const a = document.createElement("a");
    a.href = `?sample=${s}`;
    a.textContent = s === name ? `[${s}]` : s;
    picker.appendChild(a);
}
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
for (const script of ["balls/balls.py"]) {
    const a = document.createElement("a");
    a.href = `?testbed=/Falcor/scripts/python/${script}`;
    a.textContent = testbedScript?.endsWith(script) ? `[testbed: ${script}]` : `testbed: ${script}`;
    picker.appendChild(a);
}
if (testbedScript) {
    // Falcor's Python Testbed scripts (Falcor/scripts/python), run unmodified.
    const device = await Device.create();
    await initProgramSystem(device);
    await initScripting("/node_modules/pyodide");
    await runTestbedScript(device, testbedScript, { canvas });
    console.log(`${testbedScript} finished`);
} else {
    const Sample = kSamples[name];
    if (!Sample) throw new Error(`Unknown sample '${name}'`);
    const app = new Sample({ ...Sample.config, windowDesc: { ...Sample.config.windowDesc, width: undefined, height: undefined }, canvas, uiContainer: document.getElementById("gui")! });
    void app.run().then((code) => console.log(`${name} exited with ${code}`));
}
