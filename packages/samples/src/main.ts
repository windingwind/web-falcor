/** Sample launcher: `?sample=<Name>` runs one of Falcor's samples on the web SampleApp. */

import "@web-falcor/render-passes";
import type { SampleApp, SampleAppConfig } from "@web-falcor/falcor";
import { SampleAppTemplate } from "./SampleAppTemplate.js";
import { ShaderToy } from "./ShaderToy.js";
import { Visualization2D } from "./Visualization2D.js";
import { MultiSampling } from "./MultiSampling.js";
import { HelloDXR } from "./HelloDXR.js";

type SampleClass = (new (config: SampleAppConfig) => SampleApp) & { config: SampleAppConfig };
export const kSamples: Record<string, SampleClass> = { SampleAppTemplate, ShaderToy, Visualization2D, MultiSampling, HelloDXR };

const name = new URLSearchParams(location.search).get("sample") ?? "ShaderToy";
const picker = document.getElementById("picker")!;
for (const s of Object.keys(kSamples)) {
    const a = document.createElement("a");
    a.href = `?sample=${s}`;
    a.textContent = s === name ? `[${s}]` : s;
    picker.appendChild(a);
}
const Sample = kSamples[name];
if (!Sample) throw new Error(`Unknown sample '${name}'`);
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const app = new Sample({ ...Sample.config, windowDesc: { ...Sample.config.windowDesc, width: undefined, height: undefined }, canvas, uiContainer: document.getElementById("gui")! });
void app.run().then((code) => console.log(`${name} exited with ${code}`));
