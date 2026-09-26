/**
 * Every render pass's property dictionary against native getDictionary() (tests/oracle/pass-properties.py):
 * the same keys in the same order and the same values (floats compared at f32 precision).
 */

import { Properties, RenderGraph, ResourceBindFlags, ResourceFormat, createPass, float16ToFloat32 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { expectEq, gpuTest } from "../harness/registry.js";
import { Expect } from "../harness/expect.js";

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };

function diff(path: string, web: unknown, nat: Json, out: string[]): void {
    if (typeof nat === "number" && typeof web === "number") {
        if (web !== nat && Math.fround(web) !== Math.fround(nat)) out.push(`${path}: ${web} vs ${nat}`);
    } else if (Array.isArray(nat)) {
        const w = Array.isArray(web) ? web : web && typeof web === "object" ? ["x", "y", "z", "w"].slice(0, nat.length).map((c) => (web as Record<string, unknown>)[c]) : null;
        if (!w || w.length !== nat.length) out.push(`${path}: ${JSON.stringify(web)} vs ${JSON.stringify(nat)}`);
        else nat.forEach((v, i) => diff(`${path}[${i}]`, w[i], v, out));
    } else if (nat && typeof nat === "object") {
        if (!web || typeof web !== "object") return void out.push(`${path}: ${JSON.stringify(web)} vs object`);
        const wk = Object.keys(web), nk = Object.keys(nat);
        if (wk.join() !== nk.join()) out.push(`${path}: keys [${wk.join(", ")}] vs [${nk.join(", ")}]`);
        for (const k of nk) if (k in web) diff(`${path}.${k}`, (web as Record<string, unknown>)[k], nat[k]!, out);
    } else if (web !== nat) out.push(`${path}: ${JSON.stringify(web)} vs ${JSON.stringify(nat)}`);
}

gpuTest("PassProperties.matchNativeDictionaries", async ({ device }) => {
    const oracle = (await (await fetch("/tests/oracle/out-native/pass-properties.json")).json()) as Record<string, { props: Record<string, never>; dict?: Json; error?: string }>;
    const e = new Expect();
    for (const [key, { props, dict }] of Object.entries(oracle)) {
        if (dict === undefined) continue; // native can't create it here (ROVs, NRD plugin)
        const name = key.split("#")[0]!;
        const out: string[] = [];
        diff(key, createPass(device, name, new Properties(props)).getProperties().toJSON(), dict, out);
        // useRealMonitorInfo: native under Xvfb reports no monitor, the browser its screen width.
        if (props["useRealMonitorInfo"]) out.splice(0, out.length, ...out.filter((l) => !l.includes(".monitorWidthPixels:")));
        for (const line of out) console.error(`# ${line}`);
        e.check(out.length === 0, () => out.join("; "));
    }
    e.done("pass properties");
});

// AccumulatePass 'outputFormat' (native: any format; here the storage-capable float ones).
gpuTest("AccumulatePass.outputFormat", async ({ device }) => {
    const graph = new RenderGraph(device, "Acc");
    graph.addPass(createPass(device, "AccumulatePass", new Properties({ outputFormat: "RGBA16Float" })), "Acc");
    graph.markOutput("Acc.output");
    const src = device.createTexture2D(4, 4, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(64).fill(0.5), ResourceBindFlags.ShaderResource);
    graph.setInput("Acc.input", src);
    await graph.init();
    graph.onResize(4, 4);
    graph.execute(device.renderContext);
    graph.execute(device.renderContext);
    const out = graph.getOutput("Acc.output")!;
    expectEq(ResourceFormat[out.format], "RGBA16Float", "output format");
    const half = new Uint16Array((await device.renderContext.readTextureSubresource(out)).buffer);
    expectEq(float16ToFloat32(half[0]!), 0.5, "accumulated value");
    expectEq(graph.getPass("Acc")!.getProperties().toJSON()["outputFormat"], "RGBA16Float", "outputFormat round-trips");
});
