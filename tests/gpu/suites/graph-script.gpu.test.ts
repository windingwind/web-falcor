/**
 * M4 exit test: the UNMODIFIED upstream graph script
 * tests/image_tests/renderpasses/graphs/ToneMapping.py executes in the browser
 * via Pyodide (falcor bridge), loads its .hdr asset, and renders.
 */

import { initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("GraphScript.upstreamToneMappingPy", async ({ device }) => {
    // Fetch the untouched upstream script and run it.
    const scriptUrl = "/Falcor/tests/image_tests/renderpasses/graphs/ToneMapping.py";
    const source = await (await fetch(scriptUrl)).text();
    expectEq(source.includes("createPass(\"ToneMapper\")"), true, "fetched upstream script");

    await initScripting("/node_modules/pyodide");
    const graphs = await runGraphScript(device, source);
    expectEq(graphs.length, 1, "one graph registered via m.addGraph");
    const graph = graphs[0]!;
    expectEq(graph.name, "ToneMapper", "graph name from script");

    // Load assets (ImageLoader fetches the .hdr), compile + run.
    await graph.init();
    graph.onResize(256, 128);
    const ctx = device.renderContext;
    graph.execute(ctx);

    // Bisect: the loaded HDR must be non-zero after ImageLoader's blit.
    const loaded = graph.getOutput("ImageLoader.dst")!;
    const hdrPx = new Float32Array((await ctx.readTextureSubresource(loaded)).buffer);
    let hdrSum = 0;
    for (let i = 0; i < hdrPx.length; i += 4) hdrSum += hdrPx[i]!;
    expectEq(hdrSum > 0, true, `ImageLoader output non-zero (sum=${hdrSum.toFixed(2)})`);

    const out = graph.getOutput("BlitPass.dst");
    expectEq(out !== undefined, true, "marked output exists");

    // BlitPass.dst is a graph-default RGBA32Float target; the blit samples the
    // sRGB tone-mapped texture (decoding to linear). The env map is a bright sky:
    // average must be clearly non-zero and bounded.
    const px = new Float32Array((await ctx.readTextureSubresource(out!)).buffer);
    let sum = 0;
    let finite = true;
    for (let i = 0; i < px.length; i += 4) {
        sum += px[i]!;
        finite &&= Number.isFinite(px[i]!);
    }
    const avg = sum / (px.length / 4);
    expectEq(finite, true, "all values finite");
    expectEq(avg > 0.02 && avg <= 1.0, true, `plausible tonemapped luminance (avg=${avg.toFixed(4)})`);
});
