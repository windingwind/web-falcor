/**
 * EXR decode integration: ImageLoader loads a native-written EXR capture
 * through a graph; the GPU readback must match the CPU decode exactly
 * (1:1 blit at texel centers).
 */

import { initScripting, runGraphScript, decodeExr } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const kExr = "tests/oracle/out-native/oracle-imgtest-mpt.ToneMapper.dst.0.exr";

gpuTest("ExrLoad.imageLoaderMatchesCpuDecode", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const graphSource = `
from falcor import *
g = RenderGraph("ExrLoad")
g.addPass(createPass("ImageLoader", {'filename': '../../${kExr}', 'mips': False, 'srgb': False}), "ImageLoader")
g.markOutput("ImageLoader.dst")
m.addGraph(g)
`;
    const [graph] = await runGraphScript(device, graphSource);
    await graph!.init();
    graph!.onResize(size, size);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("ImageLoader.dst")!)).buffer);
    const cpu = decodeExr(await (await fetch(`/${kExr}`)).arrayBuffer());
    expectEq(cpu.width, size, "decoded width");

    let maxD = 0;
    for (let i = 0; i < size * size * 4; i++) maxD = Math.max(maxD, Math.abs(web[i]! - cpu.data[i]!));
    console.error(`# exrLoad: maxDiff=${maxD.toExponential(2)}`);
    expectEq(maxD < 1e-6, true, `GPU readback matches CPU decode (maxDiff ${maxD})`);
});
