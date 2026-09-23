/** Transplanted FalcorTest GPU test: Utils/Debug/WarpProfilerTests (native: D3D12 only). */

import { WarpProfiler } from "@web-falcor/falcor";
import { gpuTest, SkipError } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

gpuTest("FalcorTest.WarpProfiler", async ({ device }) => {
    if (!device.hasFeature("subgroups" as GPUFeatureName)) throw new SkipError("WebGPU 'subgroups' feature unavailable");
    const profiler = new WarpProfiler(device, 4);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Utils/Debug/WarpProfilerTests.cs.slang", "main");
    profiler.bindShaderData(ctx.vars());
    profiler.begin(ctx.getRenderContext());
    ctx.runProgram(256, 256, 16); // 2^20 threads = 32768 warps
    profiler.end(ctx.getRenderContext());
    const e = new Expect();
    const sum = (h: number[]) => h.reduce((a, b) => a + b, 0);
    const h0 = await profiler.getWarpHistogram(0);
    e.check(h0.length === 32 && h0[31] === 32768 && sum(h0) === 32768, () => `bin 0: [31]=${h0[31]} total ${sum(h0)}`);
    const h1 = await profiler.getWarpHistogram(1);
    e.check(h1[7] === 16384 && sum(h1) === 16384, () => `bin 1: [7]=${h1[7]} total ${sum(h1)}`);
    const h01 = await profiler.getWarpHistogram(0, 2);
    e.check(h01[7] === 16384 && h01[31] === 32768, () => `bins 0-1: [7]=${h01[7]} [31]=${h01[31]}`);
    const h2 = await profiler.getWarpHistogram(2);
    e.check(h2[3] === 32768 && sum(h2) === 32768, () => `bin 2: [3]=${h2[3]} total ${sum(h2)}`);
    const h3 = await profiler.getWarpHistogram(3);
    e.check(h3[7] === 8192 && sum(h3) === 8192, () => `bin 3: [7]=${h3[7]} total ${sum(h3)}`);
    const csv = await profiler.saveWarpHistogramsAsCSV();
    e.check(csv.split("\n").length === 5 && csv.split("\n")[0]!.split(";").length === 32, () => `csv shape`);
    e.done("WarpProfiler");
});
