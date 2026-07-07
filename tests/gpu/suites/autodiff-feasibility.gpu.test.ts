/**
 * M8 FEASIBILITY — Slang automatic differentiation runs on WebGPU. This is
 * the portability gate for WARDiffPathTracer (differentiable rendering): if
 * reverse-mode autodiff (bwd_diff) and forward-mode (fwd_diff) both compile
 * to WGSL and produce the correct gradient on-device, the diff path tracer
 * is portable in principle. f(x)=x^2*k+sin(x); df/dx=2xk+cos(x).
 */

import { ComputePass, Buffer, ResourceBindFlags, MemoryType } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Autodiff.runsOnWebGPU", async ({ device }) => {
    const out = new Buffer(device, {
        size: 8,
        structSize: 4,
        bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
        memoryType: MemoryType.DeviceLocal,
        name: "autodiffOut",
    });
    const pass = ComputePass.create(device, { path: "WebFalcor/Debug/AutodiffProbe.cs.slang" });
    const root = pass.getRootVar();
    root["result"] = out;
    const ctx = device.renderContext;
    pass.execute(ctx, 1, 1);
    const r = new Float32Array((await ctx.readBuffer(out)).buffer);

    // df/dx at x=2, k=3 = 2*2*3 + cos(2) = 12 - 0.4161468 = 11.5838532.
    const expected = 2 * 2 * 3 + Math.cos(2);
    console.error(`# autodiff: bwd=${r[0]!.toFixed(5)} fwd=${r[1]!.toFixed(5)} expected=${expected.toFixed(5)}`);
    expectEq(Math.abs(r[0]! - expected) < 1e-4, true, `bwd_diff gradient ${r[0]}`);
    expectEq(Math.abs(r[1]! - expected) < 1e-4, true, `fwd_diff gradient ${r[1]}`);
});
