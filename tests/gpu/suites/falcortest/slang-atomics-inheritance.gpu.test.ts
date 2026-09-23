/**
 * Transplanted FalcorTest GPU tests: Slang/Atomics (the fp32 buffer case;
 * WGSL has no fp16 or texture atomics) and Slang/SlangInheritance (layout).
 */

import { MemoryType, Mt19937, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext, getElements } from "../../harness/unit-test-context.js";
import { Expect, uniformFloat } from "../../harness/expect.js";

gpuTest("FalcorTest.Atomics_Buffer_InterlockedAddF32", async ({ device }) => {
    const kNumElems = 256;
    const u = uniformFloat(new Mt19937());
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/Atomics.cs.slang", "testBufferAddF32");
    const elems = Float32Array.from({ length: kNumElems }, u);
    const dataBuf = device.createStructuredBuffer(4, kNumElems, ResourceBindFlags.ShaderResource, elems);
    const resultBuf = device.createStructuredBuffer(4, 2, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, new Float32Array(2));
    ctx.vars()["data"] = dataBuf;
    ctx.vars()["resultBuf"] = resultBuf;
    ctx.runProgram(kNumElems, 1, 1);
    const result = await getElements(resultBuf, Float32Array);
    let a = 0, b = 0;
    for (const v of elems) {
        a = Math.fround(a + v);
        b = Math.fround(b - v);
    }
    const e = new Expect();
    e.check(Math.abs(result[0]! - a) <= 1e-3, () => `sum ${result[0]} vs ${a}`);
    e.check(Math.abs(result[1]! - b) <= 1e-3, () => `negated sum ${result[1]} vs ${b}`);
    e.done("Atomics_Buffer_InterlockedAddF32");
});

gpuTest("FalcorTest.SlangStructInheritanceLayout", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/SlangInheritance.cs.slang", "main");
    // B : A { uint scalar; float3 vector; } — WGSL aligns the float3 to 16 bytes (native packs it at 4).
    const stride = ctx.getStructSize("result");
    const result = device.createStructuredBuffer(stride, 1);
    ctx.vars()["result"] = result;
    const asuint = (f: number) => new Uint32Array(Float32Array.of(f).buffer)[0]!;
    const initData = Uint32Array.of(59941431, asuint(3.13), asuint(5.11), asuint(7.99));
    ctx.vars()["data"] = device.createBuffer(16, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, initData);
    ctx.runProgram(1);
    const r = await getElements(result, Uint32Array);
    const vectorOffset = stride === 16 ? 1 : 4;
    const e = new Expect();
    e.check(r[0] === initData[0], () => `scalar ${r[0]}`);
    for (let c = 0; c < 3; c++) e.check(r[vectorOffset + c] === initData[1 + c], () => `vector[${c}] ${r[vectorOffset + c]}`);
    console.error(`# SlangStructInheritanceLayout: B is ${stride} bytes, vector at offset ${vectorOffset * 4}`);
    e.done("SlangStructInheritanceLayout");
});
