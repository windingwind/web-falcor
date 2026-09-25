/**
 * Transplanted FalcorTest GPU tests: Core/ResourceAliasing and Slang/ShaderModel.
 * WebGPU has no shader models (one WGSL target), so every ShaderModel6_x
 * variant compiles the same kernel. BufferAliasing_ReadWrite is not portable:
 * WebGPU rejects overlapping writable storage bindings in one dispatch.
 */

import { MemoryType, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const N = 32;

gpuTest("FalcorTest.BufferAliasing_Read", async ({ device }) => {
    const buffer = device.createBuffer(N * 4, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, Float32Array.from({ length: N }, (_, i) => i));
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/ResourceAliasing.cs.slang", "testRead");
    ctx.allocateStructuredBuffer("result", N * 3);
    ctx.vars()["bufA1"] = buffer;
    ctx.vars()["bufA2"] = buffer;
    ctx.vars()["bufA3"] = buffer;
    ctx.runProgram(N, 1, 1);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < N; i++) e.check(r[i] === i && r[i + N] === i && r[i + 2 * N] === i, () => `i = ${i}: ${r[i]}, ${r[i + N]}, ${r[i + 2 * N]}`);
    e.done("BufferAliasing_Read");
});

// Disabled natively ("<uint> version fails"); read-only aliasing through three structured views works here.
gpuTest("FalcorTest.BufferAliasing_StructRead", async ({ device }) => {
    const buffer = device.createBuffer(N * 4, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, Float32Array.from({ length: N }, (_, i) => i));
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/ResourceAliasing.cs.slang", "testStructRead");
    ctx.allocateStructuredBuffer("result", N * 3);
    ctx.vars()["bufStruct1"] = buffer;
    ctx.vars()["bufStruct2"] = buffer;
    ctx.vars()["bufStruct3"] = buffer;
    ctx.runProgram(N, 1, 1);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < N; i++) e.check(r[i] === i && r[i + N] === i && r[i + 2 * N] === i, () => `i = ${i}: ${r[i]}, ${r[i + N]}, ${r[i + 2 * N]}`);
    e.done("BufferAliasing_StructRead");
});

for (const sm of ["6_0", "6_1", "6_2", "6_3", "6_4", "6_5"]) {
    gpuTest(`FalcorTest.ShaderModel${sm}`, async ({ device }) => {
        const ctx = new GPUUnitTestContext(device);
        ctx.createProgram("Tests/Slang/ShaderModel.cs.slang", "main");
        ctx.allocateStructuredBuffer("result", 256);
        ctx.runProgram(256, 1, 1);
        const r = await ctx.readBuffer("result", Uint32Array);
        const e = new Expect();
        for (let i = 0; i < 256; i++) e.check(r[i] === 3 * i, () => `i = ${i}: ${r[i]}`);
        e.done(`ShaderModel${sm}`);
    });
}
