/**
 * Transplanted FalcorTest GPU tests: Core/ConstantBufferTests, Core/ParamBlockCB,
 * Core/TextureArrays.
 */

import { ResourceBindFlags, ResourceFormat, StandaloneParameterBlock } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

gpuTest("FalcorTest.BuiltinConstantBuffer1", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/ConstantBufferTests.cs.slang", "testCbuffer1");
    ctx.allocateStructuredBuffer("result", 3);
    ctx.vars()["CB"]["params1"]["a"] = 1;
    ctx.vars()["CB"]["params1"]["b"] = 3;
    ctx.vars()["CB"]["params1"]["c"] = 5.5;
    ctx.runProgram(1, 1, 1);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    e.check(r[0] === 1 && r[1] === 3 && r[2] === 5.5, () => `result ${Array.from(r)}`);
    e.done("BuiltinConstantBuffer1");
});

gpuTest("FalcorTest.BuiltinConstantBuffer2", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/ConstantBufferTests.cs.slang", "testCbuffer2");
    ctx.allocateStructuredBuffer("result", 3);
    ctx.vars()["params2"]["a"] = 1;
    ctx.vars()["params2"]["b"] = 3;
    ctx.vars()["params2"]["c"] = 5.5;
    ctx.runProgram(1, 1, 1);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    e.check(r[0] === 1 && r[1] === 3 && r[2] === 5.5, () => `result ${Array.from(r)}`);
    e.done("BuiltinConstantBuffer2");
});

gpuTest("FalcorTest.ParamBlockCB", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/ParamBlockCB.cs.slang", "main");
    ctx.allocateStructuredBuffer("result", 1);
    const paramBlock = StandaloneParameterBlock.create(device, ctx.getReflector().getParameterBlock("gParamBlock")!);
    paramBlock.getRootVar()["a"] = 42.1;
    ctx.vars()["gParamBlock"] = paramBlock;
    ctx.runProgram(1, 1, 1);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    e.check(r[0] === Math.fround(42.1), () => `result ${r[0]}`);
    e.done("ParamBlockCB");
});

// Native expects arrays of textures to fail on Vulkan; WGSL has none either (binding_array
// is unavailable), so the web mirrors the Vulkan branch.
gpuTest("FalcorTest.Texture_NestedArrays", async ({ device }) => {
    const e = new Expect();
    for (const bits of [[1, 1, 1], [2, 0, 1], [0, 0, 3]]) {
        let threw = false;
        try {
            const ctx = new GPUUnitTestContext(device);
            ctx.createProgram("Tests/Core/TextureArrays.cs.slang", "testWrite", { BITS_I: bits[0]!, BITS_J: bits[1]!, BITS_K: bits[2]! });
            const tex = device.createTexture2D(16, 16, ResourceFormat.R32Float, 1, 1, new Float32Array(256), ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
            ctx.vars()["tex"][0][0][0] = tex;
            ctx.runProgram(16, 16, 8);
        } catch {
            threw = true;
        }
        e.check(threw, () => `bits ${bits}: expected a failure as on Vulkan`);
    }
    e.done("Texture_NestedArrays");
});
