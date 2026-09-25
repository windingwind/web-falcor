/**
 * Transplanted FalcorTest GPU tests for Slang language features, compiled to
 * WGSL: Slang/SlangMutatingTests, SlangExtension, SlangGenerics, NestedStructs, SlangTests (SlangEnum,
 * SlangDefaultInitializers without its double case, SlangHashedStrings; the rest of that file needs
 * 16/64-bit scalar types), Slang/TraceRayFlags (the RAY_FLAG_* constants against the host RayFlags).
 */

import { RayFlags, type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const asuint = (f: number) => new Uint32Array(Float32Array.of(f).buffer)[0]!;

gpuTest("FalcorTest.SlangMutating", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/SlangMutatingTests.cs.slang", "main");
    ctx.allocateStructuredBuffer("result", 1);
    ctx.vars()["buffer"] = device.createStructuredBuffer(16, 1, undefined, Uint32Array.of(11, 22, 33, 44));
    ctx.runProgram(1);
    const result = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    e.check(result[0] === 33, () => `result ${result[0]}`);
    e.done("SlangMutating");
});

gpuTest("FalcorTest.Slang_Extension", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/SlangExtension.cs.slang", "main");
    ctx.allocateStructuredBuffer("result", 6);
    ctx.runProgram(1, 1, 1);
    const result = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    for (let i = 0; i < 6; i++) e.check(result[i] === 2, () => `result[${i}] = ${result[i]}`);
    e.done("Slang_Extension");
});

async function runGenerics(device: Device, entry: string, defines: Record<string, number>): Promise<void> {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/SlangGenerics.cs.slang", entry, defines);
    ctx.allocateStructuredBuffer("result", 128);
    ctx.runProgram(32, 1, 1);
    const result = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    for (let i = 0; i < 32; i++) for (let k = 0; k < 4; k++) e.check(result[4 * i + k] === (i + k) * 12, () => `result[${4 * i + k}] = ${result[4 * i + k]}`);
    e.done(entry);
}

gpuTest("FalcorTest.Slang_GenericsInterface_Int", async ({ device }) => runGenerics(device, "testGenericsInterface", { TEST_A: 1, USE_INT: 1 }));
gpuTest("FalcorTest.Slang_GenericsInterface_UInt", async ({ device }) => runGenerics(device, "testGenericsInterface", { TEST_A: 1 }));
gpuTest("FalcorTest.Slang_GenericsFunction_Int", async ({ device }) => runGenerics(device, "testGenericsFunction", { TEST_B: 1, USE_INT: 1 }));
gpuTest("FalcorTest.Slang_GenericsFunction_UInt", async ({ device }) => runGenerics(device, "testGenericsFunction", { TEST_B: 1 }));

gpuTest("FalcorTest.NestedStructs", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/NestedStructs.cs.slang", "main");
    ctx.allocateStructuredBuffer("result", 27);
    const v = ctx.vars()["CB"];
    v["a"] = 1.1;
    v["s3"]["a"] = 17;
    v["s3"]["b"] = true;
    v["s3"]["s2"]["a"] = [true, false, true];
    v["s3"]["s2"]["s1"]["a"] = [9.3, 2.1];
    v["s3"]["s2"]["s1"]["b"] = 23;
    v["s3"]["s2"]["b"] = 0.99;
    v["s3"]["s2"]["c"] = [4, 8];
    v["s3"]["c"] = [0.1, 0.2, 0.3];
    v["s3"]["s1"]["a"] = [1.88, 1.99];
    v["s3"]["s1"]["b"] = 711;
    v["s2"]["a"] = [false, true, false];
    v["s2"]["s1"]["a"] = [0.55, 8.31];
    v["s2"]["s1"]["b"] = 431;
    v["s2"]["b"] = 1.65;
    v["s2"]["c"] = [7, 3];
    ctx.runProgram(1);
    const result = await ctx.readBuffer("result", Uint32Array);
    // prettier-ignore
    const want = [asuint(1.1), 17, 1, 1, 0, 1, asuint(9.3), asuint(2.1), 23, asuint(0.99), 4, 8, asuint(0.1), asuint(0.2), asuint(0.3),
        asuint(1.88), asuint(1.99), 711, 0, 1, 0, asuint(0.55), asuint(8.31), 431, asuint(1.65), 7, 3];
    const e = new Expect();
    want.forEach((w, i) => e.check(result[i] === w, () => `result[${i}] = ${result[i]}, expected ${w}`));
    e.done("NestedStructs");
});

gpuTest("FalcorTest.SlangEnum", async ({ device }) => {
    // Values of Tests/Slang/SlangShared.slang's Type1/Type2/Type3 (the C++ side of native's check).
    const want = [0, 1, 2, 3, 0, 1, 20, 21, 1, 2, 4, 8];
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/SlangTests.cs.slang", "testEnum");
    ctx.allocateStructuredBuffer("result", 12);
    ctx.runProgram(1, 1, 1);
    const result = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    want.forEach((w, i) => e.check(result[i] === w, () => `result[${i}] = ${result[i]}, expected ${w}`));
    e.done("SlangEnum");
});

gpuTest("FalcorTest.SlangDefaultInitializers", async ({ device }) => {
    const maxTests = 100;
    const usedTests = 43;
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/SlangTests.cs.slang", "testDefaultInitializers");
    ctx.allocateStructuredBuffer("result", maxTests, new Uint32Array(maxTests).fill(0xffffffff));
    ctx.runProgram(1, 1, 1);
    const result = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    for (let i = 0; i < maxTests; i++) {
        // i = 9..14 held the double case (not in WGSL), so they keep the init value.
        const expected = i === 42 ? 4 /* Type3::C */ : i < usedTests && !(i >= 9 && i < 15) ? 0 : 0xffffffff;
        e.check(result[i] === expected, () => `i = ${i}: ${result[i]} != ${expected}`);
    }
    e.done("SlangDefaultInitializers");
});

gpuTest("FalcorTest.SlangHashedStrings", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/SlangTests.cs.slang", "testHashedStrings");
    ctx.allocateStructuredBuffer("result", 4);
    ctx.runProgram(1, 1, 1);
    const hashedStrings = ctx.getReflector().getHashedStrings();
    const e = new Expect();
    e.check(hashedStrings.length === 4, () => `${hashedStrings.length} hashed strings`);
    for (let i = 0; i < 4; i++) e.check(hashedStrings[i]?.string === `Test String ${i}`, () => `string ${i}: ${hashedStrings[i]?.string}`);
    const result = await ctx.readBuffer("result", Uint32Array);
    for (let i = 0; i < 4; i++) e.check(result[i] === hashedStrings[i]?.hash, () => `hash ${i}: ${result[i]} != ${hashedStrings[i]?.hash}`);
    e.done("SlangHashedStrings");
});

// Slang/TraceRayFlags.cpp
for (const [name, dxr11] of [["TraceRayFlagsDXR1_0", false], ["TraceRayFlagsDXR1_1", true]] as const) {
    gpuTest(`FalcorTest.${name}`, async ({ device }) => {
        const expected = [RayFlags.None, RayFlags.ForceOpaque, RayFlags.ForceNonOpaque, RayFlags.AcceptFirstHitAndEndSearch, RayFlags.SkipClosestHitShader,
            RayFlags.CullBackFacingTriangles, RayFlags.CullFrontFacingTriangles, RayFlags.CullOpaque, RayFlags.CullNonOpaque];
        if (dxr11) expected.push(RayFlags.SkipTriangles, RayFlags.SkipProceduralPrimitives);
        const ctx = new GPUUnitTestContext(device);
        ctx.createProgram("Tests/Slang/TraceRayFlags.cs.slang", "testRayFlags", dxr11 ? { DXR_1_1: 1 } : {});
        ctx.allocateStructuredBuffer("result", expected.length);
        ctx.runProgram(1, 1, 1);
        const result = await ctx.readBuffer("result", Uint32Array);
        const e = new Expect();
        expected.forEach((v, i) => e.check(result[i] === v, () => `result[${i}] = ${result[i]}, expected ${v}`));
        e.done(name);
    });
}
