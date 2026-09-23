/**
 * Transplanted FalcorTest GPU tests: Scene/Material/HairChiang16Tests, with
 * native's pbrt-v3 reference data (Falcor/data/tests/pbrt_hair_bsdf.dat).
 */

import { SAMPLE_GENERATOR_UNIFORM, SampleGenerator, type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kShaderFile = "Tests/Scene/Material/HairChiang16Tests.cs.slang";
const fr = Math.fround;

/** Native's betaM/betaN grid: float loops from 0.1 in steps of 0.2 below 1. */
function roughnessGrid(): Float32Array {
    const out: number[] = [];
    for (let betaM = fr(0.1); betaM < 1; betaM = fr(betaM + fr(0.2))) for (let betaN = fr(0.1); betaN < 1; betaN = fr(betaN + fr(0.2))) out.push(betaM, betaN);
    return Float32Array.from(out);
}

function program(device: Device, entry: string): GPUUnitTestContext {
    const sg = SampleGenerator.create(device, SAMPLE_GENERATOR_UNIFORM);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kShaderFile, entry, sg.getDefines());
    sg.bindShaderData(ctx.vars());
    return ctx;
}

gpuTest("FalcorTest.HairChiang16_PbrtReference", async ({ device }) => {
    const testCount = 50000;
    const buf = new Float32Array(await (await fetch("/Falcor/data/tests/pbrt_hair_bsdf.dat")).arrayBuffer());
    const e = new Expect();
    e.check(buf.length >= testCount * 17, () => `reference data has ${buf.length} floats`);
    const col = (k: number) => buf.subarray(k * testCount, (k + 1) * testCount);
    const vec3 = (k: number) => {
        const out = new Float32Array(testCount * 4); // float3 at WGSL's 16-byte stride
        for (let i = 0; i < testCount; i++) out.set([buf[k * testCount + i]!, buf[(k + 1) * testCount + i]!, buf[(k + 2) * testCount + i]!], i * 4);
        return out;
    };
    const ctx = program(device, "testPbrtReference");
    ctx.allocateStructuredBuffer("gBetaM", testCount, col(0));
    ctx.allocateStructuredBuffer("gBetaN", testCount, col(1));
    ctx.allocateStructuredBuffer("gAlpha", testCount, col(2));
    ctx.allocateStructuredBuffer("gIoR", testCount, col(3));
    ctx.allocateStructuredBuffer("gSigmaA", testCount, vec3(4));
    ctx.allocateStructuredBuffer("gH", testCount, col(7));
    ctx.allocateStructuredBuffer("gWi", testCount, vec3(8));
    ctx.allocateStructuredBuffer("gWo", testCount, vec3(11));
    ctx.allocateStructuredBuffer("gResultOurs", testCount);
    ctx.vars()["TestCB"]["resultSize"] = testCount;
    ctx.runProgram(testCount);
    const result = await ctx.readBuffer("gResultOurs", Float32Array);
    const rs = ctx.getStructSize("gResultOurs") / 4;
    for (let i = 0; i < testCount; i++)
        for (let c = 0; c < 3; c++) {
            const ref = buf[(14 + c) * testCount + i]!;
            const got = result[i * rs + c]!;
            const rel = Math.abs(ref) < 1e-6 ? 0 : Math.abs(got - ref) / ref;
            e.check(rel <= 1e-3, () => `PbrtReferenceTestCase(${i}, ${c}): expected ${ref}, got ${got}`);
        }
    e.done("HairChiang16_PbrtReference");
});

async function testWhiteFurnace(device: Device, entry: string, threshold: number): Promise<void> {
    const roughness = roughnessGrid();
    const testCount = roughness.length / 2;
    const ctx = program(device, entry);
    ctx.allocateStructuredBuffer("roughness", testCount, roughness);
    ctx.allocateStructuredBuffer("result", testCount);
    ctx.vars()["TestCB"]["resultSize"] = testCount;
    ctx.vars()["TestCB"]["sampleCount"] = 300000;
    ctx.runProgram(testCount);
    const result = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < testCount; i++) e.check(Math.abs(result[i]! - 1) <= threshold, () => `WhiteFurnaceTestCase${i}: expected 1, got ${result[i]}`);
    e.done(entry);
}

gpuTest("FalcorTest.HairChiang16_WhiteFurnaceUniform", async ({ device }) => testWhiteFurnace(device, "testWhiteFurnaceUniform", 0.05));
gpuTest("FalcorTest.HairChiang16_WhiteFurnaceImportanceSampling", async ({ device }) => testWhiteFurnace(device, "testWhiteFurnaceImportanceSampling", 0.01));

gpuTest("FalcorTest.HairChiang16_ImportanceSamplingWeights", async ({ device }) => {
    const sampleCount = 10000;
    const roughness = roughnessGrid();
    const testCount = roughness.length / 2;
    const ctx = program(device, "testImportanceSamplingWeights");
    ctx.allocateStructuredBuffer("roughness", testCount, roughness);
    ctx.allocateStructuredBuffer("result", testCount * sampleCount);
    ctx.vars()["TestCB"]["resultSize"] = testCount;
    ctx.vars()["TestCB"]["sampleCount"] = sampleCount;
    ctx.runProgram(testCount, sampleCount);
    const result = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    let failedSampling = 0;
    for (let idx = 0; idx < testCount * sampleCount; idx++) {
        if (Math.abs(result[idx]! + 1) < 1e-6) {
            failedSampling++; // native: "Importance sampling failed. Ignore this test case."
            continue;
        }
        e.check(Math.abs(result[idx]! - 1) <= 1e-3, () => `ImportanceSamplingWeightsTestCase(${Math.floor(idx / sampleCount)}, ${idx % sampleCount}): got ${result[idx]}`);
    }
    console.error(`# HairChiang16_ImportanceSamplingWeights: ${failedSampling} samples ignored as failed sampling`);
    e.done("HairChiang16_ImportanceSamplingWeights");
});

gpuTest("FalcorTest.HairChiang16_SamplingConsistency", async ({ device }) => {
    const roughness = roughnessGrid();
    const testCount = roughness.length / 2;
    const ctx = program(device, "testSamplingConsistency");
    ctx.allocateStructuredBuffer("roughness", testCount, roughness);
    ctx.allocateStructuredBuffer("result", testCount);
    ctx.vars()["TestCB"]["resultSize"] = testCount;
    ctx.vars()["TestCB"]["sampleCount"] = 300000;
    ctx.runProgram(testCount);
    const result = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < testCount; i++) e.check(result[i]! <= 0.05, () => `SamplingConsistencyTestCase${i}: expected 0, got ${result[i]}`);
    e.done("HairChiang16_SamplingConsistency");
});
