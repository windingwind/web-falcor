/** Transplanted FalcorTest GPU tests: Sampling/SampleGeneratorTests. */

import { SAMPLE_GENERATOR_TINY_UNIFORM, SAMPLE_GENERATOR_UNIFORM, SampleGenerator } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kShaderFile = "Tests/Sampling/SampleGeneratorTests.cs.slang";
const kDispatchDim = [64, 64, 16] as const;
const kDimensions = 32;

/** Pearson correlation between elems[i] and elems[i + stride] (double accumulation, as native). */
function correlation(elems: Float32Array, stride: number): number {
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, n = 0;
    for (let i = 0; i + stride < elems.length; i++) {
        const x = elems[i]!;
        const y = elems[i + stride]!;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        n++;
    }
    return (n * sxy - sx * sy) / (Math.sqrt(n * sxx - sx * sx) * Math.sqrt(n * syy - sy * sy));
}

async function testSampleGenerator(ctx: GPUUnitTestContext, type: number, meanError: number, corrThreshold: number, testInstances: boolean): Promise<void> {
    const sg = SampleGenerator.create(ctx.device, type);
    ctx.createProgram(kShaderFile, "test", sg.getDefines());
    sg.bindShaderData(ctx.vars());
    const numSamples = kDispatchDim[0] * kDispatchDim[1] * kDispatchDim[2] * kDimensions;
    ctx.allocateStructuredBuffer("result", numSamples);
    ctx.vars()["CB"]["gDispatchDim"] = [...kDispatchDim];
    ctx.vars()["CB"]["gDimensions"] = kDimensions;
    ctx.runProgram(...kDispatchDim);
    const result = await ctx.readBuffer("result", Float32Array);

    let min = Infinity, max = -Infinity, mean = 0;
    for (const u of result) {
        min = Math.min(min, u);
        max = Math.max(max, u);
        mean += u;
    }
    mean /= numSamples;
    const e = new Expect();
    e.check(min >= 0 && max < 1, () => `range [${min}, ${max}]`);
    e.check(Math.abs(mean - 0.5) <= meanError, () => `mean ${mean}`);
    const corr = (stride: number) => Math.abs(correlation(result, stride));
    for (let i = 1; i <= 8; i++) e.check(corr(i) <= corrThreshold, () => `i = ${i}: ${corr(i)}`);
    const xStride = kDimensions;
    const yStride = kDispatchDim[0] * kDimensions;
    for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++) {
            if (x === 0 && y === 0) continue;
            e.check(corr(x * xStride + y * yStride) <= corrThreshold, () => `x = ${x} y = ${y}: ${corr(x * xStride + y * yStride)}`);
        }
    if (testInstances) {
        const instanceStride = kDispatchDim[0] * kDispatchDim[1] * kDimensions;
        for (let i = 1; i <= 4; i++) e.check(corr(i * instanceStride) <= corrThreshold, () => `instance ${i}: ${corr(i * instanceStride)}`);
    }
    e.done(`SampleGenerator type ${type}`);
}

// Thresholds are native's (tuned to observed correlations at these sample counts).
gpuTest("FalcorTest.SampleGenerator_TinyUniform", async ({ device }) => testSampleGenerator(new GPUUnitTestContext(device), SAMPLE_GENERATOR_TINY_UNIFORM, 0.01, 0.0025, true));
gpuTest("FalcorTest.SampleGenerator_Uniform", async ({ device }) => testSampleGenerator(new GPUUnitTestContext(device), SAMPLE_GENERATOR_UNIFORM, 0.01, 0.002, true));
