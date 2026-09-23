/**
 * Transplanted FalcorTest GPU tests: Slang/WaveOps, on WebGPU subgroups.
 * WaveMatch (native: D3D12 only) has no WGSL subgroup equivalent.
 */

import { Mt19937, type Device } from "@web-falcor/falcor";
import { gpuTest, SkipError } from "../../harness/registry.js";
import { GPUUnitTestContext, getElements } from "../../harness/unit-test-context.js";
import { Expect, uniformFloat } from "../../harness/expect.js";

const kShaderFilename = "Tests/Slang/WaveOps.cs.slang";
const kNumElems = 32 * 128;

function requireSubgroups(device: Device): void {
    if (!device.hasFeature("subgroups" as GPUFeatureName)) throw new SkipError("WebGPU 'subgroups' feature unavailable");
}

function laneCountBuffer(device: Device) {
    return device.createStructuredBuffer(4, 1, undefined, new Uint32Array(1));
}

async function queryLaneCount(ctx: GPUUnitTestContext): Promise<number> {
    ctx.createProgram(kShaderFilename, "testWaveGetLaneCount");
    const lc = laneCountBuffer(ctx.device);
    ctx.vars()["laneCount"] = lc;
    ctx.runProgram(1, 1, 1);
    return (await getElements(lc, Uint32Array))[0]!;
}

gpuTest("FalcorTest.WaveGetLaneCount", async ({ device }) => {
    requireSubgroups(device);
    const laneCount = await queryLaneCount(new GPUUnitTestContext(device));
    console.error(`# WaveGetLaneCount: ${laneCount}`);
    const e = new Expect();
    e.check(laneCount >= 4 && laneCount <= 128, () => `lane count ${laneCount}`);
    e.done("WaveGetLaneCount");
});

async function testWaveMinMax(device: Device, conditional: boolean): Promise<void> {
    requireSubgroups(device);
    const rng = new Mt19937();
    const u = uniformFloat(rng);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kShaderFilename, "testWaveMinMax", { CONDITIONAL: conditional ? 1 : 0 });
    ctx.allocateStructuredBuffer("result", kNumElems * 2);
    const lc = laneCountBuffer(device);
    ctx.vars()["laneCount"] = lc;
    const testData = new Float32Array(kNumElems);
    for (let i = 0; i < kNumElems; i += 32) {
        const offset = Math.fround(Math.fround(10 * u()) - 5);
        for (let j = 0; j < 32; j++) testData[i + j] = Math.fround(offset + Math.fround(Math.fround(2 * u()) - 1));
    }
    ctx.vars()["testData"] = device.createStructuredBuffer(4, kNumElems, undefined, testData);
    ctx.runProgram(kNumElems, 1, 1);
    const laneCount = (await getElements(lc, Uint32Array))[0]!;
    if (laneCount < 4 || laneCount > 128) throw new Error(`Unsupported wave lane count ${laneCount}`);
    const result = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < kNumElems; i += laneCount) {
        const active = (v: number) => !conditional || v - Math.floor(v) < 0.5;
        let min = Infinity, max = -Infinity;
        for (let j = 0; j < laneCount; j++) if (active(testData[i + j]!)) (min = Math.min(min, testData[i + j]!)), (max = Math.max(max, testData[i + j]!));
        for (let j = 0; j < laneCount; j++) {
            const k = i + j;
            const [wmin, wmax] = active(testData[k]!) ? [min, max] : [0, 0];
            // result is uint4-typed: element 2k.x holds min, 2k+1 .x holds max
            e.check(result[8 * k] === wmin && result[8 * k + 4] === wmax, () => `i = ${k}: ${result[8 * k]}/${result[8 * k + 4]} vs ${wmin}/${wmax}`);
        }
    }
    e.done(`WaveMinMax${conditional ? "Conditional" : ""}`);
}

gpuTest("FalcorTest.WaveMinMax", async ({ device }) => testWaveMinMax(device, false));
// Native disables this one ("compiler issues"); it passes on the WGSL path.
gpuTest("FalcorTest.WaveMinMaxConditional", async ({ device }) => testWaveMinMax(device, true));

async function testWaveMaxSimple(device: Device, entry: string, isFloat: boolean): Promise<void> {
    requireSubgroups(device);
    const ctx = new GPUUnitTestContext(device);
    if ((await queryLaneCount(ctx)) !== 32) throw new SkipError("Test assumes warp size 32");
    const data = isFloat ? Float32Array.from({ length: 32 }, (_, i) => i - 15) : Int32Array.from({ length: 32 }, (_, i) => i - 15);
    ctx.createProgram(kShaderFilename, entry);
    ctx.allocateStructuredBuffer("result", 32);
    ctx.vars()["testData"] = device.createStructuredBuffer(4, 32, undefined, data);
    ctx.runProgram(32, 1, 1);
    const raw = await ctx.readBuffer("result", isFloat ? Float32Array : Int32Array);
    const e = new Expect();
    for (let i = 0; i < 32; i++) {
        const want = data[i]! <= -2 ? -2 : data[i]!;
        e.check(raw[4 * i] === want, () => `i = ${i}: ${raw[4 * i]} vs ${want}`);
    }
    e.done(entry);
}

gpuTest("FalcorTest.WaveMaxSimpleInt", async ({ device }) => testWaveMaxSimple(device, "testWaveMaxSimpleInt", false));
// Native disables this one ("compiler issues"); it passes on the WGSL path.
gpuTest("FalcorTest.WaveMaxSimpleFloat", async ({ device }) => testWaveMaxSimple(device, "testWaveMaxSimpleFloat", true));
