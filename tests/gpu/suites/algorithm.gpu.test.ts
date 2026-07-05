/**
 * M3 algorithm GPU tests: ParallelReduction and PrefixSum host classes driving
 * (overridden) upstream shaders, verified against CPU references. Mirrors
 * FalcorTest's Tests/Utils coverage.
 */

import {
    ParallelReduction,
    ParallelReductionType,
    PrefixSum,
    ResourceFormat,
    ResourceBindFlags,
    MemoryType,
} from "@web-falcor/falcor";
import { gpuTest, expectClose, expectEq } from "../harness/registry.js";

function makeTestTexture(device: any, w: number, h: number): { tex: any; pixels: Float32Array } {
    const pixels = new Float32Array(w * h * 4);
    // Deterministic pseudo-random values.
    let state = 12345;
    const rand = () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return (state >>> 8) / 0x1000000;
    };
    for (let i = 0; i < pixels.length; i++) pixels[i] = rand() * 2 - 0.5;
    const tex = device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, pixels);
    return { tex, pixels };
}

gpuTest("ParallelReduction.sumFloat", async ({ device }) => {
    const w = 100, h = 70; // non-multiple of tile size on purpose
    const { tex, pixels } = makeTestTexture(device, w, h);
    const reduction = new ParallelReduction(device);
    const result = (await reduction.execute(device.renderContext, tex, ParallelReductionType.Sum)) as Float32Array;

    const expected = [0, 0, 0, 0];
    for (let i = 0; i < w * h; i++) for (let c = 0; c < 4; c++) expected[c]! += pixels[i * 4 + c]!;
    for (let c = 0; c < 4; c++) expectClose(result[c]!, expected[c]!, Math.abs(expected[c]!) * 1e-4 + 1e-2, `sum[${c}]`);
    tex.destroy();
});

gpuTest("ParallelReduction.minMaxFloat", async ({ device }) => {
    const w = 64, h = 64;
    const { tex, pixels } = makeTestTexture(device, w, h);
    const reduction = new ParallelReduction(device);
    const result = (await reduction.execute(device.renderContext, tex, ParallelReductionType.MinMax)) as Float32Array;

    const expMin = [Infinity, Infinity, Infinity, Infinity];
    const expMax = [-Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < w * h; i++) {
        for (let c = 0; c < 4; c++) {
            expMin[c] = Math.min(expMin[c]!, pixels[i * 4 + c]!);
            expMax[c] = Math.max(expMax[c]!, pixels[i * 4 + c]!);
        }
    }
    for (let c = 0; c < 4; c++) {
        expectClose(result[c]!, expMin[c]!, 1e-6, `min[${c}]`);
        expectClose(result[4 + c]!, expMax[c]!, 1e-6, `max[${c}]`);
    }
    tex.destroy();
});

gpuTest("ParallelReduction.sumLargeMultiPass", async ({ device }) => {
    // >1024 tiles forces the finalPass ping-pong loop: 2048x1024 -> 64x32=2048 tiles.
    const w = 2048, h = 1024;
    const pixels = new Float32Array(w * h * 4).fill(0.25);
    const tex = device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, pixels);
    const reduction = new ParallelReduction(device);
    const result = (await reduction.execute(device.renderContext, tex, ParallelReductionType.Sum)) as Float32Array;
    const expected = w * h * 0.25;
    for (let c = 0; c < 4; c++) expectClose(result[c]!, expected, expected * 1e-5, `sum[${c}]`);
    tex.destroy();
});

async function runPrefixSumTest(device: any, n: number): Promise<void> {
    const input = new Uint32Array(n);
    let state = 99;
    for (let i = 0; i < n; i++) {
        state = (state * 1664525 + 1013904223) >>> 0;
        input[i] = state % 100;
    }
    const buf = device.createBuffer(n * 4, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, MemoryType.DeviceLocal, input);

    const prefixSum = new PrefixSum(device);
    const total = await prefixSum.execute(device.renderContext, buf, n, true);

    const gpu = new Uint32Array((await buf.getBlob()).buffer, 0, n);
    let running = 0;
    for (let i = 0; i < n; i++) {
        if (gpu[i] !== running) throw new Error(`prefixSum[${i}]: expected ${running}, got ${gpu[i]}`);
        running += input[i]!;
    }
    expectEq(total, running, "total sum");
    buf.destroy();
}

gpuTest("PrefixSum.smallSingleGroup", async ({ device }) => {
    await runPrefixSumTest(device, 1000);
});

gpuTest("PrefixSum.multiGroup", async ({ device }) => {
    await runPrefixSumTest(device, 100_000);
});

gpuTest("PrefixSum.nonPowerOfTwo", async ({ device }) => {
    await runPrefixSumTest(device, 12_347);
});
