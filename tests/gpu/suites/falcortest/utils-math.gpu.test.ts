/**
 * Transplanted FalcorTest GPU tests (Source/Tools/FalcorTest/Tests): the
 * native kernels and assertions, run through GPUUnitTestContext.
 * Utils/HashUtilsTests, Utils/BitTricksTests, Utils/MathHelpersTests,
 * Utils/PackedFormatsTests, Sampling/LowDiscrepancyTests, Sampling/PseudorandomTests.
 */

import { Mt19937, ResourceBindFlags, MemoryType, canonicalFloat } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../../harness/registry.js";
import { GPUUnitTestContext, getElements } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

// Utils/HashUtilsTests.cpp

function jenkinsHash(a: number): number {
    a = (a + 0x7ed55d16 + (a << 12)) >>> 0;
    a = (a ^ 0xc761c23c ^ (a >>> 19)) >>> 0;
    a = (a + 0x165667b1 + (a << 5)) >>> 0;
    a = ((a + 0xd3a2646c) ^ (a << 9)) >>> 0;
    a = (a + 0xfd7046c5 + (a << 3)) >>> 0;
    a = (a ^ 0xb55a4f09 ^ (a >>> 16)) >>> 0;
    return a;
}

gpuTest("FalcorTest.JenkinsHash_CompareToCPU", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    // Native uses a typed buffer; WGSL has none (kernel override declares a structured buffer).
    const result = device.createStructuredBuffer(4, 1 << 16);
    ctx.getRenderContext().clearBuffer(result);
    ctx.createProgram("Tests/Utils/HashUtilsTests.cs.slang", "testJenkinsHash");
    ctx.vars()["result"] = result;
    ctx.runProgram(1 << 16, 1, 1);
    const r = await getElements(result, Uint32Array);
    const e = new Expect();
    for (let i = 0; i < r.length; i++) e.check(r[i] === jenkinsHash(i), () => `i = ${i}: ${r[i]} != ${jenkinsHash(i)}`);
    e.done("JenkinsHash");
});

gpuTest("FalcorTest.JenkinsHash_PerfectHashGPU", async ({ device }) => {
    // Native binds one 2^27-dword bitfield; here it is split by the hash's top bits to fit binding limits.
    const lim = device.gpuDevice.limits;
    const maxBytes = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize, 128 << 20);
    let chunkBits = 0;
    while ((1 << (29 - chunkBits)) > maxBytes) chunkBits++;
    const ctx = new GPUUnitTestContext(device);
    const result = device.createStructuredBuffer(4, 1 << (27 - chunkBits));
    ctx.createProgram("Tests/Utils/HashUtilsTests.cs.slang", "testJenkinsHash_PerfectHash");
    const e = new Expect();
    for (let chunk = 0; chunk < 1 << chunkBits; chunk++) {
        ctx.getRenderContext().clearBuffer(result);
        ctx.vars()["result"] = result;
        ctx.vars()["CB"]["chunk"] = chunk;
        ctx.vars()["CB"]["chunkBits"] = chunkBits;
        ctx.runProgram(1 << 16, 1 << 16, 1);
        const r = await getElements(result, Uint32Array);
        let bad = -1;
        for (let i = 0; i < r.length; i++) if (r[i] !== 0xffffffff) { bad = i; break; }
        e.check(bad < 0, () => `chunk ${chunk}: result[${bad}] = ${r[bad]!.toString(16)}`);
    }
    e.done("JenkinsHash_PerfectHashGPU");
});

// Utils/BitTricksTests.cpp

function referenceBitInterleave(x: number, y: number, m: number): number {
    let result = 0;
    for (let i = 0; i < m; i++) {
        result |= ((x >>> i) & 1) << (2 * i);
        result |= ((y >>> i) & 1) << (2 * i + 1);
    }
    return result >>> 0;
}

gpuTest("FalcorTest.BitInterleave", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    const tests = 5;
    const n = 1 << 16;
    const e = new Expect();
    e.check(referenceBitInterleave(0xe38e, 0xbe8b, 16) === 0xdeadc0de, () => "reference 16");
    e.check(referenceBitInterleave(0xe38e, 0xbe8b, 12) === 0x00adc0de, () => "reference 12");
    const rng = new Mt19937();
    const testData = Uint32Array.from({ length: n }, () => rng.next());
    const testDataBuffer = device.createBuffer(n * 4, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, testData);
    ctx.createProgram("Tests/Utils/BitTricksTests.cs.slang", "testBitInterleave");
    ctx.allocateStructuredBuffer("result", n * tests);
    ctx.vars()["testData"] = testDataBuffer;
    ctx.runProgram(n);
    const result = await ctx.readBuffer("result", Uint32Array);
    for (let i = 0; i < n; i++) {
        const bits = testData[i]!;
        const interleavedBits = referenceBitInterleave(bits, bits >>> 16, 16);
        const want = [interleavedBits, interleavedBits & 0xffff, bits & 0x00ff00ff, bits & 0x000f000f, bits & 0x0f0f0f0f].map((v) => v >>> 0);
        for (let k = 0; k < tests; k++) e.check(result[tests * i + k] === want[k], () => `i = ${i} test ${k}: ${result[tests * i + k]} != ${want[k]}`);
    }
    e.done("BitInterleave");
});

// Utils/MathHelpersTests.cpp

/** Double-precision erf (std::erf reference): Taylor series near 0, continued fraction for erfc beyond. */
function erf(x: number): number {
    const ax = Math.abs(x);
    if (ax < 2.5) {
        let term = ax;
        let sum = ax;
        for (let n = 1; n < 100; n++) {
            term *= (-ax * ax) / n;
            sum += term / (2 * n + 1);
        }
        return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum;
    }
    let f = ax;
    for (let k = 60; k >= 1; k--) f = ax + k / 2 / f;
    return Math.sign(x) * (1 - Math.exp(-ax * ax) / Math.sqrt(Math.PI) / f);
}

for (const [name, entry] of [["MathHelpers_SphericalCoordinates", "testSphericalCoordinates"], ["MathHelpers_SphericalCoordinatesRad", "testSphericalCoordinatesRad"]] as const) {
    gpuTest(`FalcorTest.${name}`, async ({ device }) => {
        const ctx = new GPUUnitTestContext(device);
        ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", entry);
        const n = 1024 * 1024;
        ctx.allocateStructuredBuffer("result", n);
        ctx.runProgram(n);
        const r = await ctx.readBuffer("result", Float32Array);
        const e = new Expect();
        for (let i = 0; i < n; i++) e.check(r[i]! > 0.999 && r[i]! < 1.001, () => `i = ${i}: ${r[i]}`);
        e.done(name);
    });
}

gpuTest("FalcorTest.MathHelpers_ErrorFunction", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testErrorFunction");
    const n = 25;
    const input = Float32Array.from({ length: n }, (_, i) => Math.fround(-5 + (5 - -5) * Math.fround(i / (n - 1))));
    const ref = Array.from(input, (x) => Math.fround(erf(x)));
    ctx.allocateStructuredBuffer("result", n);
    ctx.allocateStructuredBuffer("input", n, input);
    ctx.runProgram(n);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < n; i++) e.check(r[i]! >= ref[i]! - 1e-6 && r[i]! <= ref[i]! + 1e-6, () => `i = ${i}: ${r[i]} vs ${ref[i]}`);
    e.done("ErrorFunction");
});

gpuTest("FalcorTest.MathHelpers_InverseErrorFunction", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testInverseErrorFunction");
    const n = 25;
    const input = Float32Array.from({ length: n }, (_, i) => Math.fround(-1 + 2 * Math.fround(i / (n - 1))));
    ctx.allocateStructuredBuffer("result", n);
    ctx.allocateStructuredBuffer("input", n, input);
    ctx.runProgram(n);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < n; i++) {
        const v = Math.fround(erf(r[i]!));
        e.check(v >= input[i]! - 1e-6 && v <= input[i]! + 1e-6, () => `i = ${i}: erf(${r[i]}) = ${v} vs ${input[i]}`);
    }
    e.done("InverseErrorFunction");
});

// Utils/PackedFormatsTests.cpp

gpuTest("FalcorTest.LogLuvHDR", async ({ device }) => {
    const testData: number[][] = [
        [0, 0, 0],
        [1e-30, 1e-30, 1e-30],
        [1e-10, 1e-10, 1e-10],
        [1e10, 1e10, 1e10],
        [1e30, 1e30, 1e30],
    ].map((c) => c.map(Math.fround));
    const rng = new Mt19937();
    const u = () => canonicalFloat(rng);
    for (let i = 0; i < 10000; i++) {
        const scale = Math.fround(Math.pow(2, Math.fround(Math.fround(u() * 40) - 20)));
        const c = [u(), u(), u()];
        testData.push(c.map((v) => Math.fround(v * scale)));
    }
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Utils/PackedFormatsTests.cs.slang", "testLogLuvHDR");
    // float3 elements are 16 bytes in WGSL storage buffers (native: 12).
    const stride = ctx.getStructSize("testData") / 4;
    const packed = new Float32Array(testData.length * stride);
    testData.forEach((c, i) => packed.set(c, i * stride));
    ctx.allocateStructuredBuffer("testData", testData.length, packed);
    ctx.allocateStructuredBuffer("result", testData.length);
    ctx.runProgram(testData.length);
    const raw = await ctx.readBuffer("result", Float32Array);
    const rs = ctx.getStructSize("result") / 4;
    const result = testData.map((_, i) => [raw[i * rs]!, raw[i * rs + 1]!, raw[i * rs + 2]!]);
    const e = new Expect();
    for (let i = 0; i < 3; i++) e.check(result[i]!.every((v) => v === 0), () => `i = ${i}: ${result[i]}`);
    for (let i = 3; i < 5; i++) e.check(result[i]!.every((v) => v >= 1e6 && v <= 1.1e6), () => `i = ${i}: ${result[i]}`);
    for (let i = 5; i < testData.length; i++) {
        const t = testData[i]!;
        const threshold = Math.max(...t) * 0.0105;
        const expMin = (v: number) => (v > 1e-5 ? Math.max(0, v - threshold) : 0);
        for (let c = 0; c < 3; c++) e.check(result[i]![c]! >= expMin(t[c]!) && result[i]![c]! <= t[c]! + threshold, () => `i = ${i} c ${c}: ${result[i]![c]} vs ${t[c]}`);
    }
    e.done("LogLuvHDR");
});

// Sampling/LowDiscrepancyTests.cpp

gpuTest("FalcorTest.RadicalInverse", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Sampling/LowDiscrepancyTests.cs.slang", "testRadicalInverse");
    ctx.allocateStructuredBuffer("result", 4);
    ctx.vars()["TestCB"]["resultSize"] = 4;
    ctx.runProgram(1);
    const s = await ctx.readBuffer("result", Float32Array);
    expectEq([...s].join(), "0,0.5,0.25,0.75", "radicalInverse(0..3)");
});

// Sampling/PseudorandomTests.cpp

const kPrngShader = "Tests/Sampling/PseudorandomTests.cs.slang";
const kInstances = 256;
const kDimensions = 64;

function createSeed(device: GPUUnitTestContext["device"], elements: number) {
    const rng = new Mt19937();
    const seed = Uint32Array.from({ length: elements }, () => rng.next());
    return { seed, buffer: device.createBuffer(seed.byteLength, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, seed) };
}

gpuTest("FalcorTest.XoshiroPRNG", async ({ device }) => {
    const { seed, buffer } = createSeed(device, kInstances * 4);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kPrngShader, "testXoshiro");
    ctx.allocateStructuredBuffer("result", kInstances * kDimensions);
    ctx.vars()["seed"] = buffer;
    ctx.runProgram(kInstances);
    const result = await ctx.readBuffer("result", Uint32Array);
    const rotl = (x: number, k: number) => ((x << k) | (x >>> (32 - k))) >>> 0;
    const e = new Expect();
    for (let i = 0; i < kInstances; i++) {
        const s = Array.from(seed.subarray(i * 4, i * 4 + 4));
        for (let j = 0; j < kDimensions; j++) {
            const ref = Math.imul(rotl(Math.imul(s[0]!, 5) >>> 0, 7), 9) >>> 0;
            const t = (s[1]! << 9) >>> 0;
            s[2] = (s[2]! ^ s[0]!) >>> 0;
            s[3] = (s[3]! ^ s[1]!) >>> 0;
            s[1] = (s[1]! ^ s[2]!) >>> 0;
            s[0] = (s[0]! ^ s[3]!) >>> 0;
            s[2] = (s[2]! ^ t) >>> 0;
            s[3] = rotl(s[3]!, 11);
            const res = result[j * kInstances + i];
            e.check(ref === res, () => `instance = ${i} dimension = ${j}: ${res} != ${ref}`);
        }
    }
    e.done("Xoshiro");
});

gpuTest("FalcorTest.SplitMixPRNG", async ({ device }) => {
    const { seed, buffer } = createSeed(device, kInstances * 2);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kPrngShader, "testSplitMix");
    ctx.allocateStructuredBuffer("result64", kInstances * kDimensions);
    ctx.vars()["seed"] = buffer;
    ctx.runProgram(kInstances);
    const result = await ctx.readBuffer("result64", Uint32Array);
    const M = (1n << 64n) - 1n;
    const e = new Expect();
    for (let i = 0; i < kInstances; i++) {
        let x = (BigInt(seed[i * 2 + 1]!) << 32n) | BigInt(seed[i * 2]!);
        for (let j = 0; j < kDimensions; j++) {
            x = (x + 0x9e3779b97f4a7c15n) & M;
            let z = x;
            z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M;
            z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M;
            const ref = z ^ (z >> 31n);
            const k = (j * kInstances + i) * 2;
            const res = (BigInt(result[k + 1]!) << 32n) | BigInt(result[k]!);
            e.check(ref === res, () => `instance = ${i} dimension = ${j}: ${res} != ${ref}`);
        }
    }
    e.done("SplitMix64");
});

gpuTest("FalcorTest.LCGPRNG", async ({ device }) => {
    const { seed, buffer } = createSeed(device, kInstances);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kPrngShader, "testLCG");
    ctx.allocateStructuredBuffer("result", kInstances * kDimensions);
    ctx.vars()["seed"] = buffer;
    ctx.runProgram(kInstances);
    const result = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    for (let i = 0; i < kInstances; i++) {
        let state = seed[i]!;
        for (let j = 0; j < kDimensions; j++) {
            state = (Math.imul(1664525, state) + 1013904223) >>> 0;
            const res = result[j * kInstances + i];
            e.check(state === res, () => `instance = ${i} dimension = ${j}: ${res} != ${state}`);
        }
    }
    e.done("LCG");
});
