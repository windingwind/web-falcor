/**
 * Transplanted FalcorTest GPU tests: Rendering/Materials/MicrofacetTests.
 * Sampling histograms are chi^2-tested against the tabulated pdf (native's
 * hypothesis::chi2_test settings); the kernel override replaces double with float.
 */

import { type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext, getElements } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";
import { chi2Test } from "../../harness/hypothesis.js";

const kShaderFile = "Tests/Rendering/Materials/MicrofacetTests.cs.slang";
const kNdfs = ["TrowbridgeReitzNDF", "BeckmannSpizzichinoNDF"];

interface NDFConfig {
    name: string;
    alpha: [number, number];
    rotation: number;
}

interface SamplingTestSpec {
    visibleNormals: boolean;
    ndf: string;
    ndfConfig: NDFConfig;
    incidentAngles: number[];
}

const kPhiBins = 128 + 2;
const kCosThetaBins = 64 + 1;
const kSampleCount = 64 * 1024 * 1024;
const kThreadSampleCount = 128 * 1024;
const kBinSampleCount = 128;

function setup(ctx: GPUUnitTestContext, spec: SamplingTestSpec, entry: string, testCount: number) {
    ctx.createProgram(kShaderFile, entry, { TEST_NDF_TYPE: spec.ndf });
    const v = ctx.vars()["gMicrofacetSamplingTest"];
    v["testCount"] = testCount;
    v["phiBinCount"] = kPhiBins;
    v["cosThetaBinCount"] = kCosThetaBins;
    v["sampleCount"] = kSampleCount;
    v["threadSampleCount"] = kThreadSampleCount;
    v["binSampleCount"] = kBinSampleCount;
    v["visibleNormals"] = spec.visibleNormals;
    v["alpha"] = spec.ndfConfig.alpha;
    v["rotation"] = spec.ndfConfig.rotation;
    const wi = new Float32Array(testCount * 4); // float3 at WGSL's 16-byte stride
    for (let i = 0; i < testCount; i++) {
        const theta = Math.fround((Math.PI * spec.incidentAngles[i]!) / 180);
        wi.set([Math.fround(Math.sin(theta)), 0, Math.fround(Math.cos(theta))], i * 4);
    }
    v["testWi"] = ctx.device.createStructuredBuffer(16, testCount, undefined, wi);
    return v;
}

async function testSampling(device: Device, spec: SamplingTestSpec, e: Expect): Promise<void> {
    const testCount = spec.visibleNormals ? spec.incidentAngles.length : 1;
    const binCount = kPhiBins * kCosThetaBins;
    const ctx = new GPUUnitTestContext(device);

    let v = setup(ctx, spec, "tabulateHistogram", testCount);
    const hist = device.createStructuredBuffer(4, testCount * binCount);
    ctx.getRenderContext().clearBuffer(hist);
    v["histogramSampling"] = hist;
    ctx.runProgram(kSampleCount / kThreadSampleCount, 1, testCount);
    const obs = await getElements(hist, Uint32Array);

    v = setup(ctx, spec, "tabulatePdf", testCount);
    const pdf = device.createStructuredBuffer(4, testCount * binCount);
    v["histogramPdf"] = pdf;
    ctx.runProgram(kPhiBins, kCosThetaBins, testCount);
    const exp = await getElements(pdf, Float32Array);

    for (let t = 0; t < testCount; t++) {
        const name = `${spec.ndf}__${spec.visibleNormals ? "visible" : "all"}__${spec.ndfConfig.name}__${spec.incidentAngles[t]}`;
        const { success, report } = chi2Test(binCount, obs.subarray(t * binCount, (t + 1) * binCount), exp.subarray(t * binCount, (t + 1) * binCount), kSampleCount, 5, 0.01, testCount);
        e.check(success, () => `${name}: ${report}`);
    }
}

gpuTest("FalcorTest.MicrofacetSampling", async ({ device }) => {
    const configs: NDFConfig[] = [
        { name: "isotropic_high_roughness", alpha: [1, 1], rotation: 0 },
        { name: "isotropic_medium_roughness", alpha: [0.6, 0.6], rotation: 0 },
        { name: "anisotropic_axisaligned", alpha: [0.6, 1], rotation: 0 },
        { name: "anisotropic_rotated", alpha: [0.6, 1], rotation: 0.6 },
    ];
    const e = new Expect();
    for (const visibleNormals of [false, true])
        for (const ndf of kNdfs)
            for (const ndfConfig of configs) await testSampling(device, { visibleNormals, ndf, ndfConfig, incidentAngles: [0, 30, 80, 130] }, e);
    e.done("MicrofacetSampling");
});

/** Runs a two-result kernel over N directions wi = (sqrt(1 - mu^2), 0, mu). */
async function runPair(device: Device, entry: string, mus: number[]): Promise<{ r1: Float32Array; r2: Float32Array }[]> {
    const out: { r1: Float32Array; r2: Float32Array }[] = [];
    for (const ndf of kNdfs) {
        const ctx = new GPUUnitTestContext(device);
        ctx.createProgram(kShaderFile, entry, { TEST_NDF_TYPE: ndf });
        const wi = new Float32Array(mus.length * 4);
        mus.forEach((mu, i) => wi.set([Math.fround(Math.sqrt(Math.fround(1 - Math.fround(mu * mu)))), 0, mu], i * 4));
        ctx.allocateStructuredBuffer("testWi", mus.length, wi);
        ctx.allocateStructuredBuffer("result1", mus.length);
        ctx.allocateStructuredBuffer("result2", mus.length);
        ctx.runProgram(mus.length, 1, 1);
        out.push({ r1: await ctx.readBuffer("result1", Float32Array), r2: await ctx.readBuffer("result2", Float32Array) });
    }
    return out;
}

const N = 32;
const kSignedMus = Array.from({ length: N }, (_, i) => Math.fround(-1 + Math.fround((2 * i) / (N - 1))));

gpuTest("FalcorTest.MicrofacetSigmaIntegration", async ({ device }) => {
    const e = new Expect();
    for (const [t, { r1, r2 }] of (await runPair(device, "sigmaIntegration", kSignedMus)).entries())
        for (let i = 0; i < N; i++) e.check(Math.abs(r1[i]! - r2[i]!) < 5e-3, () => `${kNdfs[t]} i=${i}: integrated ${r1[i]} vs analytic ${r2[i]}`);
    e.done("MicrofacetSigmaIntegration");
});

gpuTest("FalcorTest.MicrofacetSigmaLambdaConsistency", async ({ device }) => {
    const e = new Expect();
    for (const [t, { r1, r2 }] of (await runPair(device, "sigmaLambdaConsistency", kSignedMus)).entries())
        for (let i = 0; i < N; i++) {
            const mu = kSignedMus[i]!;
            const rhs = mu > 0 ? r2[i]! * mu : (1 + r2[i]!) * -mu;
            e.check(Math.abs(r1[i]! - rhs) < 1e-3, () => `${kNdfs[t]} i=${i}: sigma ${r1[i]} vs ${rhs}`);
        }
    e.done("MicrofacetSigmaLambdaConsistency");
});

gpuTest("FalcorTest.MicrofacetLambdaNonsymmetry", async ({ device }) => {
    const mus = Array.from({ length: N }, (_, i) => Math.max(Math.fround(1e-4), Math.fround(i / (N - 1))));
    const e = new Expect();
    for (const [t, { r1, r2 }] of (await runPair(device, "lambdaNonsymmetry", mus)).entries())
        for (let i = 0; i < N; i++) e.check(Math.abs(r2[i]! - (1 + r1[i]!)) < 1e-3, () => `${kNdfs[t]} i=${i}: Lambda(-w) ${r2[i]} vs 1 + Lambda(w) ${1 + r1[i]!}`);
    e.done("MicrofacetLambdaNonsymmetry");
});

gpuTest("FalcorTest.MicrofacetG1Symmetry", async ({ device }) => {
    const e = new Expect();
    for (const [t, { r1, r2 }] of (await runPair(device, "g1Symmetry", kSignedMus)).entries())
        for (let i = 0; i < N; i++) e.check(Math.abs(r1[i]! - r2[i]!) < 1e-3, () => `${kNdfs[t]} i=${i}: G1(w) ${r1[i]} vs G1(-w) ${r2[i]}`);
    e.done("MicrofacetG1Symmetry");
});
