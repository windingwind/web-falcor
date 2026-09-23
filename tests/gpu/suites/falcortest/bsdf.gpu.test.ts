/**
 * Transplanted FalcorTest GPU tests: Scene/Material/BSDFTests. Each BSDF's
 * sampling is chi^2-tested against its pdf, and sample() weights/pdfs are
 * checked against eval()/evalPdf(). Native-disabled cases (SimpleBTDF: delta
 * lobe; SpecularMicrofacetBSDF: "not passing") stay disabled.
 */

import { type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext, getElements } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";
import { chi2Test } from "../../harness/hypothesis.js";

const kShaderFile = "Tests/Scene/Material/BSDFTests.cs.slang";
const kPhiBins = 128 + 2;
const kCosThetaBins = 64 + 1;
const kSampleCount = 64 * 1024 * 1024;
const kThreadSampleCount = 128 * 1024;
const kBinSampleCount = 128;
const kMaxWeightError = 1e-3;
const kMaxPdfError = 1e-3;

type V3 = [number, number, number];
const normalize = (v: V3): V3 => {
    const l = Math.fround(Math.hypot(...v));
    return v.map((x) => Math.fround(x / l)) as V3;
};
const perp = normalize([0, 0, 1]);
const oblique = normalize([0.5, 0, 0.5]);
const grazing = normalize([0, 1, 0.01]);

interface Config {
    name: string;
    wi: V3;
    params: [number, number, number, number];
}

const standardConfigs: Config[] = [
    { name: "perp", wi: perp, params: [0, 0, 0, 0] },
    { name: "oblique", wi: oblique, params: [0, 0, 0, 0] },
    { name: "grazing", wi: grazing, params: [0, 0, 0, 0] },
];

async function testSampling(device: Device, bsdfImport: string, bsdf: string, bsdfInit: string, configs: Config[]): Promise<void> {
    const testCount = configs.length;
    const binCount = kPhiBins * kCosThetaBins;
    const ctx = new GPUUnitTestContext(device);
    const setup = (entry: string) => {
        ctx.createProgram(kShaderFile, entry, { TEST_BSDF_IMPORT: bsdfImport, TEST_BSDF: bsdf, TEST_BSDF_INIT: bsdfInit });
        const v = ctx.vars()["SamplingTestCB"]["gSamplingTest"];
        v["testCount"] = testCount;
        v["phiBinCount"] = kPhiBins;
        v["cosThetaBinCount"] = kCosThetaBins;
        v["sampleCount"] = kSampleCount;
        v["threadSampleCount"] = kThreadSampleCount;
        v["binSampleCount"] = kBinSampleCount;
        const wi = new Float32Array(testCount * 4); // float3 at WGSL's 16-byte stride
        configs.forEach((c, i) => wi.set(c.wi, i * 4));
        v["testWi"] = device.createStructuredBuffer(16, testCount, undefined, wi);
        v["testParams"] = device.createStructuredBuffer(16, testCount, undefined, Float32Array.from(configs.flatMap((c) => c.params)));
        return v;
    };
    const zeroed = () => {
        const b = device.createStructuredBuffer(4, testCount * binCount);
        ctx.getRenderContext().clearBuffer(b);
        return b;
    };

    let v = setup("tabulateHistogram");
    const hist = zeroed();
    v["histogramSampling"] = hist;
    ctx.runProgram(kSampleCount / kThreadSampleCount, 1, testCount);
    const obs = await getElements(hist, Uint32Array);

    v = setup("tabulatePdf");
    const pdf = zeroed();
    v["histogramPdf"] = pdf;
    ctx.runProgram(kPhiBins, kCosThetaBins, testCount);
    const exp = await getElements(pdf, Float32Array);

    v = setup("tabulateWeightAndPdfError");
    const weightErr = zeroed();
    const pdfErr = zeroed();
    v["histogramWeightError"] = weightErr;
    v["histogramPdfError"] = pdfErr;
    ctx.runProgram(kSampleCount / kThreadSampleCount, 1, testCount);
    const we = await getElements(weightErr, Float32Array);
    const pe = await getElements(pdfErr, Float32Array);

    const e = new Expect();
    configs.forEach((c, t) => {
        const name = `${bsdf}_${c.name}`;
        const range = [t * binCount, (t + 1) * binCount] as const;
        const { success, report } = chi2Test(binCount, obs.subarray(...range), exp.subarray(...range), kSampleCount, 5, 0.01, testCount);
        e.check(success, () => `${name}: ${report}`);
        const maxWeightError = Math.max(...we.subarray(...range));
        const maxPdfError = Math.max(...pe.subarray(...range));
        e.check(maxWeightError <= kMaxWeightError, () => `${name}: max weight error ${maxWeightError}`);
        e.check(maxPdfError <= kMaxPdfError, () => `${name}: max pdf error ${maxPdfError}`);
    });
    e.done(bsdf);
}

const cases: [string, string, string, string, Config[]][] = [
    ["DisneyDiffuseBRDF", "Rendering.Materials.BSDFs.DisneyDiffuseBRDF", "DisneyDiffuseBRDF", "bsdf.albedo = float3(1.f); bsdf.roughness = 0.5f;", standardConfigs],
    ["FrostbiteDiffuseBRDF", "Rendering.Materials.BSDFs.FrostbiteDiffuseBRDF", "FrostbiteDiffuseBRDF", "bsdf.albedo = float3(1.f); bsdf.roughness = 0.5f;", standardConfigs],
    ["LambertDiffuseBRDF", "Rendering.Materials.BSDFs.LambertDiffuseBRDF", "LambertDiffuseBRDF", "bsdf.albedo = float3(1.f);", standardConfigs],
    ["LambertDiffuseBTDF", "Rendering.Materials.BSDFs.LambertDiffuseBTDF", "LambertDiffuseBTDF", "bsdf.albedo = float3(1.f);", standardConfigs],
    ["OrenNayarBRDF", "Rendering.Materials.BSDFs.OrenNayarBRDF", "OrenNayarBRDF", "bsdf.albedo = float3(1.f); bsdf.roughness = 0.5f", standardConfigs],
    [
        "SpecularMicrofacetBRDF",
        "Rendering.Materials.BSDFs.SpecularMicrofacet",
        "SpecularMicrofacetBRDF",
        "bsdf.albedo = float3(1.f); bsdf.activeLobes = 0xff; bsdf.alpha = params.x;",
        [0.05, 0.5].flatMap((alpha) =>
            standardConfigs.map((c) => ({ ...c, name: `${alpha === 0.05 ? "smooth" : "rough"}_${c.name}`, params: [alpha, 0, 0, 0] as Config["params"] })),
        ),
    ],
    ["SheenBSDF", "Rendering.Materials.BSDFs.SheenBSDF", "SheenBSDF", "bsdf.color = float3(1.f); bsdf.roughness = 0.5f", standardConfigs],
    ["DiffuseSpecularBRDF", "Rendering.Materials.BSDFs.DiffuseSpecularBRDF", "DiffuseSpecularBRDF", "bsdf.diffuse = float3(0.5f); bsdf.specular = float3(0.04f); bsdf.roughness = 0.5f", standardConfigs],
];

for (const [name, imp, bsdf, init, configs] of cases) gpuTest(`FalcorTest.TestBsdf_${name}`, async ({ device }) => testSampling(device, imp, bsdf, init, configs));
