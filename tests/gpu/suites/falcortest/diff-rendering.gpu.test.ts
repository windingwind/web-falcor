/**
 * Transplanted FalcorTest GPU tests: DiffRendering/SceneGradientsTest (native runs it on D3D12
 * only; its Vulkan code generation is broken).
 */

import { ComputePass, GradientType, ResourceBindFlags, SceneGradients, initScripting, runSceneScript, type ShaderVar } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

/** The pinned Slang build with working backward autodiff (as BSDFOptimizer and WARDiffPathTracer use). */
const kAutodiffSlang = "/tools/slang-wasm-2026.5.2/slang-wasm.js";

gpuTest("FalcorTest.AggregateGradients", async ({ device }) => {
    // 1024 atomic adds of 10^i into element i of a 3-dim gradient, spread over a 64-entry hash grid.
    const [gradDim, elemCount, hashSize] = [3, 1024, 64];
    const ctx = new GPUUnitTestContext(device);
    const grads = new SceneGradients(device, [{ type: GradientType.Material, dim: gradDim, hashSize }]);
    grads.clearGrads(device.renderContext, GradientType.Material);
    ctx.createProgram("Tests/DiffRendering/SceneGradientsTest.cs.slang", "atomicAdd");
    ctx.vars()["CB"]["sz"] = [gradDim, elemCount];
    ctx.vars()["CB"]["hashSize"] = hashSize;
    grads.bindShaderData(ctx.vars()["gSceneGradients"]);
    ctx.runProgram(gradDim, elemCount, 1);
    grads.aggregateGrads(device.renderContext, GradientType.Material);

    ctx.createProgram("Tests/DiffRendering/SceneGradientsTest.cs.slang", "testAggregateGradients");
    ctx.vars()["CB"]["sz"] = [gradDim, elemCount];
    ctx.vars()["grads"] = grads.getGradsBuffer(GradientType.Material)!;
    ctx.allocateStructuredBuffer("result", gradDim);
    ctx.runProgram(gradDim, 1, 1);
    const result = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    for (let i = 0; i < gradDim; i++) {
        const ref = elemCount * 10 ** i;
        e.check(Math.abs(result[i]! - ref) / ref <= 1e-6, () => `grad ${i}: ${result[i]} vs ${ref}`);
    }
    e.done("AggregateGradients");
});

// DiffRendering/Material/DiffMaterialTests: bwd_diff of a PBRTDiffuse eval w.r.t. albedo and wo
// (native: D3D12 only). The web scene needs geometry, so the material sits on a quad.
gpuTest("FalcorTest.DiffPBRTDiffuse", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(
        device,
        [
            "m = PBRTDiffuseMaterial('PBRTDiffuse')",
            "m.baseColor = float4(0.9, 0.6, 0.2, 1.0)",
            "sceneBuilder.addMeshInstance(sceneBuilder.addNode('Quad', Transform()), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(), m))",
        ].join("\n"),
        "/Falcor/media",
    );
    await device.programManager.loadSlangRuntime(kAutodiffSlang);
    const grads = new SceneGradients(device, [
        { type: GradientType.Material, dim: 3, hashSize: 1 },
        { type: GradientType.MeshNormal, dim: 3, hashSize: 1 },
    ]);
    grads.clearGrads(device.renderContext, GradientType.Material);
    grads.clearGrads(device.renderContext, GradientType.MeshNormal);
    const pass = ComputePass.create(device, { path: "Tests/DiffRendering/Material/DiffMaterialTests.cs.slang", csEntry: "testDiffPBRTDiffuse", defines: scene.getSceneDefines(), slangRuntime: kAutodiffSlang });
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    grads.bindShaderData(root["gSceneGradients"] as ShaderVar);
    const norm = (v: number[]) => v.map((x) => x / Math.hypot(...v));
    (root["CB"] as ShaderVar)["gWi"] = norm([0.3, 0.2, 0.8]);
    (root["CB"] as ShaderVar)["gWo"] = norm([-0.1, -0.3, 0.9]);
    const make = () => device.createStructuredBuffer(4, 3, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
    const [materialGrad, geometryGrad] = [make(), make()];
    root["materialGrad"] = materialGrad;
    root["geometryGrad"] = geometryGrad;
    pass.execute(device.renderContext, 1, 1, 1);
    const read = async (b: typeof materialGrad) => new Float32Array((await b.getBlob()).buffer);
    const [mg, gg] = [await read(materialGrad), await read(geometryGrad)];
    const kExpectedMaterialGrad = [0.3003115, 0.3003115, 0.3003115];
    const kExpectedGeometryGrad = [0, 0, 0.5411268];
    const e = new Expect();
    for (let i = 0; i < 3; i++) {
        e.check(Math.abs(mg[i]! - kExpectedMaterialGrad[i]!) <= 1e-3, () => `material grad ${Array.from(mg)}`);
        e.check(Math.abs(gg[i]! - kExpectedGeometryGrad[i]!) <= 1e-3, () => `geometry grad ${Array.from(gg)}`);
    }
    console.error(`# DiffPBRTDiffuse: material ${Array.from(mg)}, geometry ${Array.from(gg)}`);
    e.done("DiffPBRTDiffuse");
});
