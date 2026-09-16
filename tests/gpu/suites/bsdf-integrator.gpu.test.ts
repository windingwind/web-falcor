/**
 * Port of FalcorTest's BSDFIntegrator GPU test: integrate a rough dielectric
 * StandardMaterial (baseColor 0.3/0.8/0.9, roughness 1, metallic 0) over the
 * hemisphere for four incident angles and compare against the native
 * reference values (BSDFIntegratorTests.cpp kExpectedResults).
 */

import { BSDFIntegrator, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const kExpected: [number, number, number][] = [
    [0.271488, 0.666471, 0.745583],
    [0.230911, 0.580707, 0.650769],
    [0.220602, 0.562734, 0.63126],
    [0.21811, 0.560894, 0.629551],
];
const kCosThetas = [0.25, 0.5, 0.75, 1.0];
const kMaxL2 = 1e-6;

const kScene = `
mat = StandardMaterial('testMaterial')
mat.baseColor = float4(0.3, 0.8, 0.9, 1.0)
mat.metallic = 0.0
mat.roughness = 1.0
quad = TriangleMesh.createQuad(float2(1.0, 1.0))
meshID = sceneBuilder.addTriangleMesh(quad, mat)
nodeID = sceneBuilder.addNode('quad', Transform())
sceneBuilder.addMeshInstance(nodeID, meshID)
`;

gpuTest("BSDFIntegrator.matchesNativeReference", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(device, kScene, "");
    const integrator = new BSDFIntegrator(device, scene);
    expectEq(integrator.resultCount, 256, "256 thread groups per 512x512 grid");

    const results = await integrator.integrateIsotropic(device.renderContext, 0, kCosThetas);
    expectEq(results.length, kCosThetas.length, "one result per incident direction");
    let worst = 0;
    for (let i = 0; i < kCosThetas.length; i++) {
        const r = results[i]!;
        const e = kExpected[i]!;
        const l2 = Math.hypot(r.x - e[0], r.y - e[1], r.z - e[2]);
        worst = Math.max(worst, l2);
        console.log(`# bsdf-integrator cosTheta=${kCosThetas[i]} web=(${r.x.toFixed(6)}, ${r.y.toFixed(6)}, ${r.z.toFixed(6)}) native=(${e.join(", ")}) l2=${l2.toExponential(2)}`);
    }
    expectEq(worst <= kMaxL2, true, `all four integrals within ${kMaxL2} of native (worst l2 ${worst.toExponential(2)})`);

    // Single-direction convenience entry point agrees with the batch.
    const single = await integrator.integrateIsotropicSingle(device.renderContext, 0, 0.5);
    expectEq(Math.hypot(single.x - results[1]!.x, single.y - results[1]!.y, single.z - results[1]!.z) < 1e-7, true, "single == batch entry");
});
