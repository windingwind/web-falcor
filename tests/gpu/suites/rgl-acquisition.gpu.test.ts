/**
 * Port of Rendering/Materials/RGLAcquisition: virtually measure a StandardMaterial, write the
 * fields as an RGL `.bsdf` tensor file, parse it back and render it as an RGLMaterial. Native's
 * test only checks that acquisition runs; here the measured material's albedo must match the source.
 */

import { BSDFIntegrator, MaterialType, RGLAcquisition, Scene, float2, float3, float4, initScripting, parseRGLFile, parseRGLTensorFile, runSceneScript, writeRGLTensorFile } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

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
const kCosThetas = [0.3, 0.6, 0.9];

gpuTest("RGLAcquisition.measuredMaterialMatchesSource", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await runSceneScript(device, kScene, "");
    const ctx = device.renderContext;
    const acquisition = new RGLAcquisition(device, source);
    await acquisition.acquireIsotropic(ctx, 0);
    const fields = await acquisition.toRGLFile();

    // The written tensor file parses back to identical fields.
    const bytes = writeRGLTensorFile(fields);
    const parsed = parseRGLTensorFile(bytes.buffer as ArrayBuffer);
    for (const f of fields) {
        const p = parsed.get(f.name);
        expectEq(p !== undefined && p.type === f.type && p.shape.join() === f.shape.join(), true, `field '${f.name}' round-trips`);
        expectEq(p!.data.every((v, i) => v === f.data[i]), true, `field '${f.name}' data round-trips`);
    }
    for (const name of ["ndf", "sigma", "vndf", "luminance", "rgb"]) {
        const data = parsed.get(name)!.data;
        expectEq(data.every((v) => Number.isFinite(v)) && data.some((v) => v > 0), true, `'${name}' is finite and non-zero`);
    }

    const rgl = parseRGLFile(bytes.buffer as ArrayBuffer, "acquired");
    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    const measured = new Scene(device, [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }], [{ name: rgl.name, basic: {}, rgl, header: { materialType: MaterialType.RGL } }]);

    const reference = await new BSDFIntegrator(device, source).integrateIsotropic(ctx, 0, kCosThetas);
    const albedo = await new BSDFIntegrator(device, measured).integrateIsotropic(ctx, 0, kCosThetas);
    for (let i = 0; i < kCosThetas.length; i++) {
        const [r, a] = [reference[i]!, albedo[i]!];
        console.error(`# RGL acquisition cos=${kCosThetas[i]}: measured (${a.x.toFixed(4)}, ${a.y.toFixed(4)}, ${a.z.toFixed(4)}) vs source (${r.x.toFixed(4)}, ${r.y.toFixed(4)}, ${r.z.toFixed(4)})`);
        for (const [m, s] of [[a.x, r.x], [a.y, r.y], [a.z, r.z]] as const) {
            expectEq(Math.abs(m - s) / s < 0.02, true, `measured albedo within 2% of the source (${m} vs ${s})`);
        }
    }
});
