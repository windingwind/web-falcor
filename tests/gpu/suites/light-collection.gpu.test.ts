/**
 * M7: LightCollection GPU data — host-built emissive triangle list read back
 * through the unmodified upstream Scene.Lights.LightCollection module.
 * Scene: quad-emissive.gltf (mesh 0 diffuse, mesh 1 emissive small quad at
 * z=0.5 spanning [0.3,0.7]^2, emissive (1.0, 0.8, 0.4)).
 */

import { ComputePass, GltfImporter, ResourceBindFlags } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectClose, expectArrayClose } from "../harness/registry.js";

gpuTest("LightCollection.emissiveTriangleData", async ({ device }) => {
    const scene = await GltfImporter.importFromUrl(device, "/tests/oracle/assets/quad-emissive.gltf", []);

    const pass = ComputePass.create(device, { path: "LightCollectionSmoke.cs.slang", defines: scene.getSceneDefines() });
    const out = device.createStructuredBuffer(16, 8, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);

    const root = pass.getRootVar();
    scene.bindShaderData(root);
    root["gOut"] = out;
    pass.execute(device.renderContext, 1);

    const r = new Float32Array((await out.getBlob()).buffer);

    // Triangle 1 of the emissive quad: indices [0,2,3] of
    // (0.3,0.3,0.5) (0.7,0.3,0.5) (0.7,0.7,0.5) (0.3,0.7,0.5).
    expectArrayClose(r.subarray(0, 4), [0.3, 0.3, 0.5, 2], 1e-6, "posW0 + triangleCount");
    expectArrayClose(r.subarray(4, 8), [0.7, 0.7, 0.5, 2], 1e-6, "posW1 + activeCount");
    expectArrayClose(r.subarray(8, 12), [0.3, 0.7, 0.5, 0.08], 1e-5, "posW2 + area");
    expectArrayClose(r.subarray(12, 16), [0, 0, 1, 1], 1e-3, "normal + materialID");

    // averageRadiance = emissive * factor; flux = luminance * area * pi.
    const flux = (0.2126 * 1.0 + 0.7152 * 0.8 + 0.0722 * 0.4) * 0.08 * Math.PI;
    expectArrayClose(r.subarray(16, 20), [1.0, 0.8, 0.4, flux], 1e-4, "radiance + flux");

    // Texcoords of vertices 0 and 3 (f16 quantized).
    expectArrayClose(r.subarray(20, 24), [0, 0, 0, 1], 1e-3, "texcoords");

    // MeshLightData: instanceID=1, triangleOffset=0, triangleCount=2, materialID=1.
    expectArrayClose(r.subarray(24, 28), [1, 0, 2, 1], 1e-6, "meshData");

    // perMeshInstanceOffset[0] = kInvalidIndex (bit pattern), [1] = 0; active/mapping identity.
    const bits = new Uint32Array(r.buffer, 28 * 4, 1);
    expectClose(bits[0]! === 0xffffffff ? 1 : 0, 1, 0, "instance 0 has no emissive offset");
    expectArrayClose(r.subarray(29, 32), [0, 1, 1], 1e-6, "offset[1] + active[1] + mapping[1]");
    out.destroy();
});
