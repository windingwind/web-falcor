/**
 * M5 Scene GPU smoke test: the host Scene class drives the unmodified upstream
 * Scene.slang module (camera, vertices, instances, materials) end-to-end.
 */

import {
    ComputePass,
    ResourceBindFlags,
    Scene,
    float2,
    float3,
    float4,
} from "@web-falcor/falcor";
import { gpuTest, expectEq, expectClose, expectArrayClose } from "../harness/registry.js";

function makeTriangleScene(device: any): Scene {
    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 1, 0), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    return new Scene(
        device,
        [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }],
        [{ basic: { baseColor: new float4(0.25, 0.5, 0.75, 1.0) } }],
    );
}

gpuTest("Scene.gSceneSmoke", async ({ device }) => {
    const scene = makeTriangleScene(device);
    scene.camera.setPosition(new float3(0, 0, 5));
    scene.camera.setTarget(new float3(0, 0, 0));
    scene.camera.setAspectRatio(1);

    const pass = ComputePass.create(device, { path: "SceneSmoke.cs.slang", defines: scene.getSceneDefines() });
    const out = device.createStructuredBuffer(16, 8, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);

    const root = pass.getRootVar();
    scene.bindShaderData(root);
    root["gOut"] = out;
    pass.execute(device.renderContext, 8);

    const r = new Float32Array((await out.getBlob()).buffer);

    // 0: camera viewProj row 0 == CPU matrix row 0.
    const m = scene.camera.getViewProjMatrix();
    expectArrayClose(r.subarray(0, 4), [m.get(0, 0), m.get(0, 1), m.get(0, 2), m.get(0, 3)], 1e-5, "viewProj row0");

    // 1: vertex 0 position.
    expectArrayClose(r.subarray(4, 8), [0, 0, 0, 1], 1e-6, "vertex0 position");

    // 2: vertex 1 normal (f16 packed round-trip).
    expectArrayClose(r.subarray(8, 12), [0, 1, 0, 0], 1e-2, "vertex1 normal");

    // 3: instance data (materialID=0, vbOffset=0, ibOffset=0, instanceCount=1).
    expectArrayClose(r.subarray(12, 16), [0, 0, 0, 1], 1e-6, "instance data");

    // 4: indices of triangle 0.
    expectArrayClose(r.subarray(16, 20), [0, 1, 2, 0], 1e-6, "triangle indices");

    // 5: material base color (f16 quantized).
    expectArrayClose(r.subarray(20, 24), [0.25, 0.5, 0.75, 1.0], 1e-2, "material baseColor");

    // 6: material header: type Standard(1), IoR 1.5, isBasic 1.
    expectClose(r[24]!, 1, 1e-6, "material type");
    expectClose(r[25]!, 1.5, 1e-2, "IoR");
    expectClose(r[26]!, 1, 1e-6, "isBasicMaterial");

    // 7: world transform (identity) of vertex 0.
    expectArrayClose(r.subarray(28, 32), [0, 0, 0, 1], 1e-6, "world-space vertex");
    out.destroy();
});
