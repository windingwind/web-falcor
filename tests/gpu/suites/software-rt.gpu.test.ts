/**
 * M6 SoftwareRT test: primary rays traced through SceneRayQuery (software BVH)
 * must reproduce the NATIVE hardware-DXR GBufferRT posW oracle per-pixel.
 */

import { ComputePass, GltfImporter, ResourceBindFlags, float3 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("SoftwareRT.primaryRaysMatchNativeDXR", async ({ device }) => {
    const size = 256;
    const scene = await GltfImporter.importFromUrl(device, "/tests/oracle/assets/quad.gltf");
    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setAspectRatio(1.0);

    const pass = ComputePass.create(device, { path: "RayQueryTest.cs.slang", defines: scene.getSceneDefines() });
    const posW = device.createStructuredBuffer(16, size * size, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    const vis = device.createStructuredBuffer(16, size * size, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);

    const root = pass.getRootVar();
    scene.bindShaderData(root);
    root["CB"]["gFrameDim"] = [size, size];
    root["gPosW"] = posW;
    root["gVisibility"] = vis;

    const ctx = device.renderContext;
    pass.execute(ctx, size, size);

    const web = new Float32Array((await posW.getBlob()).buffer);
    const visData = new Float32Array((await vis.getBlob()).buffer);

    // Native hardware-DXR oracle (same scene/camera; EXR rows bottom-up).
    const res = await fetch("/tests/oracle/out-native/oracle.GBufferRT.posW.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let badPixels = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const webIdx = (y * size + x) * 4;
            const natIdx = ((height - 1 - y) * width + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(web[webIdx + c]! - data[natIdx + c]!);
                sum += d;
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 0.02) badPixels++;
        }
    }
    const mean = sum / (size * size * 3);
    const cc = ((size / 2) * size + size / 2) * 4;
    console.error(`# softwareRT posW: mean=${mean.toExponential(2)} bad=${badPixels}`);
    expectEq(mean < 2e-3, true, `posW mean diff ${mean}`);
    expectEq(badPixels < size * 4, true, `bad pixels ${badPixels} (edge band)`);

    // All hit points see the camera.
    let visErrors = 0;
    for (let i = 0; i < size * size; i++) {
        if (web[i * 4 + 3] === 1 && visData[i * 4] !== 1) visErrors++;
    }
    expectClose(visErrors, 0, 0.5, "visibility rays from hits to camera");

    posW.destroy();
    vis.destroy();
});
