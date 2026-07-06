/**
 * M7: complex pyscene geometry check — 25 instances with compound euler
 * rotations (ZYX, verified against math::quatFromEulerAngles) traced via
 * SceneRayQuery, compared against native GBufferRT posW per-pixel.
 */
import { ComputePass, ResourceBindFlags, float3, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("ClusterPosW.matchesNativeGBuffer", async ({ device }) => {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/oracle-pt-bvhcluster.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setPosition(new float3(3.0, 3.5, 7.0));
    scene.camera.setTarget(new float3(3.0, 0.5, 0.0));
    scene.camera.setAspectRatio(1.0);

    const pass = ComputePass.create(device, { path: "RayQueryTest.cs.slang", defines: scene.getSceneDefines() });
    const posW = device.createStructuredBuffer(16, size * size, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    const vis = device.createStructuredBuffer(16, size * size, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    root["CB"]["gFrameDim"] = [size, size];
    root["gPosW"] = posW;
    root["gVisibility"] = vis;
    pass.execute(device.renderContext, size, size);
    const web = new Float32Array((await posW.getBlob()).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-bvhcluster-gbuf.GBufferRT.posW.0.exr");
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
            if (pixelMax > 0.05) badPixels++;
        }
    }
    console.error(`# clusterPosW: mean=${(sum / (size * size * 3)).toExponential(2)} bad=${badPixels}`);
    expectEq(badPixels < 200, true, `bad pixels ${badPixels}`);
});
