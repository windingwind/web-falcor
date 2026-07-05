/**
 * M5 GBufferRaster GPU test: the first rendered image of the port — a triangle
 * scene drawn through the upstream GBufferRaster shaders (full material stack),
 * G-buffer channels verified per-pixel.
 */

import { RenderGraph, Scene, createPass, float2, float3, float4 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("GBufferRaster.firstImage", async ({ device }) => {
    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    const scene = new Scene(
        device,
        [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }],
        [{ basic: { baseColor: new float4(0.8, 0.4, 0.2, 1.0) } }],
    );
    scene.camera.setPosition(new float3(0.3, 0.3, 2));
    scene.camera.setTarget(new float3(0.3, 0.3, 0));
    scene.camera.setAspectRatio(1);

    const size = 64;
    const graph = new RenderGraph(device, "GBufferGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "GBufferRaster"), "GBufferRaster");
    graph.markOutput("GBufferRaster.posW");
    graph.markOutput("GBufferRaster.faceNormalW");
    graph.markOutput("GBufferRaster.texC");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);

    const center = (size / 2) * size + size / 2;

    // posW: the triangle lies in the z=0 plane; the camera center ray hits (0.3, 0.3, 0).
    const posW = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRaster.posW")!)).buffer);
    expectClose(posW[center * 4 + 0]!, 0.3, 0.02, "posW.x");
    expectClose(posW[center * 4 + 1]!, 0.3, 0.02, "posW.y");
    expectClose(posW[center * 4 + 2]!, 0.0, 1e-4, "posW.z");

    // faceNormalW: |n| == (0,0,±1) for the z=0 plane (derivative-based, sign may vary).
    const fn = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRaster.faceNormalW")!)).buffer);
    expectClose(Math.abs(fn[center * 4 + 2]!), 1.0, 1e-3, "|faceNormalW.z|");

    // texC: barycentric-interpolated uv == (x, y) for this parameterization.
    const texC = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRaster.texC")!)).buffer);
    expectClose(texC[center * 2 + 0]!, 0.3, 0.02, "texC.u");
    expectClose(texC[center * 2 + 1]!, 0.3, 0.02, "texC.v");

    // A pixel outside the triangle (top-right corner) stays cleared.
    const corner = 4 * size + (size - 4);
    expectEq(posW[corner * 4 + 3]!, 0, "background posW.w cleared");
});
