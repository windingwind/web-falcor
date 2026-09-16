/**
 * PixelDebug wired into the passes that own one natively (SceneDebugger here;
 * PathTracer/BSDFViewer/RTXDI share the same PixelDebug host): enabling it
 * recompiles with _PIXEL_DEBUG_ENABLED, leaves the image unchanged, exposes the
 * (empty) print/assert log, and a left click selects the pixel like native.
 */

import { RenderGraph, Scene, createPass, float2, float3, float4, type PixelDebug } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectArrayEq } from "../harness/registry.js";

gpuTest("PixelDebug.sceneDebuggerToggleAndSelect", async ({ device }) => {
    const vertices = [
        { position: new float3(-1, -1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(3, -1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(-1, 3, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    const scene = new Scene(device, [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }], [{ basic: { baseColor: new float4(0.8, 0.4, 0.2, 1.0) } }]);
    scene.camera.setPosition(new float3(0, 0, 2));
    scene.camera.setTarget(new float3(0, 0, 0));
    scene.camera.setAspectRatio(1);

    const size = 64;
    const graph = new RenderGraph(device, "Debug");
    const pass = createPass(device, "SceneDebugger", { mode: "FaceNormal" });
    graph.addPass(pass, "SceneDebugger");
    graph.markOutput("SceneDebugger.output");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    const read = async () => new Float32Array((await ctx.readTextureSubresource(graph.getOutput("SceneDebugger.output")!)).buffer);

    graph.execute(ctx);
    const plain = await read();

    const debuggable = pass as unknown as { pixelDebug: PixelDebug; onMouseEvent: (ev: { type: "buttonDown"; button: "left"; pos: [number, number] }) => boolean };
    debuggable.pixelDebug.enabled = true;
    debuggable.pixelDebug.selectedPixel = [10, 20];
    graph.execute(ctx); // recompiles with _PIXEL_DEBUG_ENABLED
    graph.execute(ctx);
    const debugged = await read();
    expectArrayEq(debugged, plain, "pixel debug leaves the image unchanged");
    ctx.submit();
    await device.gpuDevice.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 100));
    // Upstream shaders only call printSetPixel(): the log is empty and no assert fires.
    expectEq(debuggable.pixelDebug.getPrintRecords().length, 0, "no print records");
    expectEq(debuggable.pixelDebug.getAssertRecords().length, 0, "no assert records");

    // Mirrors PixelDebug::onMouseEvent: left click selects pos * frameDim.
    expectEq(debuggable.onMouseEvent({ type: "buttonDown", button: "left", pos: [0.25, 0.75] }), true, "click handled");
    expectArrayEq(debuggable.pixelDebug.selectedPixel, [16, 48], "selected pixel");
});
