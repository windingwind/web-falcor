/**
 * In-session shader reload (ProgramManager::reloadAllPrograms): recompiling
 * every program from the current sources without tearing down the device, the
 * scene or the render graph — Falcor's F5.
 *
 * The test edits a shader's source in memory, reloads, and checks that a pass
 * built before the reload now runs the new code. A live render graph is then
 * reloaded mid-flight to confirm its passes rebuild their pipelines instead of
 * breaking, which is the part that makes the feature usable while rendering.
 */

import { Buffer, ComputePass, MemoryType, RenderGraph, ResourceBindFlags, createPass, runSceneScript, initScripting } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

const kShaderPath = "WebFalcor/ReloadTest.cs.slang";

gpuTest("ShaderReload.recompilesLivePassesFromEditedSources", async ({ device }) => {
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const out = new Buffer(device, { size: 4, structSize: 4, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "reload::out" });
    const pass = ComputePass.create(device, { path: kShaderPath });
    const ctx = device.renderContext;

    const run = async (): Promise<number> => {
        pass.getRootVar()["gOutput"] = out;
        pass.execute(ctx, 1);
        return new Float32Array((await ctx.readBuffer(out)).buffer)[0]!;
    };

    expectClose(await run(), 11, 1e-6, "the shader's original constant");

    // Patch the source the same way an editor save would, then reload.
    const manager = device.programManager;
    const previous = manager.getSourceProvider();
    const original = previous.resolveSource(kShaderPath)!;
    expectEq(original.includes("11.0"), true, "the fixture still carries its constant");
    const patched = original.replace("11.0", "22.0");
    const generation = manager.generation;
    manager.reloadAllPrograms({
        resolveSource: (path) => (path === kShaderPath ? patched : previous.resolveSource(path)),
        filePaths: previous.filePaths,
    });
    expectEq(manager.generation, generation + 1, "the reload advanced the shader generation");

    // The same pass object, built before the reload, must now run the new code.
    expectClose(await run(), 22, 1e-6, "the reloaded shader's constant");

    // And a reload with no edits leaves the result alone.
    manager.reloadAllPrograms({ resolveSource: previous.resolveSource, filePaths: previous.filePaths });
    expectClose(await run(), 11, 1e-6, "reverting the source reverts the result");
});

gpuTest("ShaderReload.keepsALiveRenderGraphRunning", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, source, "/Falcor/media/test_scenes");
    const size = 64;
    scene.camera.setAspectRatio(1);

    // A rasterizing pass and a ray-traced one, so the reload has to rebuild both
    // a graphics pipeline and a compute pipeline that were created before it.
    const graph = new RenderGraph(device, "Reload");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRaster", { samplePattern: "Center" }), "Raster");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBuffer");
    graph.markOutput("Raster.vbuffer");
    graph.markOutput("GBuffer.posW");
    graph.setScene(scene);
    await graph.init();

    const ctx = device.renderContext;
    const render = async (): Promise<{ vbuffer: Uint8Array; posW: Uint8Array }> => {
        graph.execute(ctx);
        return {
            vbuffer: await ctx.readTextureSubresource(graph.getOutput("Raster.vbuffer")!),
            posW: await ctx.readTextureSubresource(graph.getOutput("GBuffer.posW")!),
        };
    };

    const before = await render();
    // Reloading every program while the graph is alive: its passes hold kernels
    // and pipelines compiled before the reload and have to rebuild them.
    const previous = device.programManager.getSourceProvider();
    device.programManager.reloadAllPrograms({ resolveSource: previous.resolveSource, filePaths: previous.filePaths });
    const after = await render();

    // Both outputs are deterministic (centre samples, no accumulation), so the
    // same sources must give back the same bytes.
    let hits = 0;
    const posF32 = new Float32Array(before.posW.buffer);
    for (let i = 0; i < size * size; i++) if (posF32[i * 4 + 3] !== 0) hits++;
    let diffs = 0;
    for (let i = 0; i < before.vbuffer.length; i++) if (before.vbuffer[i] !== after.vbuffer[i]) diffs++;
    for (let i = 0; i < before.posW.length; i++) if (before.posW[i] !== after.posW[i]) diffs++;
    console.error(`# shader reload: ${hits}/${size * size} hits, ${diffs} differing bytes after reloading every program`);
    expectEq(hits > (size * size) / 2, true, `the graph renders the scene (${hits} hits)`);
    expectEq(diffs, 0, "the same sources give the same image after a reload");
});
