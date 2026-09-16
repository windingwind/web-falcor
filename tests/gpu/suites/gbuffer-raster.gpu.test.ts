/**
 * M5 GBufferRaster GPU test: the first rendered image of the port — a triangle
 * scene drawn through the upstream GBufferRaster shaders (full material stack),
 * G-buffer channels verified per-pixel.
 */

import { RenderGraph, Scene, createPass, float2, float3, float4, initScripting, runSceneScript } from "@web-falcor/falcor";
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

/**
 * Every native GBufferRaster channel (kGBufferChannels + kGBufferExtraChannels, minus the
 * primitive-id-bound vbuffer) is produced by the web raster path via batched render targets;
 * cross-check against GBufferRT on the Cornell box (same material stack, same shading code).
 */
gpuTest("GBufferRaster.allChannelsMatchGBufferRT", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    const ctx = device.renderContext;
    const size = 128;

    // 16 raster channels -> 2 batches (8 + 8); pnFwidth is raster-only (native GBufferRT lacks it).
    const shared = ["posW", "normW", "tangentW", "faceNormalW", "texC", "texGrads", "mvec", "mtlData", "guideNormalW", "diffuseOpacity", "specRough", "emissive", "viewW", "linearZ", "mask"];
    const render = async (passType: string) => {
        const channels = passType === "GBufferRaster" ? [...shared, "pnFwidth"] : shared;
        const g = new RenderGraph(device, passType);
        g.addPass(createPass(device, passType, { samplePattern: "Center" }), "Pass");
        for (const ch of channels) g.markOutput(`Pass.${ch}`);
        g.onResize(size, size);
        g.setScene(scene);
        await g.init();
        g.execute(ctx);
        const out: Record<string, { f32: Float32Array; u32: Uint32Array; comps: number }> = {};
        for (const ch of channels) {
            const tex = g.getOutput(`Pass.${ch}`)!;
            const bytes = (await ctx.readTextureSubresource(tex)).buffer;
            const comps = bytes.byteLength / (size * size * 4);
            out[ch] = { f32: new Float32Array(bytes), u32: new Uint32Array(bytes), comps };
        }
        return out;
    };
    const rt = await render("GBufferRT");
    const raster = await render("GBufferRaster");

    // Hit mask agreement (raster coverage vs. ray hits): both write mask = 1 on surfaces.
    let hits = 0;
    let maskMismatch = 0;
    for (let i = 0; i < size * size; i++) {
        const a = rt["mask"]!.f32[i]!;
        const b = raster["mask"]!.f32[i]!;
        if (a !== b) maskMismatch++;
        if (a === 1 && b === 1) hits++;
    }
    expectEq(hits > size * size * 0.5, true, `enough hit pixels (${hits})`);
    expectEq(maskMismatch <= size * size * 0.01, true, `mask mismatches ${maskMismatch}`);

    // Per-channel comparison over agreeing hit pixels. mtlData is integer-exact;
    // float channels get a small tolerance (raster interpolation vs. ray hit).
    const compare = (ch: string, eps: number, comps = raster[ch]!.comps) => {
        let bad = 0;
        let maxDiff = 0;
        for (let i = 0; i < size * size; i++) {
            if (rt["mask"]!.f32[i] !== 1 || raster["mask"]!.f32[i] !== 1) continue;
            for (let c = 0; c < comps; c++) {
                const d = Math.abs(rt[ch]!.f32[i * raster[ch]!.comps + c]! - raster[ch]!.f32[i * raster[ch]!.comps + c]!);
                maxDiff = Math.max(maxDiff, d);
                if (d > eps) {
                    if (bad < 6) console.error(`#   ${ch} px ${i % size},${Math.floor(i / size)}: rt=${[0, 1, 2].map((k) => rt[ch]!.f32[i * raster[ch]!.comps + k]!.toFixed(4))} raster=${[0, 1, 2].map((k) => raster[ch]!.f32[i * raster[ch]!.comps + k]!.toFixed(4))}`);
                    bad++;
                    break;
                }
            }
        }
        console.error(`# gbuffer ${ch}: bad=${bad} maxDiff=${maxDiff.toExponential(2)}`);
        expectEq(bad <= hits * 0.01, true, `${ch}: ${bad} pixels differ by more than ${eps}`);
    };
    compare("posW", 5e-3, 3);
    compare("normW", 2e-2, 3);
    compare("tangentW", 2e-2, 4);
    compare("faceNormalW", 2e-2, 3);
    compare("texC", 5e-3, 2);
    compare("guideNormalW", 2e-2, 3);
    compare("diffuseOpacity", 1e-2, 4);
    compare("specRough", 1e-2, 4);
    compare("emissive", 1e-2, 3);
    compare("viewW", 5e-3, 3);
    compare("mvec", 1e-4, 2); // static scene: both zero
    // linearZ: native raster stores clip z*w, native RT stores view depth w — different
    // quantities by design; check the raster channel is populated and finite.
    {
        let ok = 0;
        for (let i = 0; i < size * size; i++) {
            if (raster["mask"]!.f32[i] !== 1) continue;
            const z = raster["linearZ"]!.f32[i * 2]!;
            const dz = raster["linearZ"]!.f32[i * 2 + 1]!;
            if (Number.isFinite(z) && z > 0 && Number.isFinite(dz) && dz >= 0) ok++;
        }
        expectEq(ok, hits, "linearZ populated on all hits");
    }
    let mtlBad = 0;
    for (let i = 0; i < size * size; i++) {
        if (rt["mask"]!.f32[i] !== 1 || raster["mask"]!.f32[i] !== 1) continue;
        for (let c = 0; c < 4; c++) if (rt["mtlData"]!.u32[i * 4 + c] !== raster["mtlData"]!.u32[i * 4 + c]) { mtlBad++; break; }
    }
    expectEq(mtlBad <= hits * 0.01, true, `mtlData mismatches ${mtlBad}`);
    // Derivative-based channels: RT ray differentials vs. raster ddx/ddy — finite, same sign/scale.
    for (const ch of ["texGrads", "pnFwidth"]) {
        let finite = 0;
        let nonZero = 0;
        for (let i = 0; i < size * size; i++) {
            if (raster["mask"]!.f32[i] !== 1) continue;
            const v = raster[ch]!.f32[i * raster[ch]!.comps]!;
            if (Number.isFinite(v)) finite++;
            if (v !== 0) nonZero++;
        }
        expectEq(finite, hits, `${ch} finite on hits`);
        expectEq(nonZero > hits * 0.5, true, `${ch} populated (${nonZero}/${hits})`);
    }
});

/** Depth-only consumers still get a full geometry pass (scratch color target). */
gpuTest("GBufferRaster.depthOnlyOutput", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    const size = 64;
    const g = new RenderGraph(device, "DepthOnly");
    g.addPass(createPass(device, "GBufferRaster"), "Pass");
    g.markOutput("Pass.depth");
    g.onResize(size, size);
    g.setScene(scene);
    await g.init();
    g.execute(device.renderContext);
    const depth = new Float32Array((await device.renderContext.readTextureSubresource(g.getOutput("Pass.depth")!)).buffer);
    let covered = 0;
    for (const d of depth) if (d < 1) covered++;
    expectEq(covered > size * size * 0.5, true, `depth written (${covered} px < 1)`);
});
