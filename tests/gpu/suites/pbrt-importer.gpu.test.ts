/**
 * PBRTImporter end-to-end: loads Benedikt Bitterli's pbrt-v4 scenes (CC0)
 * through runPbrtScene and renders them with the verified cornell path-tracer
 * graph. The importer mirrors upstream Falcor's PBRTImporter subset.
 *
 * The cornell-box check is a correctness anchor: the LeftWall material is red
 * (reflectance 0.63,0.065,0.05) and the RightWall is green (0.14,0.45,0.091).
 * With the camera at +z looking down -z (pbrt's kInvertZ handling), a correct
 * (non-mirrored) render puts red on the left half and green on the right — so
 * this simultaneously verifies geometry, camera orientation, materials, and the
 * emissive area light.
 */

import {
    runPbrtScene,
    presentToCanvas,
    RenderGraph,
    createPass,
    Texture,
    ResourceType,
    ResourceFormat,
    ResourceBindFlags,
    type Scene,
    type Device,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, saveArtifact } from "../harness/registry.js";

const W = 512;
const H = 512;

async function renderScene(device: Device, scene: Scene, frames: number, exposure = 0): Promise<Uint8Array> {
    scene.camera.setAspectRatio(W / H);
    const graph = new RenderGraph(device, "Default");
    graph.onResize(W, H);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, emissiveSampler: "LightBVH" }), "PathTracer");
    graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Accumulate");
    graph.addPass(createPass(device, "ToneMapper", { autoExposure: false, exposureCompensation: exposure }), "ToneMapper");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.addEdge("PathTracer.color", "Accumulate.input");
    graph.addEdge("Accumulate.output", "ToneMapper.src");
    graph.markOutput("ToneMapper.dst");
    graph.setScene(scene);
    for (let f = 0; f < frames; f++) graph.execute(device.renderContext);
    const outTex = graph.getOutput("ToneMapper.dst")!;
    const present = new Texture(device, {
        type: ResourceType.Texture2D,
        width: W,
        height: H,
        mipLevels: 1,
        format: ResourceFormat.BGRA8Unorm,
        bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        name: "pbrt::present",
    });
    presentToCanvas(device, outTex, present.gpuTexture, "bgra8unorm");
    return device.renderContext.readTextureSubresource(present); // BGRA
}

gpuTest("PbrtImporter.cornellBox", async ({ device }) => {
    const base = "/tests/gpu/assets/pbrt";
    const src = await (await fetch(`${base}/cornell-box.pbrt`)).text();
    const scene = await runPbrtScene(device, src, base);

    const px = await renderScene(device, scene, 64, 2.5);
    await saveArtifact("pbrt-cornell-box", px, W, H);

    // Mean R/G over the left/right wall strips (the outer 12% columns, where the
    // colored side walls are) + a lit-fraction over the whole frame.
    let lR = 0, lG = 0, lN = 0, rR = 0, rG = 0, rN = 0, lit = 0;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const i = (y * W + x) * 4;
            const b = px[i]!, g = px[i + 1]!, r = px[i + 2]!;
            if (r + g + b > 20) lit++;
            if (x < W / 8) { lR += r; lG += g; lN++; }
            else if (x >= (7 * W) / 8) { rR += r; rG += g; rN++; }
        }
    }
    lR /= lN; lG /= lN; rR /= rN; rG /= rN;
    console.error(`# cornell: lit=${lit}/${W * H}  leftStrip(R=${lR.toFixed(1)},G=${lG.toFixed(1)})  rightStrip(R=${rR.toFixed(1)},G=${rG.toFixed(1)})`);

    // The box fills most of the frame.
    expectEq(lit > 0.6 * W * H, true, `cornell box too empty (${lit} lit px)`);
    // Left wall reads red, right wall reads green — verifies non-mirrored geometry+camera+materials.
    expectEq(lR > lG * 1.5, true, `left wall should read red (R=${lR.toFixed(1)} G=${lG.toFixed(1)})`);
    expectEq(rG > rR * 1.5, true, `right wall should read green (R=${rR.toFixed(1)} G=${rG.toFixed(1)})`);
});

gpuTest("PbrtImporter.veachMis", async ({ device }) => {
    const base = "/tests/gpu/assets/pbrt";
    const src = await (await fetch(`${base}/veach-mis.pbrt`)).text();
    const scene = await runPbrtScene(device, src, base);

    const px = await renderScene(device, scene, 64);
    await saveArtifact("pbrt-veach-mis", px, W, H);

    // Non-trivial, lit frame (conductor plates lit by several area lights + spheres).
    let lit = 0, bright = 0;
    for (let i = 0; i < W * H; i++) {
        const s = px[i * 4]! + px[i * 4 + 1]! + px[i * 4 + 2]!;
        if (s > 15) lit++;
        if (s > 400) bright++;
    }
    console.error(`# veach-mis: lit=${lit}/${W * H} bright=${bright}`);
    expectEq(lit > 0.25 * W * H, true, `veach-mis too empty (${lit} lit px)`);
    expectEq(bright > 50, true, `veach-mis has no bright light sources (${bright} px)`);
});
