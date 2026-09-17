/**
 * pbrt `Texture` directives: a named `imagemap` bound to a material parameter.
 * The pbrt-v4 scenes ship their images as TGA, which browsers cannot decode, so
 * this also covers the CPU TGA path (native gets it from FreeImage).
 *
 * Fetch the images with: npm run download:scenes -- bathroom
 */

import { RenderGraph, createPass, decodeTGA, runPbrtScene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

const size = 128;

gpuTest("PBRTTextures.imagemapReachesTheMaterial", async ({ device }) => {
    if (!(await fetch("/Falcor/media/bathroom/textures/rug.tga", { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/bathroom missing (npm run download:scenes -- bathroom)");
    }
    const source = await (await fetch("/tests/oracle/assets/pbrt-textured.pbrt")).text();
    // Texture paths resolve against the media root (the scene references bathroom/textures/).
    const scene = await runPbrtScene(device, source, "/Falcor/media");
    scene.camera.setAspectRatio(1.0);
    expectEq(scene.stats.textures > 0, true, `the imagemap was loaded (${scene.stats.textures} textures)`);

    const graph = new RenderGraph(device, "PbrtTextured");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBufferRT");
    graph.markOutput("GBufferRT.diffuseOpacity");
    graph.markOutput("GBufferRT.texC");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);

    const albedo = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRT.diffuseOpacity")!)).buffer);
    const texC = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRT.texC")!)).buffer);

    // Ground truth: the decoded TGA itself. Both native Falcor and this port pass
    // pbrt's uvs through unchanged and sample with a top-left origin, so v = 0 is
    // the image's first row (pbrt's own renderer flips v; Falcor does not).
    const tgaBytes = await (await fetch("/Falcor/media/bathroom/textures/rug.tga")).arrayBuffer();
    const image = decodeTGA(tgaBytes);
    const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

    /** Bilinear fetch of the texture's red channel — GBufferRT samples mip 0 by default. */
    const bilinearRed = (u: number, v: number) => {
        const x = u * image.width - 0.5;
        const y = v * image.height - 0.5;
        const x0 = Math.floor(x);
        const y0 = Math.floor(y);
        const fx = x - x0;
        const fy = y - y0;
        const clampX = (i: number) => Math.max(0, Math.min(image.width - 1, i));
        const clampY = (i: number) => Math.max(0, Math.min(image.height - 1, i));
        const at = (i: number, j: number) => image.rgba[(clampY(j) * image.width + clampX(i)) * 4]! / 255;
        return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
    };

    let compared = 0;
    let matched = 0;
    let worst = 0;
    let sumRendered = 0;
    let sumExpected = 0;
    for (let i = 0; i < size * size; i++) {
        const u = texC[i * 2]!;
        const v = texC[i * 2 + 1]!;
        if (!Number.isFinite(u) || (u === 0 && v === 0)) continue; // background
        const r = albedo[i * 4]!;
        if (r === 0 && albedo[i * 4 + 1] === 0 && albedo[i * 4 + 2] === 0) continue;
        compared++;
        const expected = srgbToLinear(bilinearRed(u, v));
        sumRendered += r;
        sumExpected += expected;
        const diff = Math.abs(r - expected);
        worst = Math.max(worst, diff);
        if (diff < 0.02) matched++;
    }
    console.error(`# pbrt imagemap: ${compared} shaded pixels, ${matched} match the bilinear mip-0 fetch, worst |diff| ${worst.toFixed(3)}, mean ${(sumRendered / compared).toFixed(4)} vs ${(sumExpected / compared).toFixed(4)}`);

    expectEq(matched / Math.max(compared, 1) > 0.95, true, `the rendered albedo is the sampled texture (${matched}/${compared})`);
    expectEq(compared > 5000, true, `the textured quad fills the frame (${compared} shaded pixels)`);
});
