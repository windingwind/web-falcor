/**
 * ImageLoader options that used to be accepted but ignored: mip generation +
 * mip level selection, array-slice clamping, Fixed output size (the image's
 * own size regardless of the graph dims) and an explicit output format.
 */

import { RenderGraph, ResourceFormat, createPass, type Texture } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const kImage = "test_images/smoke_puff.png";

async function loadGraph(device: import("@web-falcor/falcor").Device, props: Record<string, unknown>, dims: [number, number]): Promise<Texture> {
    const graph = new RenderGraph(device, "IL");
    graph.addPass(createPass(device, "ImageLoader", { filename: kImage, srgb: false, ...props }), "Img");
    graph.markOutput("Img.dst");
    await graph.init();
    graph.onResize(dims[0], dims[1]);
    graph.execute(device.renderContext);
    return graph.getOutput("Img.dst")!;
}

function stats(px: Float32Array): { mean: number; nonUniform: boolean } {
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < px.length; i += 4) {
        sum += px[i]!;
        min = Math.min(min, px[i]!);
        max = Math.max(max, px[i]!);
    }
    return { mean: sum / (px.length / 4), nonUniform: max - min > 0.05 };
}

gpuTest("ImageLoaderOpts.fixedSizeFormatAndMipSelection", async ({ device }) => {
    const ctx = device.renderContext;
    const bitmap = await createImageBitmap(await (await fetch(`/Falcor/media/${kImage}`)).blob());

    // Default: graph dims, image format (8-bit, blitted/rescaled).
    const dflt = await loadGraph(device, {}, [64, 48]);
    expectEq([dflt.width, dflt.height], [64, 48], "Default output size follows the graph");
    expectEq(dflt.format, ResourceFormat.RGBA8Unorm, "Unknown output format follows the image");

    // Fixed + explicit float format: the image's own size, mip 0.
    const mip0 = await loadGraph(device, { outputSize: "Fixed", outputFormat: "RGBA32Float", mips: true, mipLevel: 0 }, [64, 48]);
    expectEq([mip0.width, mip0.height], [bitmap.width, bitmap.height], "Fixed output size = image size");
    expectEq(mip0.format, ResourceFormat.RGBA32Float, "explicit output format honoured");
    const px0 = new Float32Array((await ctx.readTextureSubresource(mip0)).buffer);

    // Same image at mip 2 (generated chain), blitted back up: same mean, blurrier.
    const mip2 = await loadGraph(device, { outputSize: "Fixed", outputFormat: "RGBA32Float", mips: true, mipLevel: 2, arrayIndex: 7 }, [64, 48]);
    const px2 = new Float32Array((await ctx.readTextureSubresource(mip2)).buffer);
    const s0 = stats(px0);
    const s2 = stats(px2);
    expectEq(s0.nonUniform && s2.nonUniform, true, "both outputs carry image content");
    expectEq(Math.abs(s0.mean - s2.mean) < 0.02 * Math.max(s0.mean, 1e-3) + 1e-3, true, `mip chain preserves the mean (${s0.mean.toFixed(4)} vs ${s2.mean.toFixed(4)})`);
    let maxDiff = 0;
    for (let i = 0; i < px0.length; i += 4) maxDiff = Math.max(maxDiff, Math.abs(px0[i]! - px2[i]!));
    expectEq(maxDiff > 0.02, true, `mip 2 differs from mip 0 (max diff ${maxDiff.toFixed(3)})`);
    // arrayIndex 7 on a 2D image clamps to slice 0 (native std::min) instead of failing.
});
