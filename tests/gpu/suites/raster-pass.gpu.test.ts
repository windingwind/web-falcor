/**
 * M2 raster GPU tests: FullScreenPass using the unmodified upstream vertex
 * shader (Core/Pass/FullScreenPass.vs.slang) + merged VS/PS parameter block.
 */

import { Fbo, FullScreenPass, ResourceFormat, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("FullScreenPass.tintedTextureBlit", async ({ device }) => {
    const w = 8, h = 8;
    // Source texture: solid white.
    const src = device.createTexture2D(w, h, ResourceFormat.RGBA8Unorm, 1, 1, new Uint8Array(w * h * 4).fill(255));
    const fbo = Fbo.create2D(device, w, h, ResourceFormat.RGBA8Unorm);
    const pass = FullScreenPass.create(device, { path: "FullScreenTest.ps.slang" });

    const root = pass.getRootVar();
    root["gSrcTex"] = src;
    root["gSampler"] = device.createSampler();
    root["TintCB"]["gTint"] = [1.0, 0.5, 0.0, 1.0];

    const ctx = device.renderContext;
    ctx.clearTexture(fbo.getColorTexture(0)!, [0, 0, 1, 1]);
    pass.execute(ctx, fbo);

    const px = await ctx.readTextureSubresource(fbo.getColorTexture(0)!);
    const center = (4 * w + 4) * 4;
    expectEq(px[center], 255, "r = white * tint.r");
    expectEq(px[center + 1]! >= 127 && px[center + 1]! <= 128, true, `g = white * 0.5 (got ${px[center + 1]})`);
    expectEq(px[center + 2], 0, "b = white * tint.b");
    src.destroy();
});

gpuTest("FullScreenPass.uniformUpdateBetweenDraws", async ({ device }) => {
    const w = 4, h = 4;
    const src = device.createTexture2D(w, h, ResourceFormat.RGBA8Unorm, 1, 1, new Uint8Array(w * h * 4).fill(255));
    const fbo = Fbo.create2D(device, w, h, ResourceFormat.RGBA8Unorm);
    const pass = FullScreenPass.create(device, { path: "FullScreenTest.ps.slang" });
    const root = pass.getRootVar();
    root["gSrcTex"] = src;
    root["gSampler"] = device.createSampler();

    const ctx = device.renderContext;
    root["TintCB"]["gTint"] = [1, 0, 0, 1];
    pass.execute(ctx, fbo);
    let px = await ctx.readTextureSubresource(fbo.getColorTexture(0)!);
    expectEq([px[0], px[1]], [255, 0], "first draw red");

    root["TintCB"]["gTint"] = [0, 1, 0, 1];
    pass.execute(ctx, fbo);
    px = await ctx.readTextureSubresource(fbo.getColorTexture(0)!);
    expectEq([px[0], px[1]], [0, 255], "second draw green after cbuffer update");
    src.destroy();
});
