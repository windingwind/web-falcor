/**
 * Alpha mode derivation mirroring BasicMaterial::updateAlphaMode: the material is
 * alpha-tested (Mask) only when its base color alpha can drop below the threshold —
 * the texture's alpha range when textured (TextureAnalyzer parity). The header's
 * alpha texture handle is the base color handle (Material::updateTextureHandle).
 * An explicit header.alphaMode wins. Verified through VBufferRT's alpha test.
 */

import { RenderGraph, Scene, TextureManager, AlphaMode, TextureHandleMode, packTextureHandle, createPass, float2, float3, float4 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("MaterialAlpha.textureAlphaDrivesAlphaTest", async ({ device }) => {
    // One triangle far larger than the view frustum footprint at z=0 (fully covers the frame);
    // the view samples a single texel of the 4x4 base color texture (uv ~ 0.25).
    const vertices = [
        { position: new float3(-20, -20, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(60, -20, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(-20, 60, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    const size = 32;
    const ctx = device.renderContext;
    const makeBitmap = async (alpha: number) => {
        const canvas = new OffscreenCanvas(4, 4);
        const c2d = canvas.getContext("2d")!;
        c2d.fillStyle = `rgba(200,100,50,${alpha})`;
        c2d.fillRect(0, 0, 4, 4);
        return createImageBitmap(canvas, { premultiplyAlpha: "none" });
    };
    const hitsFor = async (texAlpha: number | null, constAlpha: number, header?: { alphaMode?: AlphaMode; alphaThreshold?: number }) => {
        const tm = new TextureManager();
        const basic: { baseColor: float4; texBaseColor?: number } = { baseColor: new float4(0.8, 0.4, 0.2, constAlpha) };
        if (texAlpha !== null) basic.texBaseColor = packTextureHandle(TextureHandleMode.Texture, tm.addTexture({ bitmap: await makeBitmap(texAlpha), srgb: true }));
        const scene = new Scene(device, [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }], [{ basic, header }], [], tm);
        scene.camera.setPosition(new float3(0, 0, 2));
        scene.camera.setTarget(new float3(0, 0, 0));
        scene.camera.setAspectRatio(1);
        const g = new RenderGraph(device, "VB");
        g.addPass(createPass(device, "VBufferRT", { samplePattern: "Center", useAlphaTest: true }), "Pass");
        g.markOutput("Pass.vbuffer");
        g.onResize(size, size);
        g.setScene(scene);
        await g.init();
        g.execute(ctx);
        const vb = new Uint32Array((await ctx.readTextureSubresource(g.getOutput("Pass.vbuffer")!)).buffer);
        let hits = 0;
        for (let i = 0; i < size * size; i++) if (vb[i * 4] !== 0) hits++;
        return hits;
    };
    const all = size * size;
    // Native: the alpha test samples the base color texture's alpha; without a texture the
    // uniform fallback is 1, so a constant alpha below the threshold never discards.
    expectEq(await hitsFor(null, 1.0), all, "untextured: fully covered");
    expectEq(await hitsFor(null, 0.2), all, "untextured alpha 0.2: Mask mode but the test samples 1 (native)");
    // Textured: updateAlphaMode derives Mask only when the texture's min alpha < threshold.
    expectEq(await hitsFor(1.0, 1.0), all, "opaque texture: covered");
    expectEq(await hitsFor(0.2, 1.0), 0, "texture alpha 0.2 < 0.5: Mask, discarded");
    expectEq(await hitsFor(0.6, 1.0), all, "texture alpha 0.6 >= 0.5: Opaque");
    expectEq(await hitsFor(0.6, 1.0, { alphaThreshold: 0.9 }), 0, "threshold 0.9 > 0.6: Mask, discarded");
    expectEq(await hitsFor(0.2, 1.0, { alphaMode: AlphaMode.Opaque }), all, "explicit Opaque header wins");
});

gpuTest("MaterialAlpha.textureAlphaRange", async () => {
    // Half of the texels transparent: range [0, 1] -> Mask; a fully opaque texture -> Opaque.
    const make = async (alphaLeft: number) => {
        const canvas = new OffscreenCanvas(4, 2);
        const c2d = canvas.getContext("2d")!;
        c2d.fillStyle = `rgba(255,0,0,${alphaLeft})`;
        c2d.fillRect(0, 0, 2, 2);
        c2d.fillStyle = "rgba(0,255,0,1)";
        c2d.fillRect(2, 0, 2, 2);
        return createImageBitmap(canvas, { premultiplyAlpha: "none" });
    };
    const tm = new TextureManager();
    const half = tm.addTexture({ bitmap: await make(0), srgb: true });
    const full = tm.addTexture({ bitmap: await make(1), srgb: true });
    const halfRange = tm.getAlphaRange(half)!;
    const fullRange = tm.getAlphaRange(full)!;
    expectClose(halfRange[0], 0, 1e-6, "min alpha (transparent texels)");
    expectClose(halfRange[1], 1, 1e-6, "max alpha");
    expectClose(fullRange[0], 1, 1e-6, "opaque texture min alpha");
    expectEq(tm.getAlphaRange(half), halfRange, "range is cached");
});
