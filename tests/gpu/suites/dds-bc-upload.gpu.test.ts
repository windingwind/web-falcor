/**
 * BC texture upload — parse a real Bistro .dds and upload its compressed blocks
 * directly to a WebGPU BC-format texture (no CPU decompression), then sample it
 * through presentToCanvas and confirm it displays structured (non-uniform)
 * content. Proves the DDS -> BC-texture path works end to end, the missing
 * piece for texturing DDS-based scenes like Bistro.
 */

import { parseDDS, Texture, ResourceType, ResourceFormat, ResourceBindFlags, presentToCanvas } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("DDSUpload.samplesBistroBC1Texture", async ({ device }) => {
    expectEq(device.getSupportedFeatures().textureCompressionBC, true, "device supports texture-compression-bc");
    const lim = device.adapter.limits;
    console.error(`# device-limits: maxTextureArrayLayers=${lim.maxTextureArrayLayers} maxTextureDimension2D=${lim.maxTextureDimension2D} maxSampledTextures=${lim.maxSampledTexturesPerShaderStage}`);

    // A 2048x2048 BC1 (DXT1) base-color texture with real content.
    const buf = await (await fetch("/Falcor/media/Bistro_v5_2/Textures/Antenna_Metal_BaseColor.dds")).arrayBuffer();
    const img = parseDDS(buf, /*srgb*/ true);
    expectEq(img.width, 2048, "DDS width");
    expectEq(img.format, ResourceFormat.BC1UnormSrgb, "BC1 sRGB format");

    const tex = new Texture(device, {
        type: ResourceType.Texture2D,
        width: img.width,
        height: img.height,
        mipLevels: img.levels.length,
        format: img.format,
        bindFlags: ResourceBindFlags.ShaderResource,
        name: "Bistro::antennaBaseColor",
    });
    // Upload every mip (compressed block data — setSubresourceBlob is block-aware).
    img.levels.forEach((lv, mip) => tex.setSubresourceBlob(mip, 0, lv.data));

    // Sample the BC texture to a bgra8 target and read it back.
    const w = 256;
    const h = 256;
    const present = new Texture(device, {
        type: ResourceType.Texture2D,
        width: w,
        height: h,
        mipLevels: 1,
        format: ResourceFormat.BGRA8Unorm,
        bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        name: "DDSUpload::present",
    });
    presentToCanvas(device, tex, present.gpuTexture, "bgra8unorm");
    const pixels = await device.renderContext.readTextureSubresource(present);

    // The texture has structure: distinct colors + spatial variation.
    const colors = new Set<number>();
    let sum = 0;
    for (let i = 0; i < w * h; i++) {
        colors.add((pixels[i * 4]! << 16) | (pixels[i * 4 + 1]! << 8) | pixels[i * 4 + 2]!);
        sum += pixels[i * 4]! + pixels[i * 4 + 1]! + pixels[i * 4 + 2]!;
    }
    const mean = sum / (w * h * 3);
    console.error(`# dds-bc-upload: ${img.width}x${img.height} ${img.levels.length} mips, sampled distinctColors=${colors.size} mean=${mean.toFixed(1)}`);
    expectEq(colors.size > 200, true, `BC texture sampled as near-uniform (${colors.size} colors)`);
    expectEq(mean > 5 && mean < 250, true, `BC texture sampled to a degenerate value (mean ${mean})`);
});
