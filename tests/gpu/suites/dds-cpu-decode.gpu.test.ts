/**
 * CPU BC decode equivalence — the CPU BC1/BC3/BC5 decoders (decodeDDSToRGBA)
 * must agree with the GPU's hardware BC decode. We upload the same DDS mip two
 * ways: as a BC texture (hardware-decoded) and as an RGBA8 texture built from
 * decodeDDSToRGBA, sample both 1:1, and require them to match within BC's
 * endpoint-rounding tolerance. This validates the decoder that feeds the
 * existing RGBA8 material-texture path for DDS scenes (Bistro).
 */

import {
    parseDDS,
    decodeDDSToRGBA,
    Texture,
    ResourceType,
    ResourceFormat,
    ResourceBindFlags,
    presentToCanvas,
} from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

async function sample1to1(device: any, tex: Texture, size: number): Promise<Uint8Array> {
    const present = new Texture(device, {
        type: ResourceType.Texture2D,
        width: size,
        height: size,
        mipLevels: 1,
        format: ResourceFormat.BGRA8Unorm,
        bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        name: "cpudecode::present",
    });
    presentToCanvas(device, tex, present.gpuTexture, "bgra8unorm");
    return device.renderContext.readTextureSubresource(present);
}

// Compare a texture decoded on GPU vs CPU. The present target is BGRA8Unorm, so
// readback index 0=B, 1=G, 2=R. `cmp` lists which of those indices to compare:
// BC1/BC3 compare all colour channels; BC5 skips blue (index 0) because the GPU
// leaves it 0 while our CPU decoder reconstructs normal-Z there — that's the
// decoder doing more, not diverging on the stored R/G data.
async function checkFormat(device: any, url: string, srgb: boolean, expectFmt: ResourceFormat, tol: number, cmp = [0, 1, 2]) {
    const buf = await (await fetch(url)).arrayBuffer();
    const img = parseDDS(buf, srgb);
    expectEq(img.format, expectFmt, `${url} format`);

    // Pick a mid-size mip (256px cap) so we compare 1:1 without resampling.
    const cpu = decodeDDSToRGBA(buf, srgb, 256);
    const size = cpu.width;
    expectEq(cpu.width === cpu.height, true, "square mip picked");

    // The matching BC mip: find the level whose width == cpu.width.
    const level = img.levels.find((l) => l.width === size)!;
    const bcTex = new Texture(device, {
        type: ResourceType.Texture2D,
        width: size,
        height: size,
        mipLevels: 1,
        format: img.format,
        bindFlags: ResourceBindFlags.ShaderResource,
        name: "cpudecode::bc",
    });
    bcTex.setSubresourceBlob(0, 0, level.data);

    // The BGRA8 present target is sRGB-unaware; to compare against a *_UnormSrgb
    // BC texture (which decodes to linear on sample), upload the CPU RGBA as the
    // same colorspace so presentToCanvas treats both identically.
    const rgbaTex = new Texture(device, {
        type: ResourceType.Texture2D,
        width: size,
        height: size,
        mipLevels: 1,
        format: srgb ? ResourceFormat.RGBA8UnormSrgb : ResourceFormat.RGBA8Unorm,
        bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget,
        name: "cpudecode::rgba",
    });
    rgbaTex.setSubresourceBlob(0, 0, cpu.rgba);

    const bcPix = await sample1to1(device, bcTex, size);
    const cpuPix = await sample1to1(device, rgbaTex, size);

    let maxDiff = 0;
    let sumDiff = 0;
    let n = 0;
    for (let i = 0; i < size * size; i++) {
        for (const c of cmp) {
            const d = Math.abs(bcPix[i * 4 + c]! - cpuPix[i * 4 + c]!);
            maxDiff = Math.max(maxDiff, d);
            sumDiff += d;
            n++;
        }
    }
    const meanDiff = sumDiff / n;
    console.error(`# cpu-decode ${url.split("/").pop()}: ${size}px maxDiff=${maxDiff} meanDiff=${meanDiff.toFixed(2)}`);
    expectEq(meanDiff < tol, true, `CPU decode diverges from GPU (meanDiff ${meanDiff.toFixed(2)} >= ${tol})`);
}

gpuTest("DDSCpuDecode.bc1MatchesHardware", async ({ device }) => {
    await checkFormat(device, "/Falcor/media/Bistro_v5_2/Textures/Antenna_Metal_BaseColor.dds", true, ResourceFormat.BC1UnormSrgb, 3.0);
});

gpuTest("DDSCpuDecode.bc3MatchesHardware", async ({ device }) => {
    await checkFormat(device, "/Falcor/media/Bistro_v5_2/Textures/Ashtray_BaseColor.dds", true, ResourceFormat.BC3UnormSrgb, 3.0);
});

gpuTest("DDSCpuDecode.bc5MatchesHardware", async ({ device }) => {
    await checkFormat(device, "/Falcor/media/Bistro_v5_2/Textures/Antenna_Metal_Normal.dds", false, ResourceFormat.BC5Unorm, 4.0, /*compare G,R; skip blue*/ [1, 2]);
});
