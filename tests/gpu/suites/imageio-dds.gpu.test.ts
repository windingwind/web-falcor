/**
 * ImageIO.saveToDDS (BCEncoder in place of NVTT) against NVTT 3.1.6 itself: the same inputs
 * are compressed by both (tests/oracle/assets/nvtt, see its README), every file is decoded by
 * the GPU (WebGPU texture-compression-bc) and scored against the source. The web encoder must
 * come close to NVTT's quality: BC1–BC5 and BC7 (all modes, alpha and opaque inputs) within 0.5 dB, BC6H (mode 11
 * only) within a few dB. Uncompressed saves round-trip bit-exactly, and saveTextureToDDS
 * reads GPU textures back into the same files.
 */

import { Bitmap, ComputePass, CompressionMode, ImageIO, ResourceBindFlags, ResourceFormat, ResourceType, Texture } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const kDir = "/tests/oracle/assets/nvtt";

// Texel loads (a blit would clamp HDR values).
const kDecode = `
Texture2D<float4> gSrc;
RWTexture2D<float4> gDst;
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) { gDst[id.xy] = gSrc.Load(int3(id.xy, 0)); }
`;

gpuTest("ImageIO.ddsSaveMatchesNvttQuality", async ({ device }) => {
    const ctx = device.renderContext;
    const fetchBytes = async (name: string) => new Uint8Array(await (await fetch(`${kDir}/${name}`)).arrayBuffer());
    // PNGs load as BGRA8 (native FreeImage order); the scores use an RGBA8 copy.
    const bgra = (await Bitmap.createFromBytes(await fetchBytes("src-ldr.png"), "src-ldr.png", true))!;
    const rgba = new Uint8Array(bgra.getData());
    for (let i = 0; i < rgba.length; i += 4) [rgba[i], rgba[i + 2]] = [rgba[i + 2]!, rgba[i]!];
    const ldr = Bitmap.create(bgra.getWidth(), bgra.getHeight(), ResourceFormat.RGBA8Unorm, rgba);
    const hdrFloat = ImageIO.loadBitmapFromDDS(await fetchBytes("src-hdr.dds"));
    // BC6H's source is the RGBE image NVTT compressed (NVTT reads float DDS input as zeros).
    const hdr = (await Bitmap.createFromBytes(await fetchBytes("src-hdr.hdr"), "src-hdr.hdr", true))!;
    const [w, h] = [ldr.getWidth(), ldr.getHeight()];
    const pass = ComputePass.create(device, { modules: [{ sources: [{ string: kDecode, path: "Tests/ImageIODecode.cs.slang" }] }], csEntry: "main" });

    /** GPU-decoded texels of a DDS file (blit of the BC texture into RGBA32Float). */
    const decode = async (dds: Uint8Array) => {
        const tex = ImageIO.loadTextureFromDDS(device, dds, false);
        const dst = new Texture(device, { type: ResourceType.Texture2D, width: w, height: h, format: ResourceFormat.RGBA32Float, bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource });
        const root = pass.getRootVar();
        root["gSrc"] = tex;
        root["gDst"] = dst;
        pass.execute(ctx, w, h);
        return new Float32Array((await ctx.readTextureSubresource(dst, 0, 0)).buffer);
    };
    const srcLdr = new Uint8Array(ldr.getData());
    const srcHdr = new Float32Array(hdr.getData().buffer.slice(0));
    const psnrLdr = (texels: Float32Array, channels: number[]) => {
        let se = 0;
        for (let i = 0; i < w * h; i++) for (const c of channels) se += (texels[i * 4 + c]! * 255 - srcLdr[i * 4 + c]!) ** 2;
        const mse = se / (w * h * channels.length);
        return mse === 0 ? 99 : 10 * Math.log10((255 * 255) / mse);
    };
    // HDR: log-space PSNR (the usual BC6H metric).
    const psnrHdr = (texels: Float32Array) => {
        let se = 0;
        for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) se += (Math.log2(1 + Math.max(0, texels[i * 4 + c]!)) - Math.log2(1 + srcHdr[i * 4 + c]!)) ** 2;
        const mse = se / (w * h * 3);
        const peak = Math.log2(1 + Math.max(...srcHdr.filter((_, i) => i % 4 !== 3)));
        return 10 * Math.log10((peak * peak) / mse);
    };

    const cases: [string, CompressionMode, number[], number][] = [
        ["bc1", CompressionMode.BC1, [0, 1, 2], 0.5],
        ["bc2", CompressionMode.BC2, [0, 1, 2, 3], 0.5],
        ["bc3", CompressionMode.BC3, [0, 1, 2, 3], 0.5],
        ["bc4", CompressionMode.BC4, [0], 0.5],
        ["bc5", CompressionMode.BC5, [0, 1], 0.5],
        ["bc7", CompressionMode.BC7, [0, 1, 2, 3], 0.5],
    ];
    const lines: string[] = [];
    for (const [name, mode, channels, margin] of cases) {
        const web = psnrLdr(await decode(ImageIO.saveToDDS(ldr, mode)), channels);
        const nvtt = psnrLdr(await decode(await fetchBytes(`nvtt-${name}.dds`)), channels);
        lines.push(`${name} web ${web.toFixed(2)} dB / nvtt ${nvtt.toFixed(2)} dB`);
        expectEq(web > nvtt - margin, true, `${name}: web ${web} dB vs NVTT ${nvtt} dB`);
    }
    // Opaque input exercises BC7's RGB-only partitioned modes (0-3).
    const opaque = Bitmap.create(w, h, ResourceFormat.RGBA8Unorm, Uint8Array.from(srcLdr, (v, i) => (i % 4 === 3 ? 255 : v)));
    const srcAlpha = srcLdr.slice();
    srcLdr.set(new Uint8Array(opaque.getData()));
    const web7o = psnrLdr(await decode(ImageIO.saveToDDS(opaque, CompressionMode.BC7)), [0, 1, 2]);
    const nvtt7o = psnrLdr(await decode(await fetchBytes("nvtt-bc7-opaque.dds")), [0, 1, 2]);
    srcLdr.set(srcAlpha);
    lines.push(`bc7 opaque web ${web7o.toFixed(2)} dB / nvtt ${nvtt7o.toFixed(2)} dB`);
    expectEq(web7o > nvtt7o - 0.5, true, `bc7 opaque: web ${web7o} dB vs NVTT ${nvtt7o} dB`);
    const web6 = psnrHdr(await decode(ImageIO.saveToDDS(hdr, CompressionMode.BC6)));
    const nvtt6 = psnrHdr(await decode(await fetchBytes("nvtt-bc6.dds")));
    lines.push(`bc6 web ${web6.toFixed(2)} dB / nvtt ${nvtt6.toFixed(2)} dB (log2)`);
    expectEq(web6 > nvtt6 - 6, true, `bc6: web ${web6} dB vs NVTT ${nvtt6} dB`);

    // Uncompressed saves round-trip exactly; a texture saves like its bitmap.
    const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i]);
    console.error(`# imageio dds: ${lines.join(", ")}`);
    expectEq(same(ImageIO.loadBitmapFromDDS(ImageIO.saveToDDS(ldr)).getData(), srcLdr), true, "RGBA8 round-trip");
    expectEq(same(ImageIO.saveToDDS(bgra, CompressionMode.BC7), ImageIO.saveToDDS(ldr, CompressionMode.BC7)), true, "BGRA8 input swizzles like native");
    expectEq(same(ImageIO.loadBitmapFromDDS(ImageIO.saveToDDS(hdrFloat)).getData(), hdrFloat.getData()), true, "RGBA32F round-trip");
    const tex = new Texture(device, { type: ResourceType.Texture2D, width: w, height: h, format: ResourceFormat.RGBA8Unorm, mipLevels: 1, bindFlags: ResourceBindFlags.ShaderResource });
    tex.setSubresourceBlob(0, 0, srcLdr);
    expectEq(same(await ImageIO.saveTextureToDDS(ctx, tex, CompressionMode.BC1), ImageIO.saveToDDS(ldr, CompressionMode.BC1)), true, "saveTextureToDDS matches saveToDDS");
    // Generated mips: full chain, and a compressed texture saves its blocks as-is.
    const withMips = ImageIO.loadTextureFromDDS(device, ImageIO.saveToDDS(ldr, CompressionMode.BC3, true), false);
    expectEq(withMips.mipCount, Math.log2(w) + 1, "full mip chain");
    const resaved = await ImageIO.saveTextureToDDS(ctx, withMips);
    expectEq(same(resaved, ImageIO.saveToDDS(ldr, CompressionMode.BC3, true)), true, "BC texture re-saves bit-exactly");
});
