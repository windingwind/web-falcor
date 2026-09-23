/**
 * Transplanted FalcorTest GPU test: Utils/ImageProcessing (copyColorChannel).
 * Covers the renderable formats; RGBA8Snorm (not renderable in WebGPU) and
 * 16-bit unorm (not core WebGPU) destinations are not portable.
 */

import { ImageProcessing, ResourceBindFlags, ResourceFormat, TextureChannelFlags, float16ToFloat32, float32ToFloat16, getFormatChannelCount } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { Expect } from "../../harness/expect.js";

type Elem = "f32" | "u32" | "f16" | "i16" | "i8";
const kArrays = { f32: Float32Array, u32: Uint32Array, f16: Uint16Array, i16: Int16Array, i8: Int8Array };

/** Native's T(c) for c = i * 2.5 * (odd ? -1 : 1) (integers truncate and wrap to the type's width). */
function convert(elem: Elem, c: number): number {
    const t = Math.trunc(c);
    switch (elem) {
        case "f32":
            return Math.fround(c);
        case "u32":
            return t >>> 0;
        case "f16":
            return float32ToFloat16(c);
        case "i16":
            return (t << 16) >> 16;
        case "i8":
            return (t << 24) >> 24;
    }
}

gpuTest("FalcorTest.CopyColorChannel", async ({ device }) => {
    const w = 15;
    const h = 3;
    const ip = new ImageProcessing(device);
    const e = new Expect();
    const cases: [Elem, ResourceFormat, ResourceFormat][] = [
        ["f32", ResourceFormat.RGBA32Float, ResourceFormat.RGBA32Float],
        ["f32", ResourceFormat.RGBA32Float, ResourceFormat.RG32Float],
        ["f32", ResourceFormat.RGBA32Float, ResourceFormat.R32Float],
        ["u32", ResourceFormat.RGBA32Uint, ResourceFormat.RGBA32Uint],
        ["u32", ResourceFormat.RGBA32Uint, ResourceFormat.RG32Uint],
        ["u32", ResourceFormat.RGBA32Uint, ResourceFormat.R32Uint],
        ["f16", ResourceFormat.RGBA16Float, ResourceFormat.RGBA16Float],
        ["f16", ResourceFormat.RGBA16Float, ResourceFormat.RG16Float],
        ["f16", ResourceFormat.RGBA16Float, ResourceFormat.R16Float],
        ["i16", ResourceFormat.RGBA16Int, ResourceFormat.RGBA16Int],
        ["i16", ResourceFormat.RGBA16Int, ResourceFormat.RG16Int],
        ["i16", ResourceFormat.RGBA16Int, ResourceFormat.R16Int],
        ["i8", ResourceFormat.RGBA8Int, ResourceFormat.RGBA8Int],
        ["i8", ResourceFormat.RGBA8Int, ResourceFormat.RG8Int],
        ["i8", ResourceFormat.RGBA8Int, ResourceFormat.R8Int],
    ];
    for (const [elem, srcFormat, dstFormat] of cases) {
        const srcChannels = getFormatChannelCount(srcFormat);
        const dstChannels = getFormatChannelCount(dstFormat);
        const data = new kArrays[elem](w * h * srcChannels);
        for (let i = 0; i < data.length; i++) data[i] = convert(elem, i * 2.5 * (i % 2 ? -1 : 1));
        const src = device.createTexture2D(w, h, srcFormat, 1, 1, data, ResourceBindFlags.ShaderResource);
        const dst = device.createTexture2D(w, h, dstFormat, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        const masks = [TextureChannelFlags.Red, TextureChannelFlags.Green, TextureChannelFlags.Blue, TextureChannelFlags.Alpha];
        for (let c = 0; c < 4; c++) {
            ip.copyColorChannel(device.renderContext, src, dst, masks[c]!);
            const raw = await device.renderContext.readTextureSubresource(dst);
            const result = new kArrays[elem](raw.slice().buffer);
            for (let j = 0; j < w * h; j++) {
                const ref = data[j * srcChannels + c]!;
                for (let k = 0; k < dstChannels; k++) {
                    const value = result[j * dstChannels + k]!;
                    const ok = elem === "f16" ? float16ToFloat32(value) === float16ToFloat32(ref) : value === ref;
                    e.check(ok, () => `${ResourceFormat[dstFormat]} channel ${c} j=${j} k=${k}: ${value} != ${ref}`);
                }
            }
        }
    }
    e.done("CopyColorChannel");
});
