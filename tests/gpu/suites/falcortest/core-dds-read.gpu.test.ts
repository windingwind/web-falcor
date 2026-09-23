/**
 * Transplanted FalcorTest GPU tests: Core/DDSReadTests. Each DDS file loads through
 * createTextureFromFile (native ImageIO::loadTextureFromDDS), keeps its BC format and
 * is blitted to RGBA32Float. Native's `diff` kernel compares the blit with itself
 * (`rv = tex[...]`), so it only checks that loading and blitting work; that check is
 * kept, and the blit is also compared with the reference PNG, which native wrote
 * with its `readback` kernel (saturate(v) * 255, truncated); Bitmap reads it as
 * bottom-up BGRA, like FreeImage. All 19 match native exactly.
 * §9: WebGPU needs BC textures sized in whole blocks, so odd sizes load padded to a
 * multiple of 4 and blit to a target of the image's own size. Native's references
 * for those sizes are exactly what a linear blit from the padded blocks gives, so
 * that is what its GPU did with the unpadded texture too.
 */

import { Bitmap, ResourceBindFlags, ResourceFormat, createTextureFromFile } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kCases: [string, ResourceFormat][] = [
    ["BC1Unorm", ResourceFormat.BC1Unorm],
    ["BC1UnormSrgb", ResourceFormat.BC1UnormSrgb],
    ["BC2Unorm", ResourceFormat.BC2Unorm],
    ["BC2UnormSrgb", ResourceFormat.BC2UnormSrgb],
    ["BC2UnormSrgbTiny", ResourceFormat.BC2UnormSrgb],
    ["BC3Unorm", ResourceFormat.BC3Unorm],
    ["BC3UnormAlpha", ResourceFormat.BC3Unorm],
    ["BC3UnormAlphaTiny", ResourceFormat.BC3Unorm],
    ["BC3UnormSrgb", ResourceFormat.BC3UnormSrgb],
    ["BC3UnormSrgbOdd", ResourceFormat.BC3UnormSrgb],
    ["BC3UnormSrgbTiny", ResourceFormat.BC3UnormSrgb],
    ["BC4Unorm", ResourceFormat.BC4Unorm],
    ["BC5Unorm", ResourceFormat.BC5Unorm],
    ["BC5UnormTiny", ResourceFormat.BC5Unorm],
    ["BC6HU16", ResourceFormat.BC6HU16],
    ["BC7Unorm", ResourceFormat.BC7Unorm],
    ["BC7UnormOdd", ResourceFormat.BC7Unorm],
    ["BC7UnormSrgb", ResourceFormat.BC7UnormSrgb],
    ["BC7UnormTiny", ResourceFormat.BC7Unorm],
];

const url = (name: string) => `/Falcor/data/tests/${name}`;

for (const [name, format] of kCases) {
    gpuTest(`FalcorTest.${name}`, async ({ device }) => {
        const e = new Expect();
        // loadAsSrgb = false: sRGB formats come from the file itself.
        const tex = await createTextureFromFile(device, url(`${name}.dds`), false, false);
        e.check(tex !== null, () => "DDS failed to load");
        if (!tex) return e.done(name);
        e.check(tex.format === format, () => `format ${ResourceFormat[tex.format]} != ${ResourceFormat[format]}`);
        const refBitmap = await Bitmap.createFromFile(url(`${name}-ref.png`), true);
        e.check(refBitmap !== null, () => "reference PNG missing");

        // The image's own size (the texture may be block-padded, see §9 above).
        const [w, h] = refBitmap ? [refBitmap.width, refBitmap.height] : [tex.width, tex.height];
        const dst = device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        device.renderContext.blit(tex, dst);

        const ctx = new GPUUnitTestContext(device);
        ctx.createProgram("Tests/Core/DDSReadTests.cs.slang", "diff");
        ctx.allocateStructuredBuffer("difference", 4 * dst.width * dst.height);
        ctx.vars()["ref"] = await createTextureFromFile(device, url(`${name}-ref.png`), false, false);
        ctx.vars()["tex"] = dst;
        ctx.vars()["CB"]["sz"] = [dst.width, dst.height];
        ctx.runProgram(dst.width, dst.height, 1);
        const diff = await ctx.readBuffer("difference", Float32Array);
        e.check(diff.every((d) => d === 0), () => "difference image is not uniformly 0");

        if (refBitmap) {
            const bytes = await device.renderContext.readTextureSubresource(dst, 0);
            const pixels = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
            const ref = refBitmap.data;
            const channels = ref.length / (refBitmap.width * refBitmap.height);
            const bgra = refBitmap.format === ResourceFormat.BGRA8Unorm || refBitmap.format === ResourceFormat.BGRX8Unorm;
            let worst = 0;
            for (let y = 0; y < refBitmap.height; y++)
                for (let x = 0; x < refBitmap.width; x++)
                    for (let c = 0; c < Math.min(channels, 4); c++) {
                        const v = Math.trunc(Math.min(Math.max(pixels[(y * dst.width + x) * 4 + c]!, 0), 1) * 255);
                        const refRow = refBitmap.height - 1 - y;
                        worst = Math.max(worst, Math.abs(v - ref[(refRow * refBitmap.width + x) * channels + (bgra && c < 3 ? 2 - c : c)]!));
                    }
            console.error(`# ${name}: ${tex.width}x${tex.height} (${refBitmap.width}x${refBitmap.height}), worst 8-bit difference from native ${worst}`);
            e.check(worst === 0, () => `worst 8-bit difference from the native reference: ${worst}`);
        }
        e.done(name);
    });
}

gpuTest("FalcorTest.BC7UnormBroken", async ({ device }) => {
    const e = new Expect();
    const tex = await createTextureFromFile(device, url("BC7UnormBroken.dds"), false, false);
    e.check(tex === null, () => "a broken DDS file loaded");
    e.done("BC7UnormBroken");
});
