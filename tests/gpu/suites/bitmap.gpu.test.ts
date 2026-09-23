/**
 * Bitmap paths that need the browser's decoders (JPEG, BMP) and
 * Texture.captureToFile. Fixtures come from scripts/gen-bitmap-fixtures.py.
 */

import { Bitmap, BitmapExportFlags, BitmapFileFormat, ResourceBindFlags, ResourceFormat, Texture, ResourceType, float16ToFloat32 } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const kDir = "/packages/falcor/tests/fixtures/bitmap/";

async function check(file: string, expectedName: string, tolerance: number): Promise<void> {
    const expected = (await (await fetch(`${kDir}${expectedName}.expected.json`)).json()) as { width: number; height: number; format: string; data: number[] };
    const bmp = await Bitmap.createFromFile(`${kDir}${file}`, true);
    expectEq(bmp !== null, true, `${file} loads`);
    expectEq(ResourceFormat[bmp!.format], expected.format, `${file} format`);
    expectEq(`${bmp!.width}x${bmp!.height}`, `${expected.width}x${expected.height}`, `${file} size`);
    let worst = 0;
    expected.data.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - bmp!.data[i]!))));
    console.error(`# bitmap ${file}: worst difference ${worst}`);
    expectEq(worst <= tolerance, true, `${file} pixels within ${tolerance} (worst ${worst})`);
}

gpuTest("Bitmap.browserDecodedFormats", async () => {
    await check("rgb24.bmp", "rgb24", 0);
    await check("gray8.bmp", "gray8-bmp", 0);
    await check("palette8.bmp", "palette8", 0);
    await check("rgb.jpg", "rgb-jpg", 3);
    await check("gray.jpg", "gray-jpg", 3);
});

gpuTest("Bitmap.saveImageBrowserFormats", async () => {
    const W = 8;
    const H = 4;
    const rgba = Uint8Array.from({ length: W * H * 4 }, (_, i) => (i % 4 === 3 ? 255 : 40 + ((i * 7) % 160)));
    const bmp = await Bitmap.saveImage(W, H, BitmapFileFormat.BmpFile, BitmapExportFlags.None, ResourceFormat.RGBA8Unorm, true, rgba);
    const back = (await Bitmap.createFromBytes(bmp, "out.bmp", true))!;
    let exact = true;
    for (let i = 0; i < W * H; i++) exact &&= back.data[i * 4] === rgba[i * 4 + 2] && back.data[i * 4 + 2] === rgba[i * 4];
    expectEq(exact, true, "BMP round trip is exact");
    const jpg = await Bitmap.saveImage(W, H, BitmapFileFormat.JpegFile, BitmapExportFlags.None, ResourceFormat.RGBA8Unorm, true, rgba);
    expectEq(jpg[0] === 0xff && jpg[1] === 0xd8, true, "JPEG signature");
    const jback = (await Bitmap.createFromBytes(jpg, "out.jpg", true))!;
    expectEq(jback.format, ResourceFormat.BGRX8Unorm, "JPEG loads as BGRX8");
});

gpuTest("Bitmap.textureCaptureToFile", async ({ device }) => {
    const W = 6;
    const H = 4;
    const tex = new Texture(device, { type: ResourceType.Texture2D, width: W, height: H, format: ResourceFormat.RG32Float, bindFlags: ResourceBindFlags.ShaderResource });
    const values = Float32Array.from({ length: W * H * 2 }, (_, i) => i * 0.25 - 3);
    tex.setSubresourceBlob(0, 0, values);
    // Two-channel float widens to RGBA32Float (native blits); EXR defaults to half, RGB.
    const exr = await tex.captureToFile(0, 0, "capture.exr", BitmapFileFormat.ExrFile, BitmapExportFlags.None, false);
    const back = (await Bitmap.createFromBytes(exr, "capture.exr", true))!;
    expectEq(back.format, ResourceFormat.RGBA16Float, "all-half EXR loads as RGBA16Float");
    const h = new Uint16Array(back.data.slice().buffer);
    let ok = true;
    for (let i = 0; i < W * H; i++) {
        ok &&= float16ToFloat32(h[i * 4]!) === values[i * 2] && float16ToFloat32(h[i * 4 + 1]!) === values[i * 2 + 1];
        ok &&= h[i * 4 + 2] === 0 && float16ToFloat32(h[i * 4 + 3]!) === 1;
    }
    expectEq(ok, true, "captured values (quarter steps are exact in half)");

    const ldr = new Texture(device, { type: ResourceType.Texture2D, width: W, height: H, format: ResourceFormat.RGBA8Unorm, bindFlags: ResourceBindFlags.ShaderResource });
    const px = Uint8Array.from({ length: W * H * 4 }, (_, i) => (i * 29) & 0xff);
    ldr.setSubresourceBlob(0, 0, px);
    expectEq(Bitmap.getFileExtFromResourceFormat(ldr.format), "png", "8-bit captures as PNG");
    const png = await ldr.captureToFile(0, 0, "capture.png", BitmapFileFormat.PngFile, BitmapExportFlags.ExportAlpha, false);
    const pback = (await Bitmap.createFromBytes(png, "capture.png", true))!;
    let same = true;
    for (let i = 0; i < W * H; i++) same &&= pback.data[i * 4] === px[i * 4 + 2] && pback.data[i * 4 + 3] === px[i * 4 + 3];
    expectEq(same, true, "PNG capture round trip");
});
