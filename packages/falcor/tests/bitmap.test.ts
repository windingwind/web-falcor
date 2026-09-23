/**
 * Bitmap (Utils/Image/Bitmap): FreeImage's load-format rules against
 * fixtures from an independent writer (scripts/gen-bitmap-fixtures.py), and
 * saveImage round trips. JPEG/BMP/GIF go through the browser's decoders and
 * are covered by the GPU suite.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Bitmap, BitmapExportFlags, BitmapFileFormat, BitmapImportFlags } from "../src/Utils/Image/Bitmap.js";
import { ResourceFormat, getFormatChannelCount, getFormatType, getNumChannelBits, FormatType } from "../src/Core/API/Formats.js";
import { float16ToFloat32, float32ToFloat16 } from "../src/Utils/Math/Float16.js";

const dir = new URL("./fixtures/bitmap/", import.meta.url);
const read = (name: string) => new Uint8Array(readFileSync(new URL(name, dir)));

describe("Bitmap.createFromBytes", () => {
    const cases: [string, string][] = [
        ["gray8.png", "gray8"],
        ["rgb8.png", "rgb8"],
        ["rgba8.png", "rgba8"],
        ["gray-alpha8.png", "gray-alpha8"],
        ["palette.png", "palette"],
        ["palette-gray-ramp.png", "palette-gray-ramp"],
        ["gray16.png", "gray16"],
        ["rgb16.png", "rgb16"],
        ["rgba8-adam7.png", "rgba8-adam7"],
        ["gray2.png", "gray2"],
        ["rgb.tga", "rgb"],
        ["rgba-rle.tga", "rgba-rle"],
        ["float.pfm", "float"],
    ];
    for (const [file, name] of cases) {
        it(`decodes ${file} like FreeImage`, async () => {
            const expected = JSON.parse(new TextDecoder().decode(read(`${name}.expected.json`))) as { width: number; height: number; format: string; data: number[] };
            const bmp = (await Bitmap.createFromBytes(read(file), file, true))!;
            expect(bmp).not.toBeNull();
            expect([bmp.width, bmp.height]).toEqual([expected.width, expected.height]);
            expect(ResourceFormat[bmp.format]).toBe(expected.format);
            const bits = getNumChannelBits(bmp.format, 0);
            const got =
                getFormatType(bmp.format) === FormatType.Float
                    ? Array.from(new Float32Array(bmp.data.slice().buffer))
                    : bits === 16
                      ? Array.from(new Uint16Array(bmp.data.slice().buffer))
                      : Array.from(bmp.data);
            expect(got).toEqual(expected.data.map((v) => (getFormatType(bmp.format) === FormatType.Float ? Math.fround(v) : v)));
        });
    }

    it("flips rows when not top-down", async () => {
        const top = (await Bitmap.createFromBytes(read("rgb8.png"), "rgb8.png", true))!;
        const bottom = (await Bitmap.createFromBytes(read("rgb8.png"), "rgb8.png", false))!;
        const pitch = top.rowPitch;
        expect(Array.from(bottom.data.subarray(0, pitch))).toEqual(Array.from(top.data.subarray((top.height - 1) * pitch)));
    });

    it("converts float images to half under ConvertToFloat16", async () => {
        const f32 = (await Bitmap.createFromBytes(read("float.pfm"), "float.pfm", true))!;
        const f16 = (await Bitmap.createFromBytes(read("float.pfm"), "float.pfm", true, BitmapImportFlags.ConvertToFloat16))!;
        expect(f16.format).toBe(ResourceFormat.RGBA16Float);
        const a = new Float32Array(f32.data.slice().buffer);
        const h = new Uint16Array(f16.data.slice().buffer);
        for (let i = 0; i < a.length; i++) expect(Math.abs(float16ToFloat32(h[i]!) - a[i]!)).toBeLessThanOrEqual(Math.abs(a[i]!) * 1e-3 + 1e-4);
    });

    it("returns null for unknown data", async () => {
        expect(await Bitmap.createFromBytes(new Uint8Array([1, 2, 3, 4]), "x.bin", true)).toBeNull();
    });
});

describe("Bitmap.saveImage", () => {
    const W = 5;
    const H = 3;
    const rgba8 = Uint8Array.from({ length: W * H * 4 }, (_, i) => (i * 37 + 11) & 0xff);
    const floats = Float32Array.from({ length: W * H * 4 }, (_, i) => Math.fround(Math.sin(i) * 7));

    for (const [fmt, ext] of [[BitmapFileFormat.PngFile, "png"], [BitmapFileFormat.TgaFile, "tga"]] as const) {
        for (const flags of [BitmapExportFlags.None, BitmapExportFlags.ExportAlpha, BitmapExportFlags.ExportAlpha | BitmapExportFlags.Uncompressed]) {
            it(`round-trips RGBA8 through ${ext} (flags ${flags})`, async () => {
                const bytes = await Bitmap.saveImage(W, H, fmt, flags, ResourceFormat.RGBA8Unorm, true, rgba8);
                const bmp = (await Bitmap.createFromBytes(bytes, `out.${ext}`, true))!;
                const alpha = (flags & BitmapExportFlags.ExportAlpha) !== 0;
                expect(bmp.format).toBe(alpha ? ResourceFormat.BGRA8Unorm : ResourceFormat.BGRX8Unorm);
                for (let i = 0; i < W * H; i++) {
                    expect([bmp.data[i * 4 + 2], bmp.data[i * 4 + 1], bmp.data[i * 4]]).toEqual([rgba8[i * 4], rgba8[i * 4 + 1], rgba8[i * 4 + 2]]);
                    expect(bmp.data[i * 4 + 3]).toBe(alpha ? rgba8[i * 4 + 3] : 255);
                }
            });
        }
    }

    it("does not modify the caller's RGBA8 data (native swaps it in place)", async () => {
        const copy = rgba8.slice();
        await Bitmap.saveImage(W, H, BitmapFileFormat.PngFile, BitmapExportFlags.None, ResourceFormat.RGBA8Unorm, true, copy);
        expect(copy).toEqual(rgba8);
    });

    it("honours isTopDown for 8-bit formats", async () => {
        const up = await Bitmap.saveImage(W, H, BitmapFileFormat.PngFile, BitmapExportFlags.ExportAlpha, ResourceFormat.RGBA8Unorm, false, rgba8);
        const bmp = (await Bitmap.createFromBytes(up, "up.png", false))!;
        for (let i = 0; i < W * H; i++) expect(bmp.data[i * 4 + 1]).toBe(rgba8[i * 4 + 1]);
    });

    it("round-trips floats through PFM", async () => {
        const bytes = await Bitmap.saveImage(W, H, BitmapFileFormat.PfmFile, BitmapExportFlags.None, ResourceFormat.RGBA32Float, true, floats);
        const bmp = (await Bitmap.createFromBytes(bytes, "out.pfm", true))!;
        const got = new Float32Array(bmp.data.slice().buffer);
        for (let i = 0; i < W * H; i++) {
            for (let c = 0; c < 3; c++) expect(got[i * 4 + c]).toBe(floats[i * 4 + c]);
            expect(got[i * 4 + 3]).toBe(1);
        }
    });

    const exrCases: [BitmapExportFlags, boolean][] = [
        [BitmapExportFlags.Uncompressed, false],
        [BitmapExportFlags.Uncompressed | BitmapExportFlags.ExportAlpha, false],
        [BitmapExportFlags.Uncompressed | BitmapExportFlags.ExrFloat16, true],
        [BitmapExportFlags.None, true],
        [BitmapExportFlags.ExportAlpha, true],
        [BitmapExportFlags.Lossy, true],
    ];
    for (const [flags, half] of exrCases) {
        it(`round-trips floats through EXR (flags ${flags})`, async () => {
            const bytes = await Bitmap.saveImage(W, H, BitmapFileFormat.ExrFile, flags, ResourceFormat.RGBA32Float, true, floats);
            const bmp = (await Bitmap.createFromBytes(bytes, "out.exr", true))!;
            expect(bmp.format).toBe(half ? ResourceFormat.RGBA16Float : ResourceFormat.RGBA32Float);
            const got = half ? Array.from(new Uint16Array(bmp.data.slice().buffer), float16ToFloat32) : Array.from(new Float32Array(bmp.data.slice().buffer));
            const alpha = (flags & BitmapExportFlags.ExportAlpha) !== 0;
            for (let i = 0; i < W * H * 4; i++) {
                const want = i % 4 === 3 && !alpha ? 1 : floats[i]!;
                expect(Math.abs(got[i]! - want)).toBeLessThanOrEqual(half ? Math.abs(want) * 1e-3 + 1e-4 : 0);
            }
        });
    }

    it("converts half and 16-bit integer data like native before PFM/EXR", async () => {
        const u16 = Uint16Array.from({ length: W * H * 2 }, (_, i) => i * 1000);
        // Native widens Uint/Sint only; 16-bit Unorm is rejected.
        await expect(Bitmap.saveImage(W, H, BitmapFileFormat.PfmFile, BitmapExportFlags.None, ResourceFormat.RG16Unorm, true, u16)).rejects.toThrow("Only support");
        const bytes = await Bitmap.saveImage(W, H, BitmapFileFormat.PfmFile, BitmapExportFlags.None, ResourceFormat.RG16Uint, true, u16);
        const bmp = (await Bitmap.createFromBytes(bytes, "out.pfm", true))!;
        const got = new Float32Array(bmp.data.slice().buffer);
        expect(got[4]).toBe(Math.fround(Math.fround(2000) / 65535));
        expect(got[6]).toBe(0); // missing channels are zero, alpha 1
        const halves = Uint16Array.from({ length: W * H * 4 }, (_, i) => float32ToFloat16(i / 8));
        const exr = await Bitmap.saveImage(W, H, BitmapFileFormat.ExrFile, BitmapExportFlags.ExportAlpha, ResourceFormat.RGBA16Float, true, halves);
        const back = (await Bitmap.createFromBytes(exr, "out.exr", true))!;
        expect(Array.from(new Uint16Array(back.data.slice().buffer))).toEqual(Array.from(halves));
    });

    it("rejects the flag combinations native rejects", async () => {
        const save = (fmt: BitmapFileFormat, flags: BitmapExportFlags, format = ResourceFormat.RGBA32Float) => Bitmap.saveImage(W, H, fmt, flags, format, true, floats);
        await expect(save(BitmapFileFormat.DdsFile, BitmapExportFlags.None)).rejects.toThrow("Cannot save DDS");
        await expect(save(BitmapFileFormat.ExrFile, BitmapExportFlags.Lossy | BitmapExportFlags.Uncompressed)).rejects.toThrow("lossy cannot be combined");
        await expect(save(BitmapFileFormat.ExrFile, BitmapExportFlags.ExrFloat16)).rejects.toThrow("EXR float16");
        await expect(save(BitmapFileFormat.PfmFile, BitmapExportFlags.ExportAlpha)).rejects.toThrow("PFM does not support alpha");
        await expect(save(BitmapFileFormat.ExrFile, BitmapExportFlags.ExportAlpha, ResourceFormat.RGB32Float)).rejects.toThrow("doesn't have an alpha-channel");
        await expect(save(BitmapFileFormat.ExrFile, BitmapExportFlags.None, ResourceFormat.RGBA8Unorm)).rejects.toThrow("Only support for 32-bit");
    });
});

describe("Bitmap format helpers", () => {
    it("maps extensions and resource formats like native", () => {
        expect(Bitmap.getFormatFromFileExtension("exr")).toBe(BitmapFileFormat.ExrFile);
        expect(Bitmap.getFormatFromFileExtension("jpg")).toBe(BitmapFileFormat.JpegFile);
        expect(() => Bitmap.getFormatFromFileExtension("jpeg")).toThrow();
        expect(Bitmap.getFileExtFromResourceFormat(ResourceFormat.RGBA32Float)).toBe("exr");
        expect(Bitmap.getFileExtFromResourceFormat(ResourceFormat.RGBA16Uint)).toBe("exr");
        expect(Bitmap.getFileExtFromResourceFormat(ResourceFormat.RGBA8UnormSrgb)).toBe("png");
        expect(Bitmap.getFileDialogFilters().map((f) => f.ext)).toEqual(["exr", "pfm", "hdr", "png", "jpg", "bmp", "tga", "dds", "hdr"]);
    });

    it("describes formats like native's kFormatDesc", () => {
        expect([getFormatChannelCount(ResourceFormat.RGBA16Float), getFormatType(ResourceFormat.RGBA16Float), getNumChannelBits(ResourceFormat.RGBA16Float, 3)]).toEqual([4, FormatType.Float, 16]);
        expect(getFormatType(ResourceFormat.RGBA8UnormSrgb)).toBe(FormatType.UnormSrgb);
        expect(getFormatType(ResourceFormat.RG32Int)).toBe(FormatType.Sint);
        expect([getFormatChannelCount(ResourceFormat.RGB32Float), getNumChannelBits(ResourceFormat.RGB32Float, 3)]).toEqual([3, 0]);
        expect(getNumChannelBits(ResourceFormat.R11G11B10Float, 2)).toBe(10);
        expect(getFormatChannelCount(ResourceFormat.BGRX8Unorm)).toBe(4);
        expect(getFormatType(ResourceFormat.BC6HU16)).toBe(FormatType.Float);
    });
});
