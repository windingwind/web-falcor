/**
 * Transplanted FalcorTest: Utils/Image/BitmapTests (Bitmap_LinearRamp_PNG). §9: there is
 * no file system, so the saved bytes are loaded back directly instead of from disk.
 */
import { describe, expect, it } from "vitest";
import { Bitmap, BitmapExportFlags, BitmapFileFormat } from "../../src/Utils/Image/Bitmap.js";
import { ResourceFormat } from "../../src/Core/API/Formats.js";

describe("BitmapTests", () => {
    it("Bitmap_LinearRamp_PNG", async () => {
        // Test saving a linear ramp as an 8-bit grayscale PNG.
        const data = Uint8Array.from({ length: 256 }, (_, i) => i);
        const png = await Bitmap.saveImage(256, 1, BitmapFileFormat.PngFile, BitmapExportFlags.None, ResourceFormat.R8Uint, true, data);

        // Saving 8-bit grayscale data as PNG results in it loading as BGRX in 8-bit unorm format.
        const bmp = await Bitmap.createFromBytes(png, "test_linear_ramp.png", true);
        expect(bmp).not.toBeNull();
        expect(bmp!.width).toBe(256);
        expect(bmp!.height).toBe(1);
        expect(bmp!.format).toBe(ResourceFormat.BGRX8Unorm);
        expect(bmp!.getSize()).toBe(1024);
        for (let i = 0; i < 256; i++) {
            expect(bmp!.data[4 * i + 0]).toBe(i); // B
            expect(bmp!.data[4 * i + 1]).toBe(i); // G
            expect(bmp!.data[4 * i + 2]).toBe(i); // R
        }
    });
});
