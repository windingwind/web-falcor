/**
 * ImageCompare (port of Falcor/Source/Tools/ImageCompare) against values printed by
 * the native tool on the Bitmap fixtures (6 significant digits, as std::cout prints).
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { compareImages, encodeCompareImage, generateHeatMap, loadCompareImage } from "../src/Utils/Image/ImageCompare.js";

const dir = new URL("./fixtures/bitmap/", import.meta.url);
const load = (name: string) => loadCompareImage(new Uint8Array(readFileSync(new URL(name, dir))), name);

// [imageA, imageB, metric, alpha, native output]
const kNative: [string, string, string, boolean, number][] = [
    ["rgb8.png", "rgba8.png", "mse", false, 0],
    ["rgb8.png", "rgba8.png", "mse", true, 0.0886016],
    ["rgb8.png", "rgba8.png", "rmse", true, 0.0885131],
    ["rgb8.png", "rgba8.png", "mape", true, 12.8863],
    ["gray8.png", "rgb16.png", "mse", false, 0.191868],
    ["gray8.png", "rgb16.png", "rmse", false, 31.2825],
    ["gray8.png", "rgb16.png", "mae", true, 0.143901],
    ["gray8.png", "rgb16.png", "mape", false, 2050.66],
    ["gray16.png", "palette.png", "mse", true, 0.265868],
    ["gray16.png", "palette.png", "rmse", true, 17.8921],
    ["gray16.png", "palette.png", "mape", true, 416.512],
    ["rgba8.png", "gray-alpha8.png", "mse", false, 0.182153],
    ["rgba8.png", "gray-alpha8.png", "rmse", true, 9.24073],
    ["rgba8.png", "gray-alpha8.png", "mape", false, 187.089],
    ["float.pfm", "rgb8.png", "mse", false, 0.308863],
    ["float.pfm", "rgb8.png", "rmse", true, 96.9266],
    ["float.pfm", "rgb8.png", "mape", false, 18781.7],
];

describe("ImageCompare", () => {
    for (const [a, b, metric, alpha, native] of kNative) {
        it(`${metric}${alpha ? " -a" : ""} ${a} ${b}`, async () => {
            const { error } = compareImages(await load(a), await load(b), metric, alpha);
            expect(Number(error.toPrecision(6))).toBe(native);
        });
    }

    it("rejects different resolutions and unknown metrics", async () => {
        const img = await load("rgb8.png");
        expect(() => compareImages(img, { width: 1, height: 1, data: new Float32Array(4) })).toThrow(/different resolutions/);
        expect(() => compareImages(img, img, "foo")).toThrow(/Unknown error metric/);
    });

    it("heat map spans blue to red and round-trips through PFM as FreeImage writes it", async () => {
        const heat = generateHeatMap(2, 1, Float32Array.of(0, 1));
        expect(Array.from(heat.data)).toEqual([0, 0, 1, 1, 1, 0, 0, 1]);
        const tall = generateHeatMap(1, 2, Float32Array.of(0, 1));
        const pfm = await encodeCompareImage(tall, "heat.pfm");
        const back = await loadCompareImage(pfm, "heat.pfm");
        expect(Array.from(back.data)).toEqual(Array.from(tall.data));
        // FreeImage stores the top row (blue) first.
        expect(Array.from(new Float32Array(pfm.buffer.slice(pfm.length - 24, pfm.length - 12)))).toEqual([0, 0, 1]);
    });
});
