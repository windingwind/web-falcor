/**
 * Transplant of FalcorTest Utils/ColorUtilsTests.cpp (color-space transforms and white balance).
 */

import { describe, expect, it } from "vitest";
import {
    RGBtoXYZ_Rec709,
    XYZtoRGB_Rec709,
    calculateWhiteBalanceTransformRGB_Rec709,
    kLMStoXYZ_Bradford,
    kLMStoXYZ_CAT02,
    kXYZtoLMS_Bradford,
    kXYZtoLMS_CAT02,
    mulMat3,
    mulMat3Vec,
} from "../../src/ToneMapper/ColorUtils.js";

type Vec3 = [number, number, number];

const kMaxError = 1e-5;
const maxAbsDiff = (a: Vec3, b: Vec3) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]));

/** libstdc++ std::default_random_engine (minstd_rand0) + uniform_real_distribution<float>. */
function makeUniform(): () => number {
    let x = 1;
    return () => {
        x = (x * 16807) % 2147483647;
        const r = Math.fround(Math.fround(x - 1) / Math.fround(2147483646));
        return r >= 1 ? Math.fround(1 - 2 ** -24) : r;
    };
}

describe("ColorUtilsTests", () => {
    it("ColorTransforms", () => {
        const n = 10000;
        const u = makeUniform();

        const LMS_CAT02 = mulMat3(kLMStoXYZ_CAT02, kXYZtoLMS_CAT02);
        const LMS_Bradford = mulMat3(kLMStoXYZ_Bradford, kXYZtoLMS_Bradford);

        let maxErr = 0;
        for (let i = 0; i < n; i++) {
            const c: Vec3 = [u(), u(), u()];
            // RGB<->XYZ round trip, then XYZ<->LMS with CAT02 and Bradford.
            maxErr = Math.max(maxErr, maxAbsDiff(XYZtoRGB_Rec709(RGBtoXYZ_Rec709(c)), c));
            maxErr = Math.max(maxErr, maxAbsDiff(mulMat3Vec(LMS_CAT02, c), c));
            maxErr = Math.max(maxErr, maxAbsDiff(mulMat3Vec(LMS_Bradford, c), c));
        }
        expect(maxErr).toBeLessThanOrEqual(kMaxError);
    });

    it("WhiteBalance", () => {
        const white: Vec3 = [1, 1, 1];

        // The white point should be 6500K. Verify that we get pure white back.
        const wbWhite = mulMat3Vec(calculateWhiteBalanceTransformRGB_Rec709(6500), white);
        expect(maxAbsDiff(wbWhite, white)).toBeLessThanOrEqual(kMaxError);

        // Cloudy (7000K) => r > g > b; Sunny (5500K) and Indoor (3000K) => r < g < b.
        const wbCloudy = mulMat3Vec(calculateWhiteBalanceTransformRGB_Rec709(7000), white);
        let wbSunny = mulMat3Vec(calculateWhiteBalanceTransformRGB_Rec709(5500), white);
        let wbIndoor = mulMat3Vec(calculateWhiteBalanceTransformRGB_Rec709(3000), white);

        expect(wbCloudy[0]).toBeGreaterThanOrEqual(wbCloudy[1]);
        expect(wbCloudy[1]).toBeGreaterThanOrEqual(wbCloudy[2]);

        expect(wbSunny[0]).toBeLessThanOrEqual(wbSunny[1]);
        expect(wbSunny[1]).toBeLessThanOrEqual(wbSunny[2]);

        expect(wbIndoor[0]).toBeLessThanOrEqual(wbIndoor[1]);
        expect(wbIndoor[1]).toBeLessThanOrEqual(wbIndoor[2]);

        // Normalize the returned RGB to max 1.0 to be able to compare the scale.
        wbSunny = wbSunny.map((v) => v / wbSunny[2]) as Vec3;
        wbIndoor = wbIndoor.map((v) => v / wbIndoor[2]) as Vec3;

        expect(wbIndoor[0]).toBeLessThanOrEqual(wbSunny[0]);
        expect(wbIndoor[1]).toBeLessThanOrEqual(wbSunny[1]);
    });
});
