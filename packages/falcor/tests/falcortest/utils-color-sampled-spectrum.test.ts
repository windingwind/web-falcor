/**
 * Transplant of FalcorTest Utils/Color/SampledSpectrumTests.cpp, plus checks of SpectrumUtils.
 */

import { describe, expect, it } from "vitest";
import { SampledSpectrum } from "../../src/Utils/Color/SampledSpectrum.js";
import { SpectrumUtils } from "../../src/Utils/Color/SpectrumUtils.js";

describe("SampledSpectrumTests", () => {
    it("SampledSpectrum", () => {
        const s = new SampledSpectrum<number>(400, 800, 5); // 400, 500, 600, 700, 800nm
        s.set([0, 0.25, 1, 0.75, 0]);

        const testData: [number, number][] = [
            [200, 0],
            [450, 0.125],
            [600, 1],
            [725, 0.5625],
            [800, 0],
            [810, 0],
        ];
        for (const [lambda, expected] of testData) expect(s.eval(lambda)).toBe(expected);
    });
});

describe("SpectrumUtils", () => {
    it("tables match native's layout", () => {
        expect(SpectrumUtils.sCIE_XYZ_1931_1nm.size()).toBe(471);
        expect(SpectrumUtils.sD65_5nm.size()).toBe(107);
        const xyz = SpectrumUtils.wavelengthToXYZ_CIE1931(360); // xyz1931_1nm[0]
        expect([xyz.x, xyz.y, xyz.z]).toEqual([Math.fround(0.00013), Math.fround(0.000004), Math.fround(0.000606)]);
        expect(SpectrumUtils.wavelengthToD65(300)).toBe(Math.fround(0.0341));
        expect(SpectrumUtils.wavelengthToD65(302.5)).toBeCloseTo((0.0341 + 1.6643) / 2, 5);
    });

    // Reference values from native's formulas compiled as C++ float (g++ -ffp-contract=off).
    it("eval and toXYZ match a native float build", () => {
        expect(SpectrumUtils.wavelengthToXYZ_CIE1931(555.3).y).toBe(Math.fround(0.999957085));
        const xyz = SpectrumUtils.toXYZ(SpectrumUtils.sD65_5nm);
        expect([xyz.x, xyz.y, xyz.z]).toEqual([Math.fround(10043.8643), Math.fround(10567.2959), Math.fround(11507.4619)]);
        // Native's hard-coded Y_D65 = 10567.0762f differs from this by ~2e-5 (relative).
        expect(Math.abs(xyz.y / 10567.0762 - 1)).toBeLessThan(2.5e-5);
    });
});
