/**
 * Transplant of FalcorTest Utils/Color/SpectrumTests.cpp (CIE matching-function integrals).
 */

import { describe, expect, it } from "vitest";
import { Spectra, kCIE_Y_Integral } from "../../src/Utils/Color/Spectrum.js";

describe("SpectrumTests", () => {
    it("SpectrumXYZ", () => {
        // Make sure the integral of the CIE matching functions is 1.
        let x = 0;
        let y = 0;
        let z = 0;
        for (let lambda = 360; lambda <= 830; lambda += 1) {
            x += Spectra.kCIE_X.eval(lambda);
            y += Spectra.kCIE_Y.eval(lambda);
            z += Spectra.kCIE_Z.eval(lambda);
        }
        x /= kCIE_Y_Integral;
        y /= kCIE_Y_Integral;
        z /= kCIE_Y_Integral;
        expect(Math.abs(1 - x)).toBeLessThan(0.005);
        expect(Math.abs(1 - y)).toBeLessThan(0.005);
        expect(Math.abs(1 - z)).toBeLessThan(0.005);
    });
});
