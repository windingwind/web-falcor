/**
 * Spectra (Utils/Color/Spectrum): named spectra from native's tables and the
 * spectrum -> RGB path the pbrt importer uses for named/sampled/blackbody values.
 */

import { describe, expect, it } from "vitest";
import { BlackbodySpectrum, PiecewiseLinearSpectrum, Spectra, kCIE_Y_Integral, spectrumToRGB, spectrumToXYZ } from "../src/Utils/Color/Spectrum.js";

describe("Spectrum", () => {
    it("normalizes illuminants to unit luminance", () => {
        // fromInterleaved(normalize=true) scales to Y = 1, so D65 is white in Rec.709.
        const d65 = Spectra.getNamedSpectrum("stdillum-D65")!;
        const xyz = spectrumToXYZ(d65);
        expect(xyz.y).toBeCloseTo(1, 5);
        const rgb = spectrumToRGB(d65);
        for (const c of [rgb.x, rgb.y, rgb.z]) expect(Math.abs(c - 1)).toBeLessThan(0.01);
    });

    it("integrates a flat spectrum to the CIE integrals", () => {
        const flat = new PiecewiseLinearSpectrum([360, 830], [1, 1]);
        expect(spectrumToXYZ(flat).y * kCIE_Y_Integral).toBeCloseTo(kCIE_Y_Integral, 2);
    });

    it("interpolates piecewise-linearly and is zero outside its range", () => {
        const s = new PiecewiseLinearSpectrum([400, 500, 600], [0, 1, 0.5]);
        expect(s.eval(450)).toBeCloseTo(0.5, 6);
        expect(s.eval(550)).toBeCloseTo(0.75, 6);
        expect(s.eval(399)).toBe(0);
        expect(s.eval(601)).toBe(0);
        expect(s.getMaxValue()).toBe(1);
    });

    it("peaks a normalized blackbody at Wien's wavelength", () => {
        const bb = new BlackbodySpectrum(6500);
        const peak = (2.8977721e-3 / 6500) * 1e9;
        expect(bb.eval(peak)).toBeCloseTo(1, 6);
        expect(bb.eval(peak - 50)).toBeLessThan(1);
        expect(bb.eval(peak + 50)).toBeLessThan(1);
    });

    it("gives copper's measured eta and k in RGB", () => {
        const eta = spectrumToRGB(Spectra.getNamedSpectrum("metal-Cu-eta")!);
        const k = spectrumToRGB(Spectra.getNamedSpectrum("metal-Cu-k")!);
        // Copper: low eta in red, rising towards blue; high k throughout.
        expect(eta.x).toBeLessThan(eta.y);
        expect(eta.y).toBeLessThan(eta.z + 0.2);
        expect(k.x).toBeGreaterThan(k.z);
        console.log(`metal-Cu eta ${[eta.x, eta.y, eta.z].map((v) => v.toFixed(4)).join()} k ${[k.x, k.y, k.z].map((v) => v.toFixed(4)).join()}`);
    });

    it("returns undefined for unknown names", () => {
        expect(Spectra.getNamedSpectrum("metal-Unobtainium-eta")).toBeUndefined();
        expect(Spectra.getNamedSpectrum("glass-BK7")).toBeDefined();
    });
});
