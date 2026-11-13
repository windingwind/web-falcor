/**
 * Regression: SimplePostFX blitted src -> dst for a vignette-only configuration because the pass-through test ignored
 * vignetteAmount: a vignette-only configuration was an exact no-op.
 * CPU-only (no GPU device needed).
 */
import { describe, expect, it } from "vitest";
import { simplePostFXIsDefault, type SimplePostFXEffectParams } from "../../src/Utils/SimplePostFXDefault.js";

const neutral: SimplePostFXEffectParams = {
    bloomAmount: 0,
    vignetteAmount: 0,
    chromaticAberrationAmount: 0,
    barrelDistortAmount: 0,
    saturationCurve: [1, 1, 1],
    colorOffset: [0.5, 0.5, 0.5],
    colorScale: [0.5, 0.5, 0.5],
    colorPower: [0.5, 0.5, 0.5],
    colorOffsetScalar: 0,
    colorScaleScalar: 0,
    colorPowerScalar: 0,
};

describe("SimplePostFX pass-through test", () => {
    it("neutral settings blit", () => {
        expect(simplePostFXIsDefault(neutral)).toBe(true);
    });
    it("vignette alone is not a pass-through", () => {
        expect(simplePostFXIsDefault({ ...neutral, vignetteAmount: 0.35 })).toBe(false);
    });
    it("each other effect alone is not a pass-through", () => {
        for (const k of ["bloomAmount", "chromaticAberrationAmount", "barrelDistortAmount", "colorOffsetScalar"] as const) {
            expect(simplePostFXIsDefault({ ...neutral, [k]: 0.2 })).toBe(false);
        }
        expect(simplePostFXIsDefault({ ...neutral, saturationCurve: [1, 0.8, 1] })).toBe(false);
    });
});
