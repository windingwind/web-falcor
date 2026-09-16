import { describe, expect, it } from "vitest";
import { float16ToFloat32, float32ToFloat16 } from "../src/Utils/Math/Float16.js";
import { f32tof16 } from "../src/Scene/Material/MaterialData.js";

describe("float32ToFloat16 (native Float16.cpp: round to nearest, ties up)", () => {
    it("rounds to nearest instead of truncating", () => {
        expect(float32ToFloat16(0.3)).toBe(0x34cd); // 0.300048828 (truncation would give 0x34cc = 0.2998)
        expect(float32ToFloat16(0.8)).toBe(0x3a66);
        expect(float32ToFloat16(0.9)).toBe(0x3b33);
        expect(float32ToFloat16(1.0)).toBe(0x3c00);
        expect(float32ToFloat16(-2.5)).toBe(0xc100);
        expect(float32ToFloat16(1.5)).toBe(0x3e00);
    });
    it("breaks exact ties upward (not to even)", () => {
        // 1 + 2^-11 is exactly halfway between 0x3c00 and 0x3c01.
        expect(float32ToFloat16(1 + 2 ** -11)).toBe(0x3c01);
        // Carry into the exponent: just below 2 rounds up to 2.0.
        expect(float32ToFloat16(1.99951171875 + 2 ** -12)).toBe(0x4000);
    });
    it("handles range limits and specials", () => {
        expect(float32ToFloat16(65504)).toBe(0x7bff);
        expect(float32ToFloat16(65520)).toBe(0x7c00); // rounds past max -> inf
        expect(float32ToFloat16(1e6)).toBe(0x7c00);
        expect(float32ToFloat16(Infinity)).toBe(0x7c00);
        expect(float32ToFloat16(-Infinity)).toBe(0xfc00);
        expect(float32ToFloat16(NaN) & 0x7c00).toBe(0x7c00);
        expect(float32ToFloat16(NaN) & 0x3ff).not.toBe(0);
        expect(float32ToFloat16(2 ** -24)).toBe(0x0001); // smallest subnormal
        expect(float32ToFloat16(2 ** -14)).toBe(0x0400); // smallest normal
        expect(float32ToFloat16(2 ** -26)).toBe(0x0000); // below subnormal range
        expect(float32ToFloat16(-0)).toBe(0x8000);
    });
    it("round-trips every half exactly", () => {
        for (let bits = 0; bits < 0x7c00; bits += 37) {
            expect(float32ToFloat16(float16ToFloat32(bits))).toBe(bits);
            expect(float32ToFloat16(float16ToFloat32(bits | 0x8000))).toBe(bits | 0x8000);
        }
    });
    it("is what the material packer uses", () => {
        expect(f32tof16(0.3)).toBe(0x34cd);
    });
});
