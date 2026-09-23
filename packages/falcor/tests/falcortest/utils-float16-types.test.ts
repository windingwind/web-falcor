/**
 * Transplant of FalcorTest Utils/Float16TypesTests.cpp (scalar float16_t <-> float over all bit patterns).
 */

import { describe, expect, it } from "vitest";
import { float16ToFloat32, float32ToFloat16 } from "../../src/Utils/Math/Float16.js";

describe("Float16TypesTests", () => {
    it("Float16Scalar", () => {
        // Test cast to float for all bit patterns.
        for (let bits = 0; bits < 0x10000; bits++) {
            const f = float16ToFloat32(bits);
            const s = bits >> 15;
            const e = (bits >> 10) & 0x1f;
            const m = bits & 0x3ff;
            const sign = s === 0 ? 1 : -1;

            if (e === 0) {
                if (m === 0) {
                    // +-zero
                    expect(f).toBe(s === 0 ? 0 : -0);
                } else {
                    expect(f).toBe(sign * 2 ** -14 * (m / 1024));
                }
            } else if (e === 0x1f) {
                if (m === 0) expect(f).toBe(sign * Infinity);
                else expect(Number.isNaN(f)).toBe(true);
            } else {
                expect(f).toBe(sign * (2 ** e * 2 ** -15) * (1 + m / 1024));
            }
        }

        // Test cast to/from float for all bit patterns.
        for (let bits = 0; bits < 0x10000; bits++) {
            const result = float32ToFloat16(float16ToFloat32(bits));
            const e = (bits >> 10) & 0x1f;
            if (e === 0x1f && (bits & 0x3ff) !== 0) {
                // JS numbers don't carry NaN sign/payload through, so only require a NaN back.
                expect((result & 0x7c00) === 0x7c00 && (result & 0x3ff) !== 0).toBe(true);
            } else {
                expect(result).toBe(bits);
            }
        }
    });
});
