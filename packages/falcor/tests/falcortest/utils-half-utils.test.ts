/**
 * Transplant of FalcorTest Utils/HalfUtilsTests.cpp (CPU conservative-rounding test only).
 */

import { describe, expect, it } from "vitest";
import { float16ToFloat32 as f16tof32, float32ToFloat16 as f32tof16 } from "../../src/Utils/Math/Float16.js";

const FLT_EPSILON = 2 ** -23;
const fround = Math.fround;

/** Converts a finite fp32 number to fp16, rounding down to the nearest representable number. */
function f32tof16_roundDown(value: number): number {
    let h = f32tof16(value);
    const res = f16tof32(h);
    if (res > value) {
        if (res < 0) h++;
        else if (res > 0) h--;
        else h = 0x8001;
    }
    return h;
}

/** Converts a finite fp32 number to fp16, rounding up to the nearest representable number. */
function f32tof16_roundUp(value: number): number {
    let h = f32tof16(value);
    const res = f16tof32(h);
    if (res < value) {
        if (res < 0) h--;
        else if (res > 0) h++;
        else h = 0x0001;
    }
    return h;
}

function isExactFP16(v: number): boolean {
    return f16tof32(f32tof16(v)) === v;
}

/** All finite fp16 values plus fp32 neighbours just above/below each. */
function generateFP16TestData(): number[] {
    const data: number[] = [];
    for (let i = 0; i < 0xfc00; i++) {
        if (i >= 0x7c00 && i < 0x8000) continue; // Skip special values (inf, nan).
        const exact = f16tof32(i);
        const x = fround(exact * fround(1 + FLT_EPSILON));
        if (x !== 0) expect(x).not.toBe(exact);
        const y = fround(exact * fround(1 - FLT_EPSILON));
        if (x !== 0) expect(y).not.toBe(exact);
        data.push(exact, x, y);
    }
    return data;
}

describe("HalfUtilsTests", () => {
    it("FP32ToFP16ConservativeRoundingCPU", () => {
        // Test assumptions on fp16 encoding.
        expect(f16tof32(0x0000)).toBe(0);
        expect(f16tof32(0x8000)).toBe(-0);
        expect(f16tof32(0x7c00)).toBe(Infinity);
        expect(f16tof32(0xfc00)).toBe(-Infinity);

        // Test f32->f16 rounding functions on the CPU.
        const testData = generateFP16TestData();
        let failures = 0;
        for (const v of testData) {
            const up = f16tof32(f32tof16_roundUp(v));
            const down = f16tof32(f32tof16_roundDown(v));
            if (isExactFP16(v)) {
                if (up !== v || down !== v) failures++;
            } else if (!(up >= v) || !(down <= v)) {
                failures++;
            }
        }
        expect(failures).toBe(0);
    });
});
