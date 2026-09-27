/**
 * Transplant of FalcorTest Utils/HashUtilsTests.cpp's CPU test. Like native (RUN_PERFECT_HASH_TESTS), it is off by
 * default: 2^32 hashes take about 80 s; set RUN_PERFECT_HASH_TESTS=1 to run it.
 */

import { describe, expect, it } from "vitest";

/** Jenkins hash, as the test's own copy natively (matches HashUtils.slang). */
function jenkinsHash(a: number): number {
    a = (a + 0x7ed55d16 + (a << 12)) >>> 0;
    a = (a ^ 0xc761c23c ^ (a >>> 19)) >>> 0;
    a = (a + 0x165667b1 + (a << 5)) >>> 0;
    a = ((a + 0xd3a2646c) ^ (a << 9)) >>> 0;
    a = (a + 0xfd7046c5 + (a << 3)) >>> 0;
    a = (a ^ 0xb55a4f09 ^ (a >>> 16)) >>> 0;
    return a;
}

describe("HashUtilsTests", () => {
    it.skipIf(!process.env.RUN_PERFECT_HASH_TESTS)("JenkinsHash_PerfectHashCPU", () => {
        const result = new Uint32Array(1 << 27);
        for (let i = 0; i < 2 ** 32; i++) {
            const h = jenkinsHash(i >>> 0);
            result[h >>> 5]! |= 1 << (h & 0x1f);
        }
        let missing = -1;
        for (let i = 0; i < result.length && missing < 0; i++) if (result[i] !== 0xffffffff) missing = i;
        expect(missing).toBe(-1);
    }, 600000);
});
