/** Core/Platform/OS bit helpers against their native definitions (0 for no set bits, as on Windows). */

import { describe, expect, it } from "vitest";
import { bitScanForward, bitScanReverse, popcount } from "../src/Core/Platform/OS.js";

describe("OS bit helpers", () => {
    it("match __builtin_clz/ctz/popcount", () => {
        expect([0, 1, 2, 3, 0x80, 0x80000000, 0xffffffff].map(bitScanReverse)).toEqual([0, 0, 1, 1, 7, 31, 31]);
        expect([0, 1, 2, 12, 0x80000000, 0xffffffff].map(bitScanForward)).toEqual([0, 0, 1, 2, 31, 0]);
        expect([0, 1, 3, 0xff, 0x80000001, 0xffffffff].map(popcount)).toEqual([0, 1, 2, 8, 2, 32]);
    });
});
