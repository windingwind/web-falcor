/**
 * SVO location-code primitives (BigInt Morton codes) pinned against the
 * gcc reference (tests/oracle/svo-loc-ref.cpp -> assets/svo-loc-ref.json):
 * encodeLocation and createChildLocationCode must byte-match the u64 codes
 * the native build produces (the unmodified runtime decodes them).
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { encodeLocation, createChildLocationCode } from "../src/Scene/SDFs/SDFSVO.js";

const ref = JSON.parse(readFileSync(resolve(__dirname, "../../../tests/oracle/assets/svo-loc-ref.json"), "utf8")) as {
    enc: string[];
    child: string[];
};

describe("SDFSVO location codes (bit-exact vs gcc u64)", () => {
    it("encodeLocation matches at all probes", () => {
        const probes: [number, number, number, number][] = [
            [0, 0, 0, 0],
            [1, 2, 3, 7],
            [5, 5, 5, 3],
            [127, 0, 63, 7],
            [1, 0, 0, 1],
        ];
        probes.forEach(([x, y, z, l], i) => {
            expect(encodeLocation(x, y, z, l).toString()).toBe(ref.enc[i]);
        });
    });

    it("createChildLocationCode matches for all 8 children of the root", () => {
        const root = encodeLocation(0, 0, 0, 0);
        for (let c = 0; c < 8; c++) {
            expect(createChildLocationCode(root, c).toString()).toBe(ref.child[c]);
        }
    });
});
