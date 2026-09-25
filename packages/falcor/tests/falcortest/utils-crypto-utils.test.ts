/** Transplant of FalcorTest Utils/CryptoUtilsTests.cpp (SHA1). */

import { describe, expect, it } from "vitest";
import { SHA1 } from "../../src/Utils/CryptoUtils.js";

const hex = (md: number[]) => md.map((b) => b.toString(16).padStart(2, "0")).join("");

describe("CryptoUtilsTests", () => {
    it("SHA1", () => {
        const md = [0x2e, 0xf7, 0xbd, 0xe6, 0x08, 0xce, 0x54, 0x04, 0xe9, 0x7d, 0x5f, 0x04, 0x2f, 0x95, 0xf8, 0x9f, 0x1c, 0x23, 0x28, 0x71];
        const sha1 = new SHA1();
        sha1.update("Hello ");
        sha1.update("World!");
        expect(SHA1.toString(sha1.finalize())).toBe(hex(md));
        expect(SHA1.toString(SHA1.compute("Hello World!"))).toBe(hex(md));
        const lorem =
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " +
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure " +
            "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non " +
            "proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
        const md2 = [0xcd, 0x36, 0xb3, 0x70, 0x75, 0x8a, 0x25, 0x9b, 0x34, 0x84, 0x50, 0x84, 0xa6, 0xcc, 0x38, 0x47, 0x3c, 0xb9, 0x5e, 0x27];
        expect(SHA1.toString(SHA1.compute(lorem))).toBe(hex(md2));
        // Empty message and a 64-byte boundary (padding spills into a second block).
        expect(SHA1.toString(SHA1.compute(""))).toBe("da39a3ee5e6b4b0d3255bfef95601890afd80709");
        expect(SHA1.toString(SHA1.compute("a".repeat(56)))).toBe("c2db330f6083854c99d4b5bfb6e8f29f201be699");
    });
});
