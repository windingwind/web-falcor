/**
 * Transplant of FalcorTest Utils/StringUtilsTests.cpp and the path cases of Platform/OSTests.cpp
 * (HasExtension, GetExtensionFromPath).
 */

import { describe, expect, it } from "vitest";
import {
    decodeBase64,
    decodeURI,
    encodeBase64,
    formatByteSize,
    getExtensionFromPath,
    hasExtension,
    removeLeadingTrailingWhitespace,
    removeLeadingWhitespace,
    removeTrailingWhitespace,
    replaceCharacters,
} from "../../src/Utils/StringUtils.js";

describe("StringUtilsTests", () => {
    it("Base64", () => {
        const test = (decoded: string, encoded: string) => {
            const bytes = new TextEncoder().encode(decoded);
            expect(encodeBase64(bytes)).toBe(encoded);
            expect(Array.from(decodeBase64(encoded))).toEqual(Array.from(bytes));
        };
        test("", "");
        test("a", "YQ==");
        test("ab", "YWI=");
        test("abc", "YWJj");
        test("Hello World!", "SGVsbG8gV29ybGQh");
        test(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQsIGNvbnNlY3RldHVyIGFkaXBpc2NpbmcgZWxpdCwgc2VkIGRvIGVpdXNtb2QgdGVtcG9yIGluY2lkaWR1bnQgdXQgbGFib3JlIGV0IGRvbG9yZSBtYWduYSBhbGlxdWEu",
        );
    });

    it("RemoveWhitespace", () => {
        const ws = " \t\n\r";
        expect(removeLeadingWhitespace("  \t\t\n\n\r\rtest", ws)).toBe("test");
        expect(removeLeadingWhitespace("test", ws)).toBe("test");
        expect(removeLeadingWhitespace("test  \t\t\n\n\r\r", ws)).toBe("test  \t\t\n\n\r\r");
        expect(removeTrailingWhitespace("  \t\t\n\n\r\rtest", ws)).toBe("  \t\t\n\n\r\rtest");
        expect(removeTrailingWhitespace("test", ws)).toBe("test");
        expect(removeTrailingWhitespace("test  \t\t\n\n\r\r", ws)).toBe("test");
        expect(removeLeadingTrailingWhitespace("  \t\t\n\n\r\rtest", ws)).toBe("test");
        expect(removeLeadingTrailingWhitespace("test", ws)).toBe("test");
        expect(removeLeadingTrailingWhitespace("test  \t\t\n\n\r\r", ws)).toBe("test");
    });

    it("ReplaceCharacters", () => {
        expect(replaceCharacters("test", "", " ")).toBe("test");
        expect(replaceCharacters("test", "x", " ")).toBe("test");
        expect(replaceCharacters("test", "t", " ")).toBe(" es ");
        expect(replaceCharacters("test", "te", " ")).toBe("  s ");
        expect(replaceCharacters("test", "tes", " ")).toBe("    ");
        expect(replaceCharacters("1122334455", "24", "_")).toBe("11__33__55");
        expect(replaceCharacters("some/path with/whitespace", " /", "_")).toBe("some_path_with_whitespace");
    });

    it("FormatByteSize", () => {
        const [kB, MB, GB, TB] = [1024, 1024 ** 2, 1024 ** 3, 1024 ** 4];
        expect(formatByteSize(0)).toBe("0 B");
        expect(formatByteSize(100)).toBe("100 B");
        expect(formatByteSize(1023)).toBe("1023 B");
        expect(formatByteSize(kB)).toBe("1.00 kB");
        expect(formatByteSize(100 * kB)).toBe("100.00 kB");
        expect(formatByteSize(1023 * kB)).toBe("1023.00 kB");
        expect(formatByteSize(MB)).toBe("1.00 MB");
        expect(formatByteSize(10 * MB)).toBe("10.00 MB");
        expect(formatByteSize(1023 * MB)).toBe("1023.00 MB");
        expect(formatByteSize(GB)).toBe("1.00 GB");
        expect(formatByteSize(10 * GB)).toBe("10.00 GB");
        expect(formatByteSize(1023 * GB)).toBe("1023.00 GB");
        expect(formatByteSize(TB)).toBe("1.00 TB");
        expect(formatByteSize(10 * TB)).toBe("10.00 TB");
    });

    it("DecodeURI", () => {
        expect(decodeURI("test")).toBe("test");
        expect(decodeURI("hello%20world")).toBe("hello world");
        expect(decodeURI("hello%20world%21")).toBe("hello world!");
        expect(decodeURI("%22hello+world%22")).toBe('"hello world"');
    });
});

describe("OSTests", () => {
    it("HasExtension", () => {
        expect(hasExtension("foo.exr", "exr")).toBe(true);
        expect(hasExtension("foo.exr", ".exr")).toBe(true);
        expect(hasExtension("foo.Exr", "exr")).toBe(true);
        expect(hasExtension("foo.Exr", ".exr")).toBe(true);
        expect(hasExtension("foo.Exr", "exR")).toBe(true);
        expect(hasExtension("foo.Exr", ".exR")).toBe(true);
        expect(hasExtension("foo.EXR", "exr")).toBe(true);
        expect(hasExtension("foo.EXR", ".exr")).toBe(true);
        expect(hasExtension("foo.xr", "exr")).toBe(false);
        expect(hasExtension("/foo/png", "")).toBe(true);
        expect(hasExtension("/foo/png", "exr")).toBe(false);
        expect(hasExtension("/foo/.profile", "")).toBe(true);
    });

    it("GetExtensionFromPath", () => {
        expect(getExtensionFromPath("foo.exr")).toBe("exr");
        expect(getExtensionFromPath("foo.Exr")).toBe("exr");
        expect(getExtensionFromPath("foo.EXR")).toBe("exr");
        expect(getExtensionFromPath("foo")).toBe("");
        expect(getExtensionFromPath("/foo/.profile")).toBe("");
    });
});
