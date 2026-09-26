/**
 * Transplant of FalcorTest Core/EnumTests.cpp (FALCOR_ENUM_INFO conversions).
 */

import { describe, expect, it } from "vitest";
import { defineEnumInfo, enumHasValue, enumToString, flagsToStringList, stringListToFlags, stringToEnum } from "../../src/Core/Enum.js";

enum TestEnum {
    A,
    B,
    C,
}
const kTestEnumInfo = defineEnumInfo([
    [TestEnum.A, "A"],
    [TestEnum.B, "B"],
    [TestEnum.C, "C"],
]);

enum TestFlags {
    A = 1 << 0,
    B = 1 << 1,
    C = 1 << 2,
}
const kTestFlagsInfo = defineEnumInfo([
    [TestFlags.A, "A"],
    [TestFlags.B, "B"],
    [TestFlags.C, "C"],
]);

// TestStruct::TestEnum (nested in a struct natively).
enum TestStructEnum {
    X,
    Y,
}
const kTestStructEnumInfo = defineEnumInfo([
    [TestStructEnum.X, "X"],
    [TestStructEnum.Y, "Y"],
]);

describe("EnumTests", () => {
    it("EnumInfo", () => {
        expect(enumHasValue(kTestEnumInfo, "A")).toBe(true);
        expect(enumHasValue(kTestEnumInfo, "B")).toBe(true);
        expect(enumHasValue(kTestEnumInfo, "C")).toBe(true);
        expect(enumHasValue(kTestEnumInfo, "D")).toBe(false);

        expect(stringToEnum(kTestEnumInfo, "A")).toBe(TestEnum.A);
        expect(stringToEnum(kTestEnumInfo, "B")).toBe(TestEnum.B);
        expect(stringToEnum(kTestEnumInfo, "C")).toBe(TestEnum.C);

        // Converting unregistered values/strings throws.
        expect(() => enumToString(kTestEnumInfo, -1 as TestEnum)).toThrow();
        expect(() => stringToEnum(kTestEnumInfo, "D")).toThrow();

        expect(enumHasValue(kTestStructEnumInfo, "X")).toBe(true);
        expect(enumHasValue(kTestStructEnumInfo, "Y")).toBe(true);
        expect(enumHasValue(kTestStructEnumInfo, "Z")).toBe(false);

        expect(enumToString(kTestStructEnumInfo, TestStructEnum.X)).toBe("X");
        expect(enumToString(kTestStructEnumInfo, TestStructEnum.Y)).toBe("Y");
        expect(stringToEnum(kTestStructEnumInfo, "X")).toBe(TestStructEnum.X);
        expect(stringToEnum(kTestStructEnumInfo, "Y")).toBe(TestStructEnum.Y);

        // Flags.
        expect(flagsToStringList(kTestFlagsInfo, 0 as TestFlags)).toEqual([]);
        expect(flagsToStringList(kTestFlagsInfo, TestFlags.A)).toEqual(["A"]);
        expect(flagsToStringList(kTestFlagsInfo, TestFlags.B)).toEqual(["B"]);
        expect(flagsToStringList(kTestFlagsInfo, TestFlags.A | TestFlags.B)).toEqual(["A", "B"]);
        expect(flagsToStringList(kTestFlagsInfo, TestFlags.A | TestFlags.B | TestFlags.C)).toEqual(["A", "B", "C"]);
        expect(() => flagsToStringList(kTestFlagsInfo, -1 as TestFlags)).toThrow();

        expect(stringListToFlags(kTestFlagsInfo, [])).toBe(0);
        expect(stringListToFlags(kTestFlagsInfo, ["A"])).toBe(TestFlags.A);
        expect(stringListToFlags(kTestFlagsInfo, ["B"])).toBe(TestFlags.B);
        expect(stringListToFlags(kTestFlagsInfo, ["A", "B"])).toBe(TestFlags.A | TestFlags.B);
        expect(stringListToFlags(kTestFlagsInfo, ["A", "B", "C"])).toBe(TestFlags.A | TestFlags.B | TestFlags.C);
        expect(() => stringListToFlags(kTestFlagsInfo, ["D"])).toThrow();
    });
});
