/**
 * Transplant of FalcorTest Utils/PropertiesTests.cpp (JSON-backed Properties container).
 * JS numbers stand in for all native scalar types; int64 extremes round to the nearest double.
 */

import { describe, expect, it } from "vitest";
import { Properties, type PropertyValue } from "../../src/Utils/Properties.js";

enum TestEnum {
    A,
    B,
    C,
}

const U32_MAX = 0xffffffff;
const U64_MAX = Number(2n ** 64n - 1n);
const I32_LOWEST = -0x80000000;
const I64_LOWEST = Number(-(2n ** 63n));
const F32_MAX = 3.4028234663852886e38;
const F64_MAX = Number.MAX_VALUE;

function testPropertyType<T extends PropertyValue>(checkValue: T, differentValue: T): void {
    const props = new Properties();
    props.set("value", checkValue);

    expect(props.has("value")).toBe(true);
    expect(props.has("value2")).toBe(false);

    expect(props.get<T>("value", differentValue)).toEqual(checkValue);
    expect(props.get<T>("value2", differentValue)).toEqual(differentValue);

    expect(props.getOpt<T>("value")).toEqual(checkValue);
    expect(props.getOpt<T>("value2")).toBeUndefined();
}

function makeJson(): Record<string, PropertyValue> {
    return {
        b: true,
        u32: U32_MAX,
        u64: U64_MAX,
        i32: I32_LOWEST,
        i64: I64_LOWEST,
        f32: F32_MAX,
        f64: F64_MAX,
        uint3: [1, 2, 3],
        int3: [-1, 2, -3],
        float3: [0.25, 0.5, 0.75],
        str: "string",
        nested: { str: "string" },
    };
}

describe("PropertiesTests", () => {
    it("PropertiesBasicValues", () => {
        testPropertyType(false, true);
        testPropertyType(true, false);

        testPropertyType(0, 1);
        testPropertyType(U32_MAX, 1);
        testPropertyType(U64_MAX, 1);
        testPropertyType(I32_LOWEST, 1);
        testPropertyType(0x7fffffff, 1);
        testPropertyType(I64_LOWEST, 1);
        testPropertyType(Number(2n ** 63n - 1n), 1);
        testPropertyType(-F32_MAX, 1);
        testPropertyType(F32_MAX, 1);
        testPropertyType(-F64_MAX, 1);
        testPropertyType(F64_MAX, 1);

        testPropertyType("", " ");
        testPropertyType("test", "test2");

        {
            const emptyProps = {};
            const testProps = { int: 123, str: "test" };
            testPropertyType<PropertyValue>(emptyProps, testProps);
            testPropertyType<PropertyValue>(testProps, emptyProps);
        }

        testPropertyType([-10, -11], [10, 11]);
        testPropertyType([-10, -11, -12], [10, 11, 12]);
        testPropertyType([-10, -11, -12, -13], [10, 11, 12, 13]);

        testPropertyType([10, 11], [110, 111]);
        testPropertyType([10, 11, 12], [110, 111, 112]);
        testPropertyType([10, 11, 12, 13], [110, 111, 112, 113]);

        testPropertyType([-10, -11], [10, 11]);
        testPropertyType([-10, -11, -12], [10, 11, 12]);
        testPropertyType([-10, -11, -12, -13], [10, 11, 12, 13]);

        testPropertyType(TestEnum.B, TestEnum.C);
    });

    it("PropertiesFromJson", () => {
        const props = new Properties(makeJson());
        expect(props.get("b", false)).toBe(true);
        expect(props.get("u32", 0)).toBe(U32_MAX);
        expect(props.get("u64", 0)).toBe(U64_MAX);
        expect(props.get("i32", 0)).toBe(I32_LOWEST);
        expect(props.get("i64", 0)).toBe(I64_LOWEST);
        expect(props.get("f32", 0)).toBe(F32_MAX);
        expect(props.get("f64", 0)).toBe(F64_MAX);
        expect(props.get("uint3", [])).toEqual([1, 2, 3]);
        expect(props.get("int3", [])).toEqual([-1, 2, -3]);
        expect(props.get("float3", [])).toEqual([0.25, 0.5, 0.75]);
        expect(props.get("str", "")).toBe("string");
        expect(props.get("nested", {})).toEqual({ str: "string" });
    });

    it("PropertiesToJson", () => {
        const props = new Properties();
        props.set("b", true);
        props.set("u32", U32_MAX);
        props.set("u64", U64_MAX);
        props.set("i32", I32_LOWEST);
        props.set("i64", I64_LOWEST);
        props.set("f32", F32_MAX);
        props.set("f64", F64_MAX);
        props.set("uint3", [1, 2, 3]);
        props.set("int3", [-1, 2, -3]);
        props.set("float3", [0.25, 0.5, 0.75]);
        props.set("str", "string");
        props.set("nested", new Properties({ str: "string" }).toJSON());
        expect(props.toJSON()).toEqual(makeJson());
    });

    it("PropertiesIterators", () => {
        const props = new Properties();
        props.set("a", 1);
        props.set("b", 2);
        props.set("c", "3");

        expect([...props.entries()]).toEqual([
            ["a", 1],
            ["b", 2],
            ["c", "3"],
        ]);
    });
});
