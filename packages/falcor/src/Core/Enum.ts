/**
 * Mirrors Core/Enum.h: FALCOR_ENUM_INFO's value/name table and the conversions built on it
 * (enumToString, stringToEnum, enumHasValue, flagsToStringList, stringListToFlags).
 */

import { RuntimeError } from "./Error.js";

export type EnumInfo<T extends number> = readonly (readonly [T, string])[];

/** Mirrors FALCOR_ENUM_INFO: the registered value/name pairs of an enum. */
export function defineEnumInfo<T extends number>(items: readonly (readonly [T, string])[]): EnumInfo<T> {
    return items;
}

/** Throws if the value is not registered. */
export function enumToString<T extends number>(info: EnumInfo<T>, value: T): string {
    const item = info.find(([v]) => v === value);
    if (!item) throw new RuntimeError(`Invalid enum value ${value}`);
    return item[1];
}

/** Throws if the name is not registered. */
export function stringToEnum<T extends number>(info: EnumInfo<T>, name: string): T {
    const item = info.find(([, n]) => n === name);
    if (!item) throw new RuntimeError(`Invalid enum name '${name}'`);
    return item[0];
}

export function enumHasValue<T extends number>(info: EnumInfo<T>, name: string): boolean {
    return info.some(([, n]) => n === name);
}

/** Names of the set flags; throws if any set bit is not registered. */
export function flagsToStringList<T extends number>(info: EnumInfo<T>, flags: T): string[] {
    const list: string[] = [];
    let rest = flags as number;
    for (const [v, n] of info) {
        // Native is_set/flip_bit: any overlapping bit counts, then the flag's bits are cleared.
        if ((rest & v) !== 0) {
            list.push(n);
            rest &= ~v;
        }
    }
    if (rest !== 0) throw new RuntimeError(`Invalid enum flags value ${flags}`);
    return list;
}

export function stringListToFlags<T extends number>(info: EnumInfo<T>, list: readonly string[]): T {
    return list.reduce<number>((flags, name) => flags | stringToEnum(info, name), 0) as T;
}
