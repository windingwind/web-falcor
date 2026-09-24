/**
 * Mirrors Utils/Scripting/ScriptWriter: builds Python lines for the scripts objects write
 * about themselves (Clock/Camera/Scene getScript, Mogwai's saved configurations).
 * Arguments print as Python literals (native getArgString: pybind11 repr).
 */

/** A value as a Python literal (bools, numbers, strings, lists, dicts, floatN vectors). */
export function pyRepr(v: unknown): string {
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "number") return Number.isFinite(v) ? String(v) : `float("${v > 0 ? "inf" : v < 0 ? "-inf" : "nan"}")`;
    if (typeof v === "string") return JSON.stringify(v);
    if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
    if (v && typeof v === "object") {
        const o = v as Record<string, unknown>;
        const comps = ["x", "y", "z", "w"].filter((c) => typeof o[c] === "number");
        if (comps.length >= 2 && comps.every((c) => Object.prototype.hasOwnProperty.call(o, c))) {
            return `float${comps.length}(${comps.map((c) => o[c]).join(", ")})`;
        }
        return `{${Object.entries(o)
            .map(([k, val]) => `${JSON.stringify(k)}: ${pyRepr(val)}`)
            .join(", ")}}`;
    }
    return "None";
}

export const ScriptWriter = {
    getArgString: pyRepr,
    /** `func(args...)` */
    makeFunc(func: string, ...args: unknown[]): string {
        return `${func}(${args.map(pyRepr).join(", ")})\n`;
    },
    /** `var.func(args...)` */
    makeMemberFunc(variable: string, func: string, ...args: unknown[]): string {
        return `${variable}.${ScriptWriter.makeFunc(func, ...args)}`;
    },
    /** `var.property` */
    makeGetProperty(variable: string, property: string): string {
        return `${variable}.${property}\n`;
    },
    /** `var.property = arg` */
    makeSetProperty(variable: string, property: string, arg: unknown): string {
        return `${variable}.${property} = ${pyRepr(arg)}\n`;
    },
    /** A path with forward slashes (native getPathString). */
    getPathString(path: string): string {
        return path.replace(/\\\\/g, "/");
    },
};
