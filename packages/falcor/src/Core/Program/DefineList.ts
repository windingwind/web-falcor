/**
 * Shader define list mirroring Falcor/Core/Program/DefineList.h.
 */

export class DefineList extends Map<string, string> {
    add(name: string, value: string | number | boolean = ""): this {
        this.set(name, typeof value === "boolean" ? (value ? "1" : "0") : String(value));
        return this;
    }

    remove(name: string): this {
        this.delete(name);
        return this;
    }

    addAll(other: DefineList | Record<string, string | number | boolean>): this {
        if (other instanceof DefineList) {
            for (const [k, v] of other) this.set(k, v);
        } else {
            for (const [k, v] of Object.entries(other)) this.add(k, v);
        }
        return this;
    }

    /** Stable serialization used as cache key (mirrors ProgramVersion define hashing). */
    key(): string {
        return [...this.entries()]
            .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
            .map(([k, v]) => `${k}=${v}`)
            .join(";");
    }

    /** Preprocessor header injected into each translation unit (see SlangCompiler). */
    toHeader(): string {
        let header = "";
        for (const [k, v] of this) header += v === "" ? `#define ${k}\n` : `#define ${k} ${v}\n`;
        return header;
    }

    clone(): DefineList {
        return new DefineList(this);
    }
}
