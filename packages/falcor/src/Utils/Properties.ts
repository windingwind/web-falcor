/**
 * Property container mirroring Falcor/Utils/Properties.h (JSON-backed pass
 * configuration; the Python dicts in graph scripts land here).
 */

/** Python ints beyond 2^53 (u64/i64 extremes) are kept exact as bigint. */
export type PropertyValue = boolean | number | bigint | string | number[] | PropertyValue[] | { [key: string]: PropertyValue };

export class Properties {
    private values = new Map<string, PropertyValue>();

    constructor(init?: Record<string, PropertyValue>) {
        if (init) for (const [k, v] of Object.entries(init)) this.values.set(k, v);
    }

    has(name: string): boolean {
        return this.values.has(name);
    }

    get<T extends PropertyValue>(name: string, defaultValue: T): T {
        return (this.values.get(name) as T) ?? defaultValue;
    }

    getOpt<T extends PropertyValue>(name: string): T | undefined {
        return this.values.get(name) as T | undefined;
    }

    /** A nested Properties is stored as its JSON object, as natively. */
    set(name: string, value: PropertyValue | Properties): void {
        this.values.set(name, value instanceof Properties ? value.toJSON() : value);
    }

    /** Mirrors get<Properties>: a nested object as Properties. */
    getProperties(name: string): Properties {
        const v = this.values.get(name);
        return new Properties(v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, PropertyValue>) : undefined);
    }

    entries(): IterableIterator<[string, PropertyValue]> {
        return this.values.entries();
    }

    toJSON(): Record<string, PropertyValue> {
        return Object.fromEntries(this.values);
    }
}

/** The visitor a serialize() method calls per field (native `ar(name, value)`); returns the field's new value. */
export type PropertiesArchive = <T>(name: string, value: T) => T;
export interface PropertiesSerializable {
    serialize(ar: PropertiesArchive): void;
}
const isSerializable = (v: unknown): v is PropertiesSerializable => typeof (v as PropertiesSerializable | null)?.serialize === "function";

/** Mirrors PropertiesWriter: an object's serialize() fields as Properties (serializable members nest). */
export const PropertiesWriter = {
    write(value: PropertiesSerializable): Properties {
        const props = new Properties();
        value.serialize((name, v) => {
            props.set(name, isSerializable(v) ? PropertiesWriter.write(v) : (v as PropertyValue));
            return v;
        });
        return props;
    },
};

/** Mirrors PropertiesReader: a default-constructed object with the fields present in `props` read back. */
export const PropertiesReader = {
    read<T extends PropertiesSerializable>(ctor: new () => T, props: Properties): T {
        const value = new ctor();
        const readInto = (target: PropertiesSerializable, p: Properties) =>
            target.serialize((name, v) => {
                if (isSerializable(v)) {
                    if (p.has(name)) readInto(v, p.getProperties(name));
                    return v;
                }
                return p.has(name) ? (p.getOpt(name) as typeof v) : v;
            });
        readInto(value, props);
        return value;
    },
};
