/**
 * Property container mirroring Falcor/Utils/Properties.h (JSON-backed pass
 * configuration; the Python dicts in graph scripts land here).
 */

export type PropertyValue = boolean | number | string | number[] | PropertyValue[] | { [key: string]: PropertyValue };

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

    set(name: string, value: PropertyValue): void {
        this.values.set(name, value);
    }

    entries(): IterableIterator<[string, PropertyValue]> {
        return this.values.entries();
    }

    toJSON(): Record<string, PropertyValue> {
        return Object.fromEntries(this.values);
    }
}
