/**
 * Global settings mirroring Utils/Settings/Settings.h + AttributeFilters:
 * colon-flattened option dictionaries and ordered attribute filters with
 * full-match regexes (deprecated `name.filter` syntax included). Web
 * divergence (docs §9): no settings.json autoload and no search-directory
 * categories (no filesystem).
 */

export type SettingsValue = boolean | number | string | null | SettingsValue[] | { [key: string]: SettingsValue };

/** Nested dicts flatten to colon-separated keys (settings::detail::flattenDictionary). */
function flattenDictionary(dict: Record<string, SettingsValue>, prefix = "", out: Record<string, SettingsValue> = {}): Record<string, SettingsValue> {
    for (const [key, value] of Object.entries(dict)) {
        const name = prefix ? `${prefix}:${key}` : key;
        if (value !== null && typeof value === "object" && !Array.isArray(value)) flattenDictionary(value, name, out);
        else out[name] = value;
    }
    return out;
}

interface FilterRecord {
    name: string;
    regex: RegExp;
    /** Flattened attributes; null values unapply the attribute. */
    attributes: Record<string, SettingsValue>;
}

/** std::regex_match semantics: the whole shape name must match. */
function fullMatch(regex: RegExp, name: string): boolean {
    const m = name.match(regex);
    return m !== null && m[0] === name;
}

export class Settings {
    private options = new Map<string, SettingsValue>();
    private filters: FilterRecord[] = [];

    /** Mirrors Settings::addOptions (nested dicts flatten with ':'). */
    addOptions(options: Record<string, SettingsValue>): void {
        for (const [k, v] of Object.entries(flattenDictionary(options))) this.options.set(k, v);
    }

    clearOptions(): void {
        this.options.clear();
    }

    /** Mirrors Settings::getOption (flattened name, e.g. "PipedOutput:enable"). */
    getOption<T extends SettingsValue>(name: string, defaultValue: T): T {
        return (this.options.get(name) as T) ?? defaultValue;
    }

    getOptions(): Record<string, SettingsValue> {
        return Object.fromEntries(this.options);
    }

    /** Mirrors Settings::addFilteredAttributes (dict or array of dicts). */
    addFilteredAttributes(attributes: Record<string, SettingsValue> | Record<string, SettingsValue>[]): void {
        if (Array.isArray(attributes)) {
            for (const dict of attributes) this.addDictionary(dict);
        } else {
            this.addDictionary(attributes);
        }
    }

    clearFilteredAttributes(): void {
        this.filters = [];
    }

    /** Mirrors AttributeFilter::addDictionary. */
    private addDictionary(dict: Record<string, SettingsValue>): void {
        const name = typeof dict["name"] === "string" ? dict["name"] : `filter_${this.filters.length}`;
        const regexStr = typeof dict["regex"] === "string" ? dict["regex"] : ".*";
        const attrBlock = dict["attributes"];
        if (dict["regex"] !== undefined && (attrBlock === null || typeof attrBlock !== "object" || Array.isArray(attrBlock))) {
            throw new Error("Settings: if `regex` is present, `attributes` must be present as well");
        }
        const flattened =
            attrBlock !== undefined && attrBlock !== null && typeof attrBlock === "object" && !Array.isArray(attrBlock)
                ? flattenDictionary(attrBlock as Record<string, SettingsValue>)
                : flattenDictionary(dict);
        delete flattened["name"];
        delete flattened["regex"];

        const attributes = this.processDeprecatedFilters(name, flattened, regexStr);
        if (Object.keys(attributes).length > 0) this.filters.push({ name, regex: new RegExp(regexStr), attributes });
    }

    /** Mirrors AttributeFilter::processDeprecatedFilters (`attr.filter` keys). */
    private processDeprecatedFilters(name: string, flattened: Record<string, SettingsValue>, regexStr: string): Record<string, SettingsValue> {
        const processed = new Set<string>();
        for (const [filterKey, filterValue] of Object.entries(flattened)) {
            if (!filterKey.endsWith(".filter")) continue;
            const attrKey = filterKey.slice(0, -".filter".length);
            if (!(attrKey in flattened)) throw new Error(`Settings: found filter '${filterKey}' but not attribute '${attrKey}'`);
            if (regexStr !== ".*") throw new Error(`Settings: filtered attribute '${attrKey}' requires the default '.*' regex`);

            let filterRegexStr: string;
            let negated = false;
            if (typeof filterValue === "string") {
                filterRegexStr = filterValue;
            } else if (Array.isArray(filterValue) && typeof filterValue[0] === "string") {
                filterRegexStr = filterValue[0];
                negated = filterValue.length === 2 && filterValue[1] === true;
            } else {
                throw new Error(`Settings: '${filterKey}' must be a string or [regex, negate] array`);
            }

            if (!negated) {
                this.filters.push({ name: `${name}_${filterKey}`, regex: new RegExp(filterRegexStr), attributes: { [attrKey]: flattened[attrKey]! } });
            } else {
                // Apply everywhere, then unapply (null) where the regex matches.
                this.filters.push({ name: `${name}_${filterKey}_apply`, regex: new RegExp(".*"), attributes: { [attrKey]: flattened[attrKey]! } });
                this.filters.push({ name: `${name}_${filterKey}_unapply`, regex: new RegExp(filterRegexStr), attributes: { [attrKey]: null } });
            }
            processed.add(filterKey);
            processed.add(attrKey);
        }
        if (processed.size === 0) return flattened;
        return Object.fromEntries(Object.entries(flattened).filter(([k]) => !processed.has(k)));
    }

    /** Mirrors Settings::getAttributes: ordered merge over matching filters. */
    getAttributes(shapeName: string): Record<string, SettingsValue> {
        const out: Record<string, SettingsValue> = {};
        for (const f of this.filters) {
            if (!fullMatch(f.regex, shapeName)) continue;
            for (const [k, v] of Object.entries(f.attributes)) {
                if (v === null) delete out[k];
                else out[k] = v;
            }
        }
        return out;
    }

    /** Mirrors Settings::getAttribute. */
    getAttribute<T extends SettingsValue>(shapeName: string, attributeName: string, defaultValue: T): T {
        return (this.getAttributes(shapeName)[attributeName] as T) ?? defaultValue;
    }
}
