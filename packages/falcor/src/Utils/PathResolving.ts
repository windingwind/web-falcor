/**
 * Search-path list semantics mirroring Falcor/Utils/PathResolving.h
 * (Renderman-style: `;` lists, `@` = standard paths, `&` = current paths,
 * `${VAR}` expansion). Paths are URLs here; there is no process environment.
 */

import { isAbsoluteUrl, normalizeUrl } from "../Core/AssetResolver.js";

export type EnvVarResolver = (name: string) => string | undefined;

/** No environment in the browser: every variable resolves to undefined (→ empty). */
export const noEnvResolver: EnvVarResolver = () => undefined;

/** Mirrors resolveEnvVariables: in-place `${VAR}` expansion; false on unbalanced tokens. */
export function resolveEnvVariables(str: string, envResolver: EnvVarResolver = noEnvResolver, beginToken = "${", endToken = "}"): { value: string; ok: boolean } {
    let out = "";
    let pos = 0;
    for (;;) {
        const begin = str.indexOf(beginToken, pos);
        if (begin < 0) return { value: out + str.slice(pos), ok: true };
        const end = str.indexOf(endToken, begin + beginToken.length);
        if (end < 0) return { value: out + str.slice(pos, begin), ok: false };
        out += str.slice(pos, begin) + (envResolver(str.slice(begin + beginToken.length, end)) ?? "");
        pos = end + endToken.length;
    }
}

export interface ResolvedPaths {
    resolved: string[];
    /** Entries that were not absolute (error reporting). */
    invalid: string[];
}

/** Mirrors resolveSearchPaths: builds the new search-path list from `update` over `current`/`standard`. */
export function resolveSearchPaths(current: readonly string[], update: readonly string[], standard: readonly string[], envResolver: EnvVarResolver = noEnvResolver): ResolvedPaths {
    const result: ResolvedPaths = { resolved: [], invalid: [] };
    for (const entry of update) {
        const env = resolveEnvVariables(entry, envResolver);
        if (!env.ok) {
            result.invalid.push(entry);
            continue;
        }
        for (const item of env.value.split(";")) {
            if (item === "") continue;
            if (item === "&") result.resolved.push(...current);
            else if (item === "@") result.resolved.push(...standard);
            else if (isAbsoluteUrl(item)) result.resolved.push(normalizeUrl(item).replace(/\/+$/, ""));
            else result.invalid.push(item);
        }
    }
    return result;
}
