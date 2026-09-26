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

/** Lexical std::filesystem::weakly_canonical for '/'-separated paths (drops '.', resolves '..'). */
export function weaklyCanonical(path: string): string {
    const root = /^([a-zA-Z]:)?\//.exec(path)?.[0] ?? "";
    const out: string[] = [];
    for (const part of path.slice(root.length).split("/")) {
        if (part === "" || part === ".") continue;
        if (part === ".." && out.length > 0 && out[out.length - 1] !== "..") out.pop();
        else if (part !== ".." || !root) out.push(part);
    }
    return root + out.join("/");
}

export type FileChecker = (path: string) => boolean;

/**
 * Mirrors resolvePath: an absolute path if it exists, a '.'-relative one against the working directory,
 * otherwise the first search path holding the file; "" when nothing matches.
 */
export function resolvePath(searchPaths: readonly string[], currentWorkingDirectory: string, filePath: string, fileChecker: FileChecker): string {
    if (!filePath) return "";
    const join = (base: string, rel: string) => (base.endsWith("/") ? base : `${base}/`) + rel;
    if (/^([a-zA-Z]:)?\//.test(filePath)) return fileChecker(filePath) ? weaklyCanonical(filePath) : "";
    if (filePath[0] === ".") {
        const result = join(currentWorkingDirectory, filePath);
        return fileChecker(result) ? weaklyCanonical(result) : "";
    }
    for (const searchPath of searchPaths) {
        const result = join(searchPath, filePath);
        if (fileChecker(result)) return weaklyCanonical(result);
    }
    return "";
}
