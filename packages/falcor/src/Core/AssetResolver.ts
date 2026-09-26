/**
 * Asset path resolution mirroring Falcor/Core/AssetResolver.h over URLs.
 * Web divergence (docs §9): file existence is an HTTP HEAD probe, so
 * resolvePath is async; resolvePathPattern globs over the dev server's directory
 * listing (`/__webfalcor/ls`, scripts/vite-plugin-dir-listing.mjs; none on static hosting).
 */

import { ArgumentError } from "./Error.js";
import { Logger } from "../Utils/Logger.js";

/** Mirrors Falcor::AssetCategory. */
export enum AssetCategory {
    Any,
    Scene,
    Texture,
    Count,
}

/** Mirrors Falcor::SearchPathPriority. */
export enum SearchPathPriority {
    First,
    Last,
}

/** Web analog of getProjectDirectory()/"media": the served Falcor media tree. */
export const kProjectMediaUrl = "/Falcor/media";

export type ExistsProbe = (url: string) => Promise<boolean>;
/** Lists a directory's regular files (names), or null if it doesn't exist or listings are unavailable. */
export type DirectoryLister = (url: string) => Promise<string[] | null>;

/** The dev server's directory listing (scripts/vite-plugin-dir-listing.mjs): files and subdirectories of a served path. */
export async function fetchDirectoryListing(url: string): Promise<{ files: string[]; dirs: string[] } | null> {
    const origin = typeof location !== "undefined" ? location.origin : "http://localhost";
    const u = new URL(url, origin);
    if (u.origin !== origin) return null;
    let pending = listingCache.get(u.pathname);
    if (!pending) {
        pending = (async () => {
            try {
                const res = await fetch(`/__webfalcor/ls?path=${encodeURIComponent(u.pathname)}`);
                if (!res.ok || !(res.headers.get("content-type") ?? "").includes("json")) return null;
                return (await res.json()) as { files: string[]; dirs: string[] };
            } catch {
                return null;
            }
        })();
        listingCache.set(u.pathname, pending);
    }
    return pending;
}
const listingCache = new Map<string, Promise<{ files: string[]; dirs: string[] } | null>>();

/** Default lister over fetchDirectoryListing. */
export const listDirectory: DirectoryLister = async (url) => (await fetchDirectoryListing(url))?.files ?? null;

/** Virtual file-system roots Python scripts see the served tree under (scene scripts, Testbed); paths map back by dropping them. */
export const kPythonFsRoots = ["/mogwai", "/testbed"];
function fromPythonFs(path: string): string {
    for (const root of kPythonFsRoots) if (path.startsWith(`${root}/`)) return path.slice(root.length);
    return path;
}

/** Absolute for URL purposes: root-relative path or scheme-qualified. */
export function isAbsoluteUrl(path: string): boolean {
    return path.startsWith("/") || /^[a-z][a-z0-9+.-]*:/i.test(path);
}

/** Collapses "." / ".." / duplicate-slash segments, keeping any scheme://host prefix. */
export function normalizeUrl(url: string): string {
    if (/^(data|blob):/i.test(url)) return url;
    const m = /^([a-z][a-z0-9+.-]*:\/\/[^/]*)?(.*)$/i.exec(url)!;
    const origin = m[1] ?? "";
    const parts: string[] = [];
    (m[2] ?? "").split("/").forEach((seg, i) => {
        if (seg === "." || (seg === "" && i > 0)) return;
        if (seg === "..") {
            if (parts.length > 0 && parts[parts.length - 1] !== "" && parts[parts.length - 1] !== "..") parts.pop();
            else if (parts.length === 0 || parts[0] !== "") parts.push("..");
            return;
        }
        parts.push(seg);
    });
    return origin + parts.join("/");
}

/** `<base>/<rel>` normalized (std::filesystem::path operator/ analog). */
export function joinUrl(base: string, rel: string): string {
    if (!base || isAbsoluteUrl(rel)) return normalizeUrl(rel);
    return normalizeUrl(`${base.replace(/\/+$/, "")}/${rel}`);
}

const probeCache = new Map<string, Promise<boolean>>();

/** Default existence probe: HEAD (GET range fallback); dev-server HTML fallbacks don't count. */
export function urlExists(url: string): Promise<boolean> {
    if (/^(data|blob):/i.test(url)) return Promise.resolve(true);
    let pending = probeCache.get(url);
    if (!pending) {
        pending = probeUrl(url);
        probeCache.set(url, pending);
    }
    return pending;
}

async function probeUrl(url: string): Promise<boolean> {
    try {
        let res = await fetch(url, { method: "HEAD" });
        if (res.status === 405 || res.status === 501) res = await fetch(url, { headers: { Range: "bytes=0-0" } });
        if (!res.ok) return false;
        const type = res.headers.get("content-type") ?? "";
        return !(type.includes("text/html") && !/\.x?html?$/i.test(url));
    } catch {
        return false;
    }
}

/** Forgets probe results (tests, or after files appear on the server). */
export function clearUrlExistsCache(): void {
    probeCache.clear();
}

export class AssetResolver {
    private searchContexts: string[][] = [[], [], []];

    constructor(
        private readonly exists: ExistsProbe = urlExists,
        private readonly list: DirectoryLister = listDirectory,
    ) {}

    /** Copy (native AssetResolver is a value type: Mogwai saves/restores the default). */
    clone(): AssetResolver {
        const r = new AssetResolver(this.exists, this.list);
        r.searchContexts = this.searchContexts.map((c) => [...c]);
        return r;
    }

    /**
     * Mirrors AssetResolver::resolvePath: absolute URLs must exist as given;
     * relative paths try the category's search paths, then Any. "" when unresolved.
     */
    async resolvePath(path: string, category: AssetCategory = AssetCategory.Any): Promise<string> {
        this.checkCategory(category);
        if (!path) return "";
        path = fromPythonFs(path);
        if (isAbsoluteUrl(path)) {
            const url = normalizeUrl(path);
            return (await this.exists(url)) ? url : "";
        }
        let resolved = await this.resolveIn(category, path);
        if (!resolved && category !== AssetCategory.Any) resolved = await this.resolveIn(AssetCategory.Any, path);
        if (!resolved) Logger.warning(`Failed to resolve path '${path}' for asset type '${AssetCategory[category]}'.`);
        return resolved;
    }

    /**
     * Mirrors resolvePathPattern: files in the directory `path` (absolute, else under the search paths) whose
     * name fully matches the ECMAScript regex `pattern` (std::regex_match), in listing order.
     */
    async resolvePathPattern(path: string, pattern: string, firstMatchOnly = false, category: AssetCategory = AssetCategory.Any): Promise<string[]> {
        this.checkCategory(category);
        const regex = new RegExp(`^(?:${pattern})$`);
        path = fromPythonFs(path);
        let resolved = isAbsoluteUrl(path) ? await this.glob(normalizeUrl(path), regex, firstMatchOnly) : [];
        if (resolved.length === 0 && !isAbsoluteUrl(path)) resolved = await this.globIn(category, path, regex, firstMatchOnly);
        if (resolved.length === 0 && !isAbsoluteUrl(path) && category !== AssetCategory.Any) resolved = await this.globIn(AssetCategory.Any, path, regex, firstMatchOnly);
        if (resolved.length === 0) Logger.warning(`Failed to resolve path pattern '${path}/${pattern}' for asset type '${AssetCategory[category]}'.`);
        return resolved;
    }

    /** Mirrors globFilesInDirectory. */
    private async glob(dir: string, regex: RegExp, firstMatchOnly: boolean): Promise<string[]> {
        const files = (await this.list(dir)) ?? [];
        const matches = files.filter((name) => regex.test(name)).map((name) => `${dir.replace(/\/+$/, "")}/${name}`);
        return firstMatchOnly ? matches.slice(0, 1) : matches;
    }
    private async globIn(category: AssetCategory, path: string, regex: RegExp, firstMatchOnly: boolean): Promise<string[]> {
        for (const searchPath of this.searchContexts[category]!) {
            const resolved = await this.glob(normalizeUrl(`${searchPath}/${path}`), regex, firstMatchOnly);
            if (resolved.length > 0) return resolved;
        }
        return [];
    }

    /** Mirrors AssetResolver::addSearchPath (absolute URL; re-adding moves it to the requested end). */
    addSearchPath(path: string, priority: SearchPathPriority = SearchPathPriority.Last, category: AssetCategory = AssetCategory.Any): void {
        if (!isAbsoluteUrl(path)) throw new ArgumentError(`Search path must be absolute: '${path}'`);
        this.checkCategory(category);
        const norm = normalizeUrl(path).replace(/\/+$/, "");
        const list = this.searchContexts[category]!;
        const existing = list.indexOf(norm);
        if (existing >= 0) list.splice(existing, 1);
        if (priority === SearchPathPriority.First) list.unshift(norm);
        else if (priority === SearchPathPriority.Last) list.push(norm);
        else throw new ArgumentError("Invalid search path priority.");
    }

    getSearchPaths(category: AssetCategory = AssetCategory.Any): readonly string[] {
        this.checkCategory(category);
        return this.searchContexts[category]!;
    }

    private async resolveIn(category: AssetCategory, path: string): Promise<string> {
        for (const searchPath of this.searchContexts[category]!) {
            const url = joinUrl(searchPath, path);
            if (await this.exists(url)) return url;
        }
        return "";
    }

    private checkCategory(category: AssetCategory): void {
        if (!(category >= 0 && category < AssetCategory.Count)) throw new ArgumentError("Invalid asset category.");
    }

    // pybind11 spellings (AssetResolver.default_resolver.add_search_path(...)).
    resolve_path(path: string, category?: AssetCategory): Promise<string> { return this.resolvePath(path, category); }
    resolve_path_pattern(path: string, pattern: string, firstMatchOnly?: boolean, category?: AssetCategory): Promise<string[]> { return this.resolvePathPattern(path, pattern, firstMatchOnly, category); }
    add_search_path(path: string, priority?: SearchPathPriority, category?: AssetCategory): void { this.addSearchPath(path, priority, category); }

    private static defaultResolver: AssetResolver | null = null;

    /** Mirrors getDefaultResolver; seeded with the project media tree like SampleApp/Testbed. */
    static getDefaultResolver(): AssetResolver {
        if (!AssetResolver.defaultResolver) {
            AssetResolver.defaultResolver = new AssetResolver();
            AssetResolver.defaultResolver.addSearchPath(kProjectMediaUrl);
        }
        return AssetResolver.defaultResolver;
    }

    /** Replaces the default (native: `getDefaultResolver() = saved`). */
    static setDefaultResolver(resolver: AssetResolver): void {
        AssetResolver.defaultResolver = resolver;
    }

    /** Python module surface: enums + `default_resolver` (native FALCOR_SCRIPT_BINDING). */
    static get pythonBindings(): Record<string, unknown> {
        return {
            AssetCategory: { Any: AssetCategory.Any, Scene: AssetCategory.Scene, Texture: AssetCategory.Texture },
            SearchPathPriority: { First: SearchPathPriority.First, Last: SearchPathPriority.Last },
            AssetResolver: {
                get default_resolver() {
                    return AssetResolver.getDefaultResolver();
                },
            },
        };
    }
}

/**
 * Resolves through `resolver`, falling back to the legacy `<baseUrl>/<path>`
 * join so a miss still fails at fetch time with the attempted URL.
 */
export async function resolveAssetUrl(path: string, baseUrl: string, category: AssetCategory = AssetCategory.Any, resolver = AssetResolver.getDefaultResolver()): Promise<string> {
    return (await resolver.resolvePath(path, category)) || (baseUrl ? `${baseUrl}/${path}` : path);
}

/**
 * Mogwai's loadScript pattern: the script directory becomes the highest-priority
 * search path for the duration of `fn`, then the default resolver is restored.
 */
export async function withScriptSearchPath<T>(directoryUrl: string, fn: () => Promise<T>): Promise<T> {
    const saved = AssetResolver.getDefaultResolver().clone();
    if (directoryUrl && isAbsoluteUrl(directoryUrl)) AssetResolver.getDefaultResolver().addSearchPath(directoryUrl, SearchPathPriority.First);
    try {
        return await fn();
    } finally {
        AssetResolver.setDefaultResolver(saved);
    }
}
