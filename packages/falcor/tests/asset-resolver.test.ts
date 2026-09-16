import { describe, expect, it } from "vitest";
import { AssetCategory, AssetResolver, SearchPathPriority, isAbsoluteUrl, joinUrl, normalizeUrl, resolveAssetUrl } from "../src/Core/AssetResolver.js";
import { resolveEnvVariables, resolveSearchPaths } from "../src/Utils/PathResolving.js";
import { Settings } from "../src/Utils/Settings.js";

/** Resolver over a fake server: `files` is the set of existing URLs. */
function makeResolver(files: string[]): { resolver: AssetResolver; probes: string[] } {
    const probes: string[] = [];
    const set = new Set(files);
    const resolver = new AssetResolver(async (url) => {
        probes.push(url);
        return set.has(url);
    });
    return { resolver, probes };
}

describe("URL helpers", () => {
    it("isAbsoluteUrl accepts root-relative and scheme-qualified paths", () => {
        expect(isAbsoluteUrl("/Falcor/media")).toBe(true);
        expect(isAbsoluteUrl("https://x/y")).toBe(true);
        expect(isAbsoluteUrl("blob:abc")).toBe(true);
        expect(isAbsoluteUrl("media/x.png")).toBe(false);
        expect(isAbsoluteUrl("./x.png")).toBe(false);
    });
    it("normalizeUrl collapses dot segments and duplicate slashes, keeps the origin", () => {
        expect(normalizeUrl("/a/b/../c/./d")).toBe("/a/c/d");
        expect(normalizeUrl("/a//b")).toBe("/a/b");
        expect(normalizeUrl("http://h:1/a/../b")).toBe("http://h:1/b");
        expect(normalizeUrl("../x")).toBe("../x");
        expect(normalizeUrl("/../x")).toBe("/x");
    });
    it("joinUrl mirrors path::operator/ (absolute right operand wins)", () => {
        expect(joinUrl("/media/", "scenes/a.pyscene")).toBe("/media/scenes/a.pyscene");
        expect(joinUrl("/media", "/abs/a.pyscene")).toBe("/abs/a.pyscene");
        expect(joinUrl("", "rel/a")).toBe("rel/a");
    });
});

describe("AssetResolver (native AssetResolver.cpp semantics)", () => {
    it("absolute URLs resolve to themselves only if they exist", async () => {
        const { resolver } = makeResolver(["/x/a.png"]);
        expect(await resolver.resolvePath("/x/a.png")).toBe("/x/a.png");
        expect(await resolver.resolvePath("/x/./b/../a.png")).toBe("/x/a.png");
        expect(await resolver.resolvePath("/x/missing.png")).toBe("");
    });

    it("relative paths search the category's paths in order, then Any", async () => {
        const { resolver, probes } = makeResolver(["/tex/t.png", "/any/s.pyscene", "/scenes/s.pyscene"]);
        resolver.addSearchPath("/any");
        resolver.addSearchPath("/scenes", SearchPathPriority.Last, AssetCategory.Scene);
        resolver.addSearchPath("/tex", SearchPathPriority.Last, AssetCategory.Texture);
        expect(await resolver.resolvePath("s.pyscene", AssetCategory.Scene)).toBe("/scenes/s.pyscene");
        expect(await resolver.resolvePath("t.png", AssetCategory.Scene)).toBe(""); // Scene → Any never sees the Texture paths
        probes.length = 0;
        expect(await resolver.resolvePath("t.png", AssetCategory.Texture)).toBe("/tex/t.png");
        expect(probes).toEqual(["/tex/t.png"]);
        // Scene category misses fall through to Any.
        probes.length = 0;
        expect(await resolver.resolvePath("s.pyscene", AssetCategory.Texture)).toBe("/any/s.pyscene");
        expect(probes).toEqual(["/tex/s.pyscene", "/any/s.pyscene"]);
        expect(await resolver.resolvePath("nowhere.png")).toBe("");
        expect(await resolver.resolvePath("")).toBe("");
    });

    it("addSearchPath honours priority, dedupes and requires absolute URLs", () => {
        const { resolver } = makeResolver([]);
        resolver.addSearchPath("/a");
        resolver.addSearchPath("/b");
        resolver.addSearchPath("/c/", SearchPathPriority.First);
        expect(resolver.getSearchPaths()).toEqual(["/c", "/a", "/b"]);
        resolver.addSearchPath("/a", SearchPathPriority.First); // moves, no duplicate
        expect(resolver.getSearchPaths()).toEqual(["/a", "/c", "/b"]);
        expect(() => resolver.addSearchPath("relative/dir")).toThrow(/absolute/);
        expect(() => resolver.addSearchPath("/x", SearchPathPriority.Last, AssetCategory.Count)).toThrow(/category/);
    });

    it("clone is an independent copy; default resolver starts with the project media tree", () => {
        const { resolver } = makeResolver([]);
        resolver.addSearchPath("/a");
        const copy = resolver.clone();
        copy.addSearchPath("/b");
        expect(resolver.getSearchPaths()).toEqual(["/a"]);
        expect(copy.getSearchPaths()).toEqual(["/a", "/b"]);
        expect(AssetResolver.getDefaultResolver().getSearchPaths()).toContain("/Falcor/media");
    });

    it("resolvePathPattern is unavailable on the web (returns [])", async () => {
        const { resolver } = makeResolver([]);
        expect(await resolver.resolvePathPattern("/dir", "tex_[0-9]+.png")).toEqual([]);
    });

    it("resolveAssetUrl falls back to the legacy <baseUrl>/<path> join on a miss", async () => {
        const { resolver } = makeResolver(["/search/found.png"]);
        resolver.addSearchPath("/search");
        expect(await resolveAssetUrl("found.png", "/base", AssetCategory.Any, resolver)).toBe("/search/found.png");
        expect(await resolveAssetUrl("missing.png", "/base", AssetCategory.Any, resolver)).toBe("/base/missing.png");
    });

    it("exposes pybind spellings and the python module surface", async () => {
        const { resolver } = makeResolver(["/p/a.png"]);
        resolver.add_search_path("/p", SearchPathPriority.Last, AssetCategory.Any);
        expect(await resolver.resolve_path("a.png")).toBe("/p/a.png");
        const py = AssetResolver.pythonBindings as { AssetCategory: Record<string, number>; AssetResolver: { default_resolver: AssetResolver } };
        expect(py.AssetCategory["Scene"]).toBe(AssetCategory.Scene);
        expect(py.AssetResolver.default_resolver).toBe(AssetResolver.getDefaultResolver());
    });
});

describe("PathResolving (Renderman-style search path lists)", () => {
    it("resolveEnvVariables expands ${VAR} and flags unbalanced tokens", () => {
        expect(resolveEnvVariables("${A}/x;${B}", (n) => (n === "A" ? "/a" : undefined))).toEqual({ value: "/a/x;", ok: true });
        expect(resolveEnvVariables("/x/${A").ok).toBe(false);
    });
    it("resolveSearchPaths splits on ';', expands @ and &, rejects relative entries", () => {
        const r = resolveSearchPaths(["/cur"], ["/new/;@", "&;rel/path", "/other/../o2"], ["/std1", "/std2"]);
        expect(r.resolved).toEqual(["/new", "/std1", "/std2", "/cur", "/o2"]);
        expect(r.invalid).toEqual(["rel/path"]);
    });
});

describe("Settings search directories (Settings::updateSearchPaths)", () => {
    it("nested or flattened searchpath keys build categories; searchpath overrides the standard path", () => {
        const s = new Settings();
        s.addOptions({ standardsearchpath: { media: "/std/media" } });
        expect(s.getSearchDirectories("media")).toEqual(["/std/media"]);
        s.addOptions({ "searchpath:media": ["/mine;@"] });
        expect(s.getSearchDirectories("media")).toEqual(["/mine", "/std/media"]);
        s.addOptions({ searchpath: { media: "&;/more" } });
        expect(s.getSearchDirectories("media")).toEqual(["/mine", "/std/media", "/more"]);
        expect(s.getSearchDirectories("other")).toEqual([]);
    });
    it("rejects non-absolute search paths like FALCOR_CHECK", () => {
        const s = new Settings();
        expect(() => s.addOptions({ searchpath: { media: "relative/dir" } })).toThrow(/invalid paths/);
    });
});
