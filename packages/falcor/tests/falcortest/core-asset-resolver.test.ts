/**
 * Transplant of FalcorTest Core/AssetResolverTests.cpp over a fake URL tree (no file system on the web):
 * existence and directory listings come from the fake tree.
 */

import { describe, expect, it } from "vitest";
import { AssetCategory, AssetResolver, SearchPathPriority } from "../../src/Core/AssetResolver.js";

const kTestRoot = "/asset_test_root";
const kTestFiles = [
    "media1/asset1",
    "media2/asset1",
    "media2/asset2",
    "media3/asset1",
    "media3/asset2",
    "media3/asset3",
    "media4/textures/mip0.png",
    "media4/textures/mip1.png",
    "media4/textures/mip2.png",
    "media4/textures/mip3.png",
].map((f) => `${kTestRoot}/${f}`);

const exists = (url: string) => Promise.resolve(kTestFiles.includes(url));
const list = (url: string) => {
    const files = kTestFiles.filter((f) => f.startsWith(`${url}/`) && !f.slice(url.length + 1).includes("/")).map((f) => f.slice(url.length + 1));
    return Promise.resolve(files.length > 0 ? files : null);
};
const kMips = [0, 1, 2, 3].map((i) => `${kTestRoot}/media4/textures/mip${i}.png`);
const unresolved = "";

describe("AssetResolverTests", () => {
    it("AssetResolver", async () => {
        // Test resolving absolute paths.
        {
            const resolver = new AssetResolver(exists, list);
            resolver.addSearchPath(`${kTestRoot}/media1`);
            expect(await resolver.resolvePath(`${kTestRoot}/media2/asset1`)).toBe(`${kTestRoot}/media2/asset1`);
            const resolved = await resolver.resolvePathPattern(`${kTestRoot}/media4/textures`, String.raw`mip[0-9]\.png`);
            expect(resolved.sort()).toEqual(kMips);
        }

        // Test resolving with search paths.
        {
            const resolver = new AssetResolver(exists);

            resolver.addSearchPath(`${kTestRoot}/media1`);
            expect(await resolver.resolvePath("asset1")).toBe(`${kTestRoot}/media1/asset1`);
            expect(await resolver.resolvePath("asset2")).toBe(unresolved);
            expect(await resolver.resolvePath("asset3")).toBe(unresolved);
            expect(await resolver.resolvePath("asset4")).toBe(unresolved);

            resolver.addSearchPath(`${kTestRoot}/media2`);
            expect(await resolver.resolvePath("asset1")).toBe(`${kTestRoot}/media1/asset1`);
            expect(await resolver.resolvePath("asset2")).toBe(`${kTestRoot}/media2/asset2`);
            expect(await resolver.resolvePath("asset3")).toBe(unresolved);
            expect(await resolver.resolvePath("asset4")).toBe(unresolved);

            resolver.addSearchPath(`${kTestRoot}/media3`);
            expect(await resolver.resolvePath("asset1")).toBe(`${kTestRoot}/media1/asset1`);
            expect(await resolver.resolvePath("asset2")).toBe(`${kTestRoot}/media2/asset2`);
            expect(await resolver.resolvePath("asset3")).toBe(`${kTestRoot}/media3/asset3`);
            expect(await resolver.resolvePath("asset4")).toBe(unresolved);
        }

        // Test asset categories.
        {
            const resolver = new AssetResolver(exists);
            resolver.addSearchPath(`${kTestRoot}/media3`, SearchPathPriority.Last, AssetCategory.Any);
            resolver.addSearchPath(`${kTestRoot}/media2`, SearchPathPriority.Last, AssetCategory.Scene);
            resolver.addSearchPath(`${kTestRoot}/media1`, SearchPathPriority.Last, AssetCategory.Texture);

            expect(await resolver.resolvePath("asset1", AssetCategory.Any)).toBe(`${kTestRoot}/media3/asset1`);
            expect(await resolver.resolvePath("asset1", AssetCategory.Scene)).toBe(`${kTestRoot}/media2/asset1`);
            expect(await resolver.resolvePath("asset1", AssetCategory.Texture)).toBe(`${kTestRoot}/media1/asset1`);

            expect(await resolver.resolvePath("asset2", AssetCategory.Any)).toBe(`${kTestRoot}/media3/asset2`);
            expect(await resolver.resolvePath("asset2", AssetCategory.Scene)).toBe(`${kTestRoot}/media2/asset2`);
            expect(await resolver.resolvePath("asset2", AssetCategory.Texture)).toBe(`${kTestRoot}/media3/asset2`);

            expect(await resolver.resolvePath("asset3", AssetCategory.Any)).toBe(`${kTestRoot}/media3/asset3`);
            expect(await resolver.resolvePath("asset3", AssetCategory.Scene)).toBe(`${kTestRoot}/media3/asset3`);
            expect(await resolver.resolvePath("asset3", AssetCategory.Texture)).toBe(`${kTestRoot}/media3/asset3`);
        }

        // Test resolving patterns with search paths.
        {
            const resolver = new AssetResolver(exists, list);
            resolver.addSearchPath(`${kTestRoot}/media4`);
            expect((await resolver.resolvePathPattern("textures", String.raw`mip[0-9]\.png`)).sort()).toEqual(kMips);
            const first = await resolver.resolvePathPattern("textures", String.raw`mip[0-9]\.png`, true);
            expect(first.length).toBe(1);
            expect(kMips).toContain(first[0]);
        }

        // Test search path priorities.
        {
            const resolver = new AssetResolver(exists);
            resolver.addSearchPath(`${kTestRoot}/media1`);
            expect(await resolver.resolvePath("asset1")).toBe(`${kTestRoot}/media1/asset1`);
            resolver.addSearchPath(`${kTestRoot}/media2`, SearchPathPriority.Last);
            expect(await resolver.resolvePath("asset1")).toBe(`${kTestRoot}/media1/asset1`);
            resolver.addSearchPath(`${kTestRoot}/media3`, SearchPathPriority.First);
            expect(await resolver.resolvePath("asset1")).toBe(`${kTestRoot}/media3/asset1`);
        }
    });
});
