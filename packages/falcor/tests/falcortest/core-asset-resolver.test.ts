/**
 * Transplant of FalcorTest Core/AssetResolverTests.cpp over a fake URL tree (no file system on the web).
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
const unresolved = "";

describe("AssetResolverTests", () => {
    it("AssetResolver", async () => {
        // Test resolving absolute paths (resolvePathPattern needs a directory listing: skipped).
        {
            const resolver = new AssetResolver(exists);
            resolver.addSearchPath(`${kTestRoot}/media1`);
            expect(await resolver.resolvePath(`${kTestRoot}/media2/asset1`)).toBe(`${kTestRoot}/media2/asset1`);
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
