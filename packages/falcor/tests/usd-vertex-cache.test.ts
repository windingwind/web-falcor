/**
 * Vertex caches (Scene.sampleVertexCache, mirroring AnimatedVertexCache) and native's
 * triangulate() port (Scene/Importer/UsdTriangulate.ts).
 */
import { describe, expect, it } from "vitest";
import { sampleVertexCache } from "../src/Scene/Scene.js";
import { triangulateUsdMesh } from "../src/Scene/Importer/UsdTriangulate.js";
import type { UsdaSubdivMesh } from "../src/Scene/Importer/UsdaScene.js";
import { float2, float3, float4 } from "../src/Utils/Math/Vector.js";

const vertex = (x: number) => ({ position: new float3(x, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) });
const cache = { times: [0.25, 0.75, 1.25], frames: [[vertex(0)], [vertex(2)], [vertex(4)]] };

describe("sampleVertexCache", () => {
    it("interpolates between samples and loops past the last", () => {
        expect(sampleVertexCache(cache, 0.5, false, cache.frames[0]!)[0]!.position.x).toBeCloseTo(1, 6);
        expect(sampleVertexCache(cache, 1.25 + 0.5, false, cache.frames[0]!)[0]!.position.x).toBeCloseTo(1, 6); // fmod(1.75, 1.25)
    });
    it("holds, or cycles from the last sample, before the first", () => {
        expect(sampleVertexCache(cache, 0.1, false, cache.frames[0]!)[0]!.position.x).toBe(0);
        // Cycle: last -> first with t = time / firstTime.
        expect(sampleVertexCache(cache, 0.125, true, cache.frames[0]!)[0]!.position.x).toBeCloseTo(2, 6);
    });
});

describe("triangulateUsdMesh", () => {
    const quad = (orientation: string): UsdaSubdivMesh => ({
        scheme: "none", orientation, interpolateBoundary: "edgeAndCorner", faceVaryingLinearInterpolation: "cornersPlus1",
        points: [], faceVertexCounts: [4, 3], faceVertexIndices: [0, 1, 2, 3, 0, 2, 4], skinned: false, hasNormals: false, holeIndices: [1],
        st: { values: [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0.5, 0.5], interpolation: "faceVarying" },
    });
    const points = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1];
    it("fans faces, skips holes, and generates flat normals", () => {
        const m = triangulateUsdMesh(quad("rightHanded"), points, undefined);
        expect(m.positions.length).toBe(2 * 3 * 3); // the quad's two triangles; face 1 is a hole
        expect(Array.from(m.positions.slice(0, 9))).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0]);
        expect(Array.from(m.normals.slice(0, 3))).toEqual([0, 0, 1]);
        expect(Array.from(m.uvs!.slice(6, 12))).toEqual([0, 0, 1, 1, 0, 1]); // face-varying corners 0, 2, 3
    });
    it("reverses the fan for left-handed meshes", () => {
        const m = triangulateUsdMesh(quad("leftHanded"), points, undefined);
        expect(Array.from(m.positions.slice(0, 9))).toEqual([0, 0, 0, 1, 1, 0, 1, 0, 0]);
        expect(Array.from(m.normals.slice(0, 3))).toEqual([0, 0, -1]);
    });
});
