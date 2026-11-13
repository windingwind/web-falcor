import { describe, expect, it } from "vitest";
import { generateSmoothNormals } from "../src/Scene/Importer/GltfImporter.js";

const n = (a: Float32Array, v: number) => [a[v * 3]!, a[v * 3 + 1]!, a[v * 3 + 2]!];
const close = (a: number[], b: number[]) => a.forEach((x, i) => expect(x).toBeCloseTo(b[i]!, 6));

describe("generateSmoothNormals (glTF primitives without NORMAL; Assimp GenSmoothNormals semantics)", () => {
    it("a quad in the y = 0 plane wound counter-clockwise seen from +y gets +y everywhere", () => {
        const pos = new Float32Array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]);
        const out = generateSmoothNormals(new Uint32Array([0, 1, 2, 0, 2, 3]), pos);
        for (let v = 0; v < 4; v++) close(n(out, v), [0, 1, 0]);
    });

    it("averages the unit normals of all faces at a position (cube corner: three faces)", () => {
        // corner at the origin shared by triangles of the faces x = 0, y = 0, z = 0 (outward normals -x, -y, -z)
        const pos = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
        const idx = new Uint32Array([0, 2, 1, /* x = 0 */ 0, 3, 2, /* y = 0 */ 0, 1, 3 /* z = 0 */]);
        const out = generateSmoothNormals(idx, pos);
        const s = -1 / Math.sqrt(3);
        close(n(out, 0), [s, s, s]);
    });

    it("vertices that share a position but not other attributes get the same normal", () => {
        // two triangles meeting at a 90 degree edge, unindexed (no shared vertex indices)
        const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, /* z = 0 face, +z */ 0, 0, 0, 0, 0, 1, 1, 0, 0 /* y = 0 face, +y */]);
        const out = generateSmoothNormals(new Uint32Array([0, 1, 2, 3, 4, 5]), pos);
        const h = Math.SQRT1_2;
        close(n(out, 0), [0, h, h]);
        close(n(out, 3), [0, h, h]);
        close(n(out, 2), [0, 0, 1]);
        close(n(out, 4), [0, 1, 0]);
    });

    it("ignores degenerate faces; a vertex used only by degenerate faces gets (0, 0, 1)", () => {
        const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 5, 5, 6, 6, 6]);
        const out = generateSmoothNormals(new Uint32Array([0, 1, 2, 3, 4, 4]), pos);
        close(n(out, 0), [0, 0, 1]);
        close(n(out, 3), [0, 0, 1]);
        close(n(out, 4), [0, 0, 1]);
    });
});
