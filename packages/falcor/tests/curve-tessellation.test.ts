import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { CubicSpline, convertToLinearSweptSphere, convertToPolytube, extractBasisCurvesFromUsda, kMeshCompensationScale } from "../src/Scene/Curves/CurveTessellation.js";
import { float4x4 } from "../src/Utils/Math/Matrix.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
// Upstream fixture lives in the Falcor media tree, absent in CI (no clone).
const curvesUsda = resolve(repoRoot, "Falcor/media/test_scenes/curves/two_curves.usda");

describe("CurveTessellation", () => {
    it("matches an independent natural-spline solve (golden values)", () => {
        const s = new CubicSpline([0, 1, 3, 2, 4], 5, 1);
        expect(s.interpolate(0, 0.5, 0)).toBeCloseTo(0.2991071428571429, 6);
        expect(s.interpolate(1, 0.25, 0)).toBeCloseTo(1.5920758928571428, 6);
        expect(s.interpolate(2, 0.5, 0)).toBeCloseTo(2.540178571428571, 6);
        expect(s.interpolate(3, 0.75, 0)).toBeCloseTo(3.257254464285714, 6);
    });

    it("reproduces collinear equally-spaced points exactly (spline linearity)", () => {
        const pts = [0, 0, 0, 1, 2, 3, 2, 4, 6, 3, 6, 9];
        const s = new CubicSpline(pts, 4, 3);
        for (const t of [0.25, 0.5, 0.75]) {
            expect(s.interpolate(1, t, 0)).toBeCloseTo(1 + t, 5);
            expect(s.interpolate(1, t, 1)).toBeCloseTo(2 + 2 * t, 5);
            expect(s.interpolate(1, t, 2)).toBeCloseTo(3 + 3 * t, 5);
        }
    });

    it("passes control points through at USD defaults (subdiv=1, keep=1)", () => {
        // Native interpolate(j, 0) == controlPoint[j] (spline coefficient a).
        const points = [0, 0, 0, 0.1, 1, 0.2, 1, 2, 0.4];
        const widths = [0.1, 0.2, 0.15];
        const r = convertToLinearSweptSphere(1, [3], points, widths, null, 1, 1, 1, 1, 1, float4x4.identity());
        expect(r.points.length).toBe(3);
        expect(r.indices).toEqual(new Uint32Array([0, 1]));
        for (let i = 0; i < 3; i++) {
            expect(r.points[i]!.x).toBeCloseTo(points[i * 3]!, 6);
            expect(r.points[i]!.y).toBeCloseTo(points[i * 3 + 1]!, 6);
            expect(r.points[i]!.z).toBeCloseTo(points[i * 3 + 2]!, 6);
            expect(r.radius[i]).toBeCloseTo(widths[i]! * 0.5, 6);
        }
    });

    it("dedups consecutive duplicate control points", () => {
        const points = [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0];
        const widths = [0.1, 0.1, 0.2, 0.3];
        const r = convertToLinearSweptSphere(1, [4], points, widths, null, 1, 1, 1, 1, 1, float4x4.identity());
        expect(r.points.length).toBe(3);
        expect(r.points[0]!.x).toBe(0);
        expect(r.points[1]!.x).toBe(1);
    });

    it.skipIf(!existsSync(curvesUsda))("extracts BasisCurves from the upstream two_curves.usda", () => {
        const source = readFileSync(curvesUsda, "utf-8");
        const curves = extractBasisCurvesFromUsda(source);
        expect(curves.length).toBe(2);
        expect(curves[0]!.name).toBe("curve0");
        expect(curves[0]!.curveVertexCounts).toEqual(new Uint32Array([5]));
        expect(curves[0]!.points.length).toBe(15);
        expect(curves[0]!.widths.length).toBe(5);
        expect(curves[0]!.points[3]).toBeCloseTo(0.1, 6);
        expect(curves[0]!.widths[1]).toBeCloseTo(0.2, 6);
    });
});

describe("convertToPolytube", () => {
    // curve0 of Falcor's two_curves.usda.
    const points = [0, 0, 0, 0.1, 1, 0.2, 1, 2, 0.4, 1.1, 3, 0.6, 2, 4, 0.8];
    const widths = [0.1, 0.2, 0.15, 0.3, 0.2];

    it("builds a quad tube with one ring per control point", () => {
        const tube = convertToPolytube(1, [5], points, widths, null, 1, 1, 1, 1, 4);
        expect(tube.radii.length).toBe(5 * 4);
        expect(tube.faceVertexIndices.length).toBe(2 * 4 * 4 * 3); // 2 triangles per quad, 4 quads per band, 4 bands
        expect(tube.texCrds).toBeNull();

        for (let j = 0; j < 5; j++) {
            // With subdiv 1 the spline passes through the control points, and the
            // ring radius is half the width scaled by kMeshCompensationScale.
            const c = [points[j * 3]!, points[j * 3 + 1]!, points[j * 3 + 2]!];
            const expectedRadius = 0.5 * kMeshCompensationScale * widths[j]!;
            for (let k = 0; k < 4; k++) {
                const v = j * 4 + k;
                const off = [0, 1, 2].map((a) => tube.vertices[v * 3 + a]! - c[a]!);
                expect(Math.hypot(off[0]!, off[1]!, off[2]!)).toBeCloseTo(expectedRadius, 5);
                expect(tube.radii[v]).toBeCloseTo(expectedRadius, 5);
                // The offset is the unit normal times the radius, perpendicular to the curve.
                const n = [0, 1, 2].map((a) => tube.normals[v * 3 + a]!);
                expect(Math.hypot(n[0]!, n[1]!, n[2]!)).toBeCloseTo(1, 5);
                const fwd = [0, 1, 2].map((a) => tube.tangents[v * 4 + a]!);
                expect(n[0]! * fwd[0]! + n[1]! * fwd[1]! + n[2]! * fwd[2]!).toBeCloseTo(0, 5);
                expect(tube.tangents[v * 4 + 3]).toBe(1);
            }
        }
    });

    it("only references vertices it created, one band at a time", () => {
        const tube = convertToPolytube(2, [5, 5], [...points, ...points.map((v, i) => (i % 3 === 0 ? v + 3 : v))], [...widths, ...widths], null, 1, 1, 1, 1, 4);
        const vertexCount = tube.radii.length;
        expect(vertexCount).toBe(2 * 5 * 4);
        for (const idx of tube.faceVertexIndices) expect(idx).toBeLessThan(vertexCount);
        // No triangle spans the two strands.
        for (let f = 0; f < tube.faceVertexIndices.length; f += 3) {
            const strands = new Set([0, 1, 2].map((c) => Math.floor(tube.faceVertexIndices[f + c]! / 20)));
            expect(strands.size).toBe(1);
        }
    });

    it("subdivides and thins like the swept-sphere path", () => {
        const denser = convertToPolytube(1, [5], points, widths, null, 3, 1, 1, 1, 4);
        expect(denser.radii.length).toBe((4 * 3 + 1) * 4);
        const thinned = convertToPolytube(1, [5], points, widths, null, 3, 1, 2, 1, 4);
        expect(thinned.radii.length).toBe((Math.ceil((4 * 3) / 2) + 1) * 4);
    });
});

describe("USDA curve prim paths", () => {
    it("reports the nesting of each BasisCurves prim", () => {
        const usda = `#usda 1.0
(
    defaultPrim = "Root"
)
def "Root"
{
    def Mesh "tri0" { int[] faceVertexCounts = [3] }
    def BasisCurves "curve0"
    {
        int[] curveVertexCounts = [2]
        point3f[] points = [(0, 0, 0), (1, 0, 0)]
    }
    def Xform "group"
    {
        def BasisCurves "curve1"
        {
            int[] curveVertexCounts = [2]
            point3f[] points = [(0, 0, 0), (0, 1, 0)] ( interpolation = "vertex" )
        }
    }
}`;
        const curves = extractBasisCurvesFromUsda(usda);
        expect(curves.map((c) => c.path)).toEqual(["/Root/curve0", "/Root/group/curve1"]);
    });
});
