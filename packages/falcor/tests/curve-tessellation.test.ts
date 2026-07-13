import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { CubicSpline, convertToLinearSweptSphere, extractBasisCurvesFromUsda } from "../src/Scene/Curves/CurveTessellation.js";
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
