import { describe, expect, it } from "vitest";
import { CubicSpline } from "../src/Utils/Math/CubicSpline.js";

describe("CubicSpline (Utils/Math/CubicSpline.h natural spline)", () => {
    const pts = [0, 0, 1, 2, 3, 1, 4, 3]; // 4 control points, 2 lanes (x,y)
    const s = new CubicSpline(pts, 4, 2);

    it("passes through the control points at section ends", () => {
        for (let i = 0; i < 3; i++) {
            expect(s.interpolate(i, 0, 0)).toBeCloseTo(pts[i * 2]!, 6);
            expect(s.interpolate(i, 0, 1)).toBeCloseTo(pts[i * 2 + 1]!, 6);
            expect(s.interpolate(i, 1, 0)).toBeCloseTo(pts[(i + 1) * 2]!, 5);
            expect(s.interpolate(i, 1, 1)).toBeCloseTo(pts[(i + 1) * 2 + 1]!, 5);
        }
    });

    it("is C1/C2 across sections with natural (zero) end curvature", () => {
        for (const lane of [0, 1]) {
            const c0 = s.coefficients(0, lane);
            expect(Math.abs(2 * c0.c)).toBeLessThan(1e-4); // S''(0) = 0
            const cl = s.coefficients(2, lane);
            expect(Math.abs(2 * cl.c + 6 * cl.d)).toBeLessThan(1e-4); // S''(end) = 0
            for (let i = 0; i < 2; i++) {
                const p = s.coefficients(i, lane);
                const q = s.coefficients(i + 1, lane);
                expect(p.b + 2 * p.c + 3 * p.d).toBeCloseTo(q.b, 4); // S' continuous
                expect(2 * p.c + 6 * p.d).toBeCloseTo(2 * q.c, 4); // S'' continuous
            }
        }
    });

    it("reproduces a straight line exactly", () => {
        const line = new CubicSpline([0, 1, 2, 3, 4], 5, 1);
        for (let sec = 0; sec < 4; sec++) for (const t of [0.25, 0.5, 0.9]) expect(line.interpolate(sec, t, 0)).toBeCloseTo(sec + t, 5);
    });
});
