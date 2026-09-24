/**
 * Transplant of FalcorTest Utils/QuaternionTests.cpp. Quaternion_Access and Quaternion_Operator test
 * C++ subscript and arithmetic operators, which TypeScript has no counterpart for.
 */

import { describe, expect, it } from "vitest";
import { conjugateQuat, crossQuat, dotQuat, eulerAngles, inverseQuat, isfiniteQuat, isinfQuat, isnanQuat, lengthQuat, lerpQuat, matrixFromQuat, mulQuat, normalizeQuat, pitch, quatFromAngleAxis, quatFromEulerAngles, quatFromMatrix, quatFromRotationBetweenVectors, quatf, roll, rotateVector, slerp, yaw } from "../../src/Utils/Math/Quaternion.js";
import { float3, normalize3 } from "../../src/Utils/Math/Vector.js";

const radians = (deg: number) => (deg * Math.PI) / 180;
const q4 = (q: quatf) => [q.x, q.y, q.z, q.w];
const v3 = (v: float3) => [v.x, v.y, v.z];

function almostEqual(a: number[], b: number[], epsilon = 1e-5): boolean {
    return a.every((v, i) => Math.abs(v - b[i]!) < epsilon);
}

/** Native almostEqualOrientation: q and -q are the same rotation. */
function expectAlmostEqOrientation(a: quatf, b: quatf): void {
    const d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    const bb = d < 0 ? q4(b).map((v) => -v) : q4(b);
    expect(almostEqual(q4(a), bb), `${q4(a)} != ${q4(b)}`).toBe(true);
}

describe("QuaternionTests", () => {
    it("Quaternion_Constructor", () => {
        expect(q4(new quatf())).toEqual([0, 0, 0, 1]);
        expect(q4(new quatf(1, 2, 3, 4))).toEqual([1, 2, 3, 4]);
        expect(q4(quatf.identity())).toEqual([0, 0, 0, 1]);
    });

    it("Quaternion_Multiply", () => {
        expect(q4(mulQuat(new quatf(1, 2, 3, 4), new quatf(2, 3, 4, 5)))).toEqual([12, 24, 30, 0]);
        expect(v3(rotateVector(new quatf(2, 3, 4, 5), new float3(2, 3, 4)))).toEqual([2, 3, 4]);
    });

    it("Quaternion_FloatChecks", () => {
        const zero = new quatf(0, 0, 0, 0);
        const inf = new quatf(Infinity, 0, 0, Infinity);
        const nan = new quatf(NaN, 0, 0, NaN);
        expect(isfiniteQuat(zero)).toEqual([true, true, true, true]);
        expect(isfiniteQuat(inf)).toEqual([false, true, true, false]);
        expect(isinfQuat(zero)).toEqual([false, false, false, false]);
        expect(isinfQuat(inf)).toEqual([true, false, false, true]);
        expect(isnanQuat(zero)).toEqual([false, false, false, false]);
        expect(isnanQuat(nan)).toEqual([true, false, false, true]);
    });

    it("Quaternion_Functions", () => {
        const q1 = new quatf(1, 2, 3, 4);
        const q2 = new quatf(2, 3, 4, 5);
        expect(dotQuat(q1, q2)).toBe(40);
        expect(q4(crossQuat(q1, q2))).toEqual([12, 24, 30, 0]);
        expect(lengthQuat(q1)).toBeCloseTo(Math.sqrt(30), 6);
        const s = 1 / Math.sqrt(30);
        expect(almostEqual(q4(normalizeQuat(q1)), [s, 2 * s, 3 * s, 4 * s], 1e-7)).toBe(true);
        expect(q4(conjugateQuat(q1))).toEqual([-1, -2, -3, 4]);
        expect(almostEqual(q4(inverseQuat(q1)), [-1 / 30, -2 / 30, -3 / 30, 4 / 30], 1e-7)).toBe(true);
        expect(q4(lerpQuat(q1, q2, 0.5))).toEqual([1.5, 2.5, 3.5, 4.5]);
    });

    it("Quaternion_slerp", () => {
        const test = (angle: number, t: number, expectedAngle: number) => {
            const from = quatf.identity();
            const to = quatFromAngleAxis(radians(angle), new float3(0, 1, 0));
            const expected = quatFromAngleAxis(radians(expectedAngle), new float3(0, 1, 0));
            expectAlmostEqOrientation(slerp(from, to, t), expected);
            expectAlmostEqOrientation(slerp(to, from, 1 - t), expected);
        };

        // Basic
        test(+160, 0.375, +60);
        test(-160, 0.375, -60);

        // Shorting
        test(+320, 0.375, -15);
        test(-320, 0.375, +15);

        // Lengthening short way
        test(320, 1.5, -60);

        // Lengthening
        test(+70, 3, +210);
        test(-70, 3, -210);

        // Edge case that often causes NaNs
        test(0, 0.5, 0);

        test(360, 0.25, 0);
    });

    it("Quaternion_quatFromAngleAxis", () => {
        const h = Math.sqrt(0.5);
        const t = Math.sqrt(0.75);
        const check = (deg: number, axis: float3, expected: number[]) => expect(almostEqual(q4(quatFromAngleAxis(radians(deg), axis)), expected)).toBe(true);
        check(90, new float3(1, 0, 0), [h, 0, 0, h]);
        check(-60, new float3(1, 0, 0), [-0.5, 0, 0, t]);
        check(90, new float3(0, 1, 0), [0, h, 0, h]);
        check(-60, new float3(0, 1, 0), [0, -0.5, 0, t]);
        check(90, new float3(0, 0, 1), [0, 0, h, h]);
        check(-60, new float3(0, 0, 1), [0, 0, -0.5, t]);
    });

    it("Quaternion_quatFromRotationBetweenVectors", () => {
        const test = (v1: float3, axis: float3, angle: number) => {
            const q1 = quatFromAngleAxis(radians(angle), axis);
            const v2 = rotateVector(q1, v1);
            const q2 = quatFromRotationBetweenVectors(v1, v2);
            const v3_ = rotateVector(q2, v1);
            expect(almostEqual(v3(v2), v3(v3_)), `${v3(v2)} != ${v3(v3_)}`).toBe(true);
        };

        test(new float3(0, 1, 0), new float3(1, 0, 0), 90);
        test(new float3(0, 0, 1), new float3(1, 0, 0), -135);

        test(new float3(1, 0, 0), new float3(0, 1, 0), 90);
        test(new float3(0, 0, 1), new float3(0, 1, 0), -135);

        test(new float3(1, 0, 0), new float3(0, 0, 1), 90);
        test(new float3(0, 1, 0), new float3(0, 0, 1), -135);

        test(new float3(1, 0, 0), normalize3(new float3(1, 1, 1)), 45);
        test(new float3(0, 1, 0), normalize3(new float3(1, 1, 1)), 90);
        test(new float3(0, 0, 1), normalize3(new float3(1, 1, 1)), 135);
    });

    it("Quaternion_Euler", () => {
        const [h, r] = [Math.sqrt(0.5), Math.sqrt(0.75)];
        const near = (a: number, b: number, eps = 1e-5) => expect(Math.abs(a - b) < eps, `${a} != ${b}`).toBe(true);
        near(pitch(new quatf(h, 0, 0, h)), radians(90));
        near(pitch(new quatf(-0.5, 0, 0, r)), radians(-60));
        near(yaw(new quatf(0, h, 0, h)), radians(90), 1e-3);
        near(yaw(new quatf(0, -0.5, 0, r)), radians(-60));
        near(roll(new quatf(0, 0, h, h)), radians(90));
        near(roll(new quatf(0, 0, -0.5, r)), radians(-60));
        expect(almostEqual(v3(eulerAngles(new quatf(-0.5, 0, 0, r))), [radians(-60), 0, 0])).toBe(true);
        expect(almostEqual(v3(eulerAngles(new quatf(0, -0.5, 0, r))), [0, radians(-60), 0])).toBe(true);
        expect(almostEqual(v3(eulerAngles(new quatf(0, 0, -0.5, r))), [0, 0, radians(-60)])).toBe(true);
        const q4v = new quatf(-0.591506362, -0.158493653, -0.591506362, 0.524519026);
        expect(almostEqual(v3(eulerAngles(q4v)), [radians(-60), radians(-60), radians(-60)])).toBe(true);
    });

    it("Quaternion_quatFromEulerAngles", () => {
        const [h, r] = [Math.sqrt(0.5), Math.sqrt(0.75)];
        const cases: [number[], number[]][] = [
            [[90, 0, 0], [h, 0, 0, h]],
            [[-60, 0, 0], [-0.5, 0, 0, r]],
            [[0, 90, 0], [0, h, 0, h]],
            [[0, -60, 0], [0, -0.5, 0, r]],
            [[0, 0, 90], [0, 0, h, h]],
            [[0, 0, -60], [0, 0, -0.5, r]],
        ];
        for (const [deg, want] of cases) expect(almostEqual(q4(quatFromEulerAngles(new float3(...(deg.map(radians) as [number, number, number])))), want), `${deg}`).toBe(true);
    });

    it("Quaternion_quatFromMatrix", () => {
        const test = (x: number, y: number, z: number) => {
            const e = new float3(radians(x), radians(y), radians(z));
            const q1 = quatFromEulerAngles(e);
            const q2 = quatFromMatrix(matrixFromQuat(q1));
            expect(almostEqual(q4(q1), q4(q2)), `${[x, y, z]}: ${q4(q1)} != ${q4(q2)}`).toBe(true);
            expect(almostEqual(v3(e), v3(eulerAngles(q2)), 1e-3), `${[x, y, z]} euler`).toBe(true);
        };
        test(0, 0, 0);
        test(90, 0, 0);
        test(0, 90, 0);
        test(0, 0, 90);
        test(-45, 0, 0);
        test(0, -45, 0);
        test(0, 0, -45);
        test(10, 20, 30);
        test(-30, -20, -10);
    });
});
