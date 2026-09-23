/**
 * Transplant of FalcorTest Utils/QuaternionTests.cpp (the subset the web quatf API covers).
 */

import { describe, expect, it } from "vitest";
import { mulQuat, normalizeQuat, quatFromAngleAxis, quatFromRotationBetweenVectors, quatf, rotateVector, slerp } from "../../src/Utils/Math/Quaternion.js";
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

    it("Quaternion_Functions", () => {
        // normalize (dot/cross/length/conjugate/inverse/lerp have no web equivalent)
        const q2 = normalizeQuat(new quatf(1, 2, 3, 4));
        const s = 1 / Math.sqrt(30);
        expect(almostEqual(q4(q2), [s, 2 * s, 3 * s, 4 * s], 1e-7)).toBe(true);
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
});
