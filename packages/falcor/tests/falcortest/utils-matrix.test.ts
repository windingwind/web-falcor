/**
 * Transplant of FalcorTest Utils/MatrixTests.cpp (float4x4 cases; the web has no 2x2/3x3/NxM matrices).
 */

import { describe, expect, it } from "vitest";
import {
    determinant,
    extractEulerAngleXYZ,
    float4x4,
    inverse,
    matrixFromLookAt,
    matrixFromRotationAxisAngle,
    matrixFromRotationXYZ,
    matrixFromScaling,
    matrixFromTranslation,
    mulMat,
    mulMatVec,
    ortho,
    perspective,
    transformPoint,
    transformVector,
    transpose,
} from "../../src/Utils/Math/Matrix.js";
import { matrixFromQuat, quatFromAngleAxis, quatf } from "../../src/Utils/Math/Quaternion.js";
import { float3, float4, normalize3 } from "../../src/Utils/Math/Vector.js";
import { decomposeTRS } from "../../src/Scene/Animation/SceneAnimation.js";

const radians = (deg: number) => (deg * Math.PI) / 180;

/** float4x4({...}) initializer-list constructor (row-major). */
const mat = (values: number[]) => new float4x4(Float32Array.from(values));
const row = (m: float4x4, r: number) => [m.get(r, 0), m.get(r, 1), m.get(r, 2), m.get(r, 3)];
const vec = (v: float3 | float4 | quatf) => ("w" in v ? [v.x, v.y, v.z, v.w] : [v.x, v.y, v.z]);

/** Native almostEqual, evaluated in fp32 like the float literals it compares against. */
function expectAlmostEq(a: number[], b: number[], epsilon = 1e-5): void {
    const f = Math.fround;
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) expect(f(Math.abs(f(a[i]!) - f(b[i]!))), `${a} != ${b}`).toBeLessThan(f(epsilon));
}

function expectRows(m: float4x4, rows: number[][], almost = true): void {
    for (let r = 0; r < 4; r++) {
        if (almost) expectAlmostEq(row(m, r), rows[r]!);
        else expect(row(m, r)).toEqual(rows[r]);
    }
}

const k1to16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const kTestM = [1, 0, 0, 10, 0, -1, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1];

describe("MatrixTests", () => {
    it("Matrix_Constructor", () => {
        // Native's default constructor is identity; web float4x4() is zeros, so only the named ones are ported.
        expectRows(mat(k1to16), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], false);
        expectRows(float4x4.identity(), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], false);
        expectRows(float4x4.zeros(), [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], false);
    });

    it("Matrix_multilply", () => {
        // Matrix/matrix multiplication
        const m3 = mulMat(mat(k1to16), mat(k1to16.map((v) => -v)));
        expectRows(m3, [[-90, -100, -110, -120], [-202, -228, -254, -280], [-314, -356, -398, -440], [-426, -484, -542, -600]], false);

        // Matrix/vector multiplication
        expect(vec(mulMatVec(mat(k1to16), new float4(1, 2, 3, 4)))).toEqual([30, 70, 110, 150]);
    });

    it("Matrix_transformPoint", () => {
        expect(vec(transformPoint(mat(k1to16), new float3(1, 2, 3)))).toEqual([18, 46, 74]);
    });

    it("Matrix_transformVector", () => {
        expect(vec(transformVector(mat(k1to16), new float3(1, 2, 3)))).toEqual([14, 38, 62]);
    });

    it("Matrix_transpose", () => {
        expectRows(transpose(mat(k1to16)), [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]], false);
    });

    // Native translate/rotate/scale(m, ...) equal mul(m, matrixFrom*(...)).
    it("Matrix_translate", () => {
        const m2 = mulMat(mat(kTestM), matrixFromTranslation(new float3(1, 2, 3)));
        expectRows(m2, [[1, 0, 0, 11], [0, -1, 0, 18], [0, 0, 1, 33], [0, 0, 0, 1]]);
    });

    it("Matrix_rotate", () => {
        const m2 = mulMat(mat(kTestM), matrixFromRotationAxisAngle(radians(90), new float3(0, 1, 0)));
        expectRows(m2, [[0, 0, 1, 10], [0, -1, 0, 20], [-1, 0, 0, 30], [0, 0, 0, 1]]);
    });

    it("Matrix_scale", () => {
        const m2 = mulMat(mat(kTestM), matrixFromScaling(new float3(2, 3, 4)));
        expectRows(m2, [[2, 0, 0, 10], [0, -3, 0, 20], [0, 0, 4, 30], [0, 0, 0, 1]]);
    });

    it("Matrix_determinant", () => {
        expect(determinant(mat(k1to16))).toBe(0);
        expect(determinant(mat([1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 12, 11, 15, 16, 13, 14]))).toBe(72);
    });

    it("Matrix_inverse", () => {
        const m = inverse(mat([1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 12, 11, 15, 16, 13, 14]));
        expectAlmostEq(row(m, 0), [1.125, 1.25, -0.5, -0.375]);
        expectAlmostEq(row(m, 1), [-1.652777, -1.527777, 0.5, 0.625]);
        expectAlmostEq(row(m, 2), [-0.625, -0.25, 0.5, -0.125]);
        expectAlmostEq(row(m, 3), [1.263888, 0.638888, -0.5, -0.125]);
    });

    it("Matrix_extractEulerAngleXYZ", () => {
        {
            const m = mat([0.5, -0.5, 0.707107, 0, 0.853553, 0.146446, -0.5, 0, 0.146446, 0.853553, 0.5, 0, 0, 0, 0, 1]);
            expectAlmostEq(vec(extractEulerAngleXYZ(m)), [radians(45), radians(45), radians(45)]);
        }
        {
            const m = mat([0.383022, -0.663414, 0.642787, 0, 0.92372, 0.279453, -0.262002, 0, -0.005813, 0.694109, 0.719846, 0, 0, 0, 0, 1]);
            expectAlmostEq(vec(extractEulerAngleXYZ(m)), [radians(20), radians(40), radians(60)]);
        }
    });

    // Web decomposeTRS covers scale/orientation/translation only (no skew/perspective, no failure result).
    it("Matrix_decompose", () => {
        const testDecompose = (m: float4x4, expectedScale: number[], expectedOrientation: number[], expectedTranslation: number[]) => {
            const { t, r, s } = decomposeTRS(m);
            expectAlmostEq(vec(s), expectedScale);
            expectAlmostEq(vec(r), expectedOrientation);
            expectAlmostEq(vec(t), expectedTranslation);
        };

        // Identity matrix
        testDecompose(float4x4.identity(), [1, 1, 1], [0, 0, 0, 1], [0, 0, 0]);
        // Scale only
        testDecompose(mat([2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1]), [2, 3, 4], [0, 0, 0, 1], [0, 0, 0]);
        // Orientation only
        testDecompose(mat([1, 0, 0, 0, 0, 0.707107, -0.707107, 0, 0, 0.707107, 0.707107, 0, 0, 0, 0, 1]), [1, 1, 1], [0.382683, 0, 0, 0.92388], [0, 0, 0]);
        // Translation only
        testDecompose(mat([1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1]), [1, 1, 1], [0, 0, 0, 1], [1, 2, 3]);
        // Affine transform
        testDecompose(mat([2, 0, 0, 1, 0, 2.12132, -2.82843, 2, 0, 2.12132, 2.82843, 3, 0, 0, 0, 1]), [2, 3, 4], [0.382683, 0, 0, 0.92388], [1, 2, 3]);
    });

    it("Matrix_matrixFromCoefficients", () => {
        expectRows(mat(k1to16), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], false);
    });

    it("Matrix_perspective", () => {
        const m = perspective(radians(45), 2, 0.1, 1000);
        expectRows(m, [[1.207107, 0, 0, 0], [0, 2.414213, 0, 0], [0, 0, -1.0001, -0.1], [0, 0, -1, 0]]);
    });

    it("Matrix_ortho", () => {
        const m = ortho(-10, 10, -10, 10, 0.1, 1000);
        expectRows(m, [[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, -0.001, -0.0001], [0, 0, 0, 1]]);
    });

    it("Matrix_matrixFromTranslation", () => {
        expectRows(matrixFromTranslation(new float3(1, 2, 3)), [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], false);
    });

    it("Matrix_matrixFromRotation", () => {
        const rot = (deg: number, axis: float3) => matrixFromRotationAxisAngle(radians(deg), axis);
        const s = 0.707106;
        expectRows(rot(90, new float3(1, 0, 0)), [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]);
        expectRows(rot(-45, new float3(1, 0, 0)), [[1, 0, 0, 0], [0, s, s, 0], [0, -s, s, 0], [0, 0, 0, 1]]);
        expectRows(rot(90, new float3(0, 1, 0)), [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]);
        expectRows(rot(-45, new float3(0, 1, 0)), [[s, 0, -s, 0], [0, 1, 0, 0], [s, 0, s, 0], [0, 0, 0, 1]]);
        expectRows(rot(90, new float3(0, 0, 1)), [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        expectRows(rot(-45, new float3(0, 0, 1)), [[s, s, 0, 0], [-s, s, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        expectRows(rot(60, normalize3(new float3(1, 1, 1))), [
            [0.666666, -0.333333, 0.666666, 0],
            [0.666666, 0.666666, -0.333333, 0],
            [-0.333333, 0.666666, 0.666666, 0],
            [0, 0, 0, 1],
        ]);
    });

    // The web has only matrixFromRotationXYZ (no single-axis matrixFromRotationX/Y/Z).
    it("Matrix_matrixFromRotationXYZ", () => {
        const s = 0.707106;
        expectRows(matrixFromRotationXYZ(radians(90), 0, 0), [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]);
        expectRows(matrixFromRotationXYZ(radians(-45), 0, 0), [[1, 0, 0, 0], [0, s, s, 0], [0, -s, s, 0], [0, 0, 0, 1]]);
        expectRows(matrixFromRotationXYZ(0, radians(90), 0), [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]);
        expectRows(matrixFromRotationXYZ(0, radians(-45), 0), [[s, 0, -s, 0], [0, 1, 0, 0], [s, 0, s, 0], [0, 0, 0, 1]]);
        expectRows(matrixFromRotationXYZ(0, 0, radians(90)), [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        expectRows(matrixFromRotationXYZ(0, 0, radians(-45)), [[s, s, 0, 0], [-s, s, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        expectRows(matrixFromRotationXYZ(radians(45), radians(45), radians(45)), [
            [0.5, -0.5, 0.707107, 0],
            [0.853553, 0.146446, -0.5, 0],
            [0.146446, 0.853553, 0.5, 0],
            [0, 0, 0, 1],
        ]);
        expectRows(matrixFromRotationXYZ(radians(20), radians(40), radians(60)), [
            [0.383022, -0.663414, 0.642787, 0],
            [0.92372, 0.279453, -0.262002, 0],
            [-0.005813, 0.694109, 0.719846, 0],
            [0, 0, 0, 1],
        ]);
    });

    it("Matrix_matrixFromScaling", () => {
        expectRows(matrixFromScaling(new float3(2, 3, 4)), [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]]);
    });

    // Right-handed only: the web matrixFromLookAt has no handedness parameter.
    it("Matrix_matrixFromLookAt", () => {
        const m = matrixFromLookAt(new float3(10, 5, 0), new float3(0, -5, 0), new float3(0, 1, 0));
        expectRows(m, [[0, 0, -1, 0], [-0.707107, 0.707107, 0, 3.535535], [0.707107, 0.707107, 0, -10.606603], [0, 0, 0, 1]]);
    });

    it("Matrix_matrixFromQuat", () => {
        expectRows(matrixFromQuat(quatf.identity()), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        const q = quatFromAngleAxis(radians(60), normalize3(new float3(1, 1, 1)));
        expectRows(matrixFromQuat(q), [
            [0.666666, -0.333333, 0.666666, 0],
            [0.666666, 0.666666, -0.333333, 0],
            [-0.333333, 0.666666, 0.666666, 0],
            [0, 0, 0, 1],
        ]);
    });
});
