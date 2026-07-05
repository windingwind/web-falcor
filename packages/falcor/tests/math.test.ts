/**
 * Math library unit tests: verifies Falcor conventions (row-major, column
 * vectors, RH lookAt, D3D depth range).
 */

import { describe, it, expect } from "vitest";
import { float3, float4, cross, dot3, normalize3 } from "../src/Utils/Math/Vector.js";
import {
    float4x4,
    mulMat,
    mulMatVec,
    transformPoint,
    inverse,
    matrixFromLookAt,
    matrixFromTranslation,
    perspective,
    transpose,
} from "../src/Utils/Math/Matrix.js";
import { quatFromAngleAxis, matrixFromQuat, rotateVector, mulQuat } from "../src/Utils/Math/Quaternion.js";
import { DxSamplePattern, HaltonSamplePattern, StratifiedSamplePattern } from "../src/Utils/SampleGenerators/CPUSampleGenerator.js";

const closeTo = (a: number, b: number, eps = 1e-5) => expect(Math.abs(a - b)).toBeLessThan(eps);

describe("Vector", () => {
    it("cross product follows right-hand rule", () => {
        const c = cross(new float3(1, 0, 0), new float3(0, 1, 0));
        expect(c.toArray()).toEqual([0, 0, 1]);
    });
    it("normalize", () => {
        const n = normalize3(new float3(3, 0, 4));
        closeTo(n.x, 0.6);
        closeTo(n.z, 0.8);
    });
});

describe("Matrix (Falcor conventions)", () => {
    it("translation lives in column 3, applied via mul(M, v)", () => {
        const m = matrixFromTranslation(new float3(10, 20, 30));
        const p = transformPoint(m, new float3(1, 2, 3));
        expect(p.toArray()).toEqual([11, 22, 33]);
    });

    it("mulMat composes left-to-right for column vectors", () => {
        const t1 = matrixFromTranslation(new float3(1, 0, 0));
        const t2 = matrixFromTranslation(new float3(0, 2, 0));
        const p = transformPoint(mulMat(t2, t1), new float3(0, 0, 0));
        expect(p.toArray()).toEqual([1, 2, 0]);
    });

    it("lookAt maps eye to origin and view direction to -Z (RH)", () => {
        const eye = new float3(0, 0, 5);
        const m = matrixFromLookAt(eye, new float3(0, 0, 0), new float3(0, 1, 0));
        const pEye = transformPoint(m, eye);
        closeTo(pEye.x, 0); closeTo(pEye.y, 0); closeTo(pEye.z, 0);
        // A point in front of the camera lands at negative Z in view space.
        const pFront = transformPoint(m, new float3(0, 0, 0));
        expect(pFront.z).toBeLessThan(0);
    });

    it("perspective maps near to depth 0 and far to depth 1 (D3D range)", () => {
        const proj = perspective(Math.PI / 2, 1, 0.1, 100);
        const near = mulMatVec(proj, new float4(0, 0, -0.1, 1));
        const far = mulMatVec(proj, new float4(0, 0, -100, 1));
        closeTo(near.z / near.w, 0, 1e-5);
        closeTo(far.z / far.w, 1, 1e-4);
    });

    it("inverse round-trips", () => {
        const m = mulMat(matrixFromTranslation(new float3(1, 2, 3)), perspective(1, 1.5, 0.5, 50));
        const r = mulMat(m, inverse(m));
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) closeTo(r.get(i, j), i === j ? 1 : 0, 1e-4);
    });

    it("transpose", () => {
        const m = float4x4.fromRows([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        expect(transpose(m).get(0, 1)).toBe(5);
        expect(m.toColumnMajorArray()[1]).toBe(5);
    });
});

describe("Quaternion", () => {
    it("90deg rotation about Y maps +X to -Z", () => {
        const q = quatFromAngleAxis(Math.PI / 2, new float3(0, 1, 0));
        const v = rotateVector(q, new float3(1, 0, 0));
        closeTo(v.x, 0); closeTo(v.y, 0); closeTo(v.z, -1);
    });
    it("matrixFromQuat agrees with rotateVector", () => {
        const q = quatFromAngleAxis(0.7, normalize3(new float3(1, 2, 3)));
        const v = new float3(0.3, -0.5, 0.9);
        const a = rotateVector(q, v);
        const b = transformPoint(matrixFromQuat(q), v);
        closeTo(a.x, b.x); closeTo(a.y, b.y); closeTo(a.z, b.z);
    });
    it("quaternion multiplication composes rotations", () => {
        const qx = quatFromAngleAxis(Math.PI / 2, new float3(1, 0, 0));
        const qy = quatFromAngleAxis(Math.PI / 2, new float3(0, 1, 0));
        const v = rotateVector(mulQuat(qy, qx), new float3(0, 0, 1));
        const v2 = rotateVector(qy, rotateVector(qx, new float3(0, 0, 1)));
        closeTo(v.x, v2.x); closeTo(v.y, v2.y); closeTo(v.z, v2.z);
    });
});

describe("Sample patterns", () => {
    it("DxSamplePattern cycles the 8x table", () => {
        const p = new DxSamplePattern();
        expect(p.getSampleCount()).toBe(8);
        const first = p.next();
        closeTo(first.x, 1 / 16);
        closeTo(first.y, -3 / 16);
        for (let i = 0; i < 7; i++) p.next();
        const wrapped = p.next();
        closeTo(wrapped.x, 1 / 16);
    });

    it("HaltonSamplePattern stays in [-0.5, 0.5) and starts at origin", () => {
        const p = new HaltonSamplePattern(16);
        const first = p.next();
        closeTo(first.x, 0); closeTo(first.y, 0);
        for (let i = 0; i < 64; i++) {
            const s = p.next();
            expect(s.x).toBeGreaterThanOrEqual(-0.5);
            expect(s.x).toBeLessThan(0.5);
            expect(s.y).toBeGreaterThanOrEqual(-0.5);
            expect(s.y).toBeLessThan(0.5);
        }
    });

    it("StratifiedSamplePattern factors count into a grid and covers bins", () => {
        const p = new StratifiedSamplePattern(16);
        expect(p.getSampleCount()).toBe(16);
        const seen = new Set<string>();
        for (let i = 0; i < 16; i++) {
            const s = p.next();
            const bx = Math.floor((s.x + 0.5) * 4);
            const by = Math.floor((s.y + 0.5) * 4);
            seen.add(`${bx},${by}`);
        }
        expect(seen.size).toBe(16); // each bin hit exactly once per round
    });
});
