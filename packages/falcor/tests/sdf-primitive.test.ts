/**
 * `.sdf` primitive lists (SDFGrid::loadPrimitivesFromFile) and the primitive
 * evaluation that turns them into grid corner values.
 *
 * Ground truth is closed-form: a sphere's signed distance is |p - c| - r, a box's
 * is the standard exterior/interior split, and the CSG operations are min/max.
 */

import { describe, it, expect } from "vitest";
import {
    SDF3DShapeType,
    SDFOperationType,
    evalSDFPrimitive,
    evaluateSDFPrimitives,
    parseSDFPrimitives,
    serializeSDFPrimitives,
    type SDF3DPrimitive,
} from "../src/Scene/SDFs/SDF3DPrimitive.js";

const kIdentity = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function sphere(radius: number, translation: [number, number, number] = [0, 0, 0], operationType = SDFOperationType.Union): SDF3DPrimitive {
    return {
        shapeType: SDF3DShapeType.Sphere,
        shapeData: [radius, 0, 0],
        shapeBlobbing: 0,
        operationType,
        operationSmoothing: 0,
        translation,
        invRotationScale: kIdentity,
    };
}

describe("SDF3DPrimitive parsing", () => {
    it("reads the JSON keys SDFGrid.cpp writes", () => {
        const text = JSON.stringify([
            {
                shape_type: "box",
                shape_data: [0.4, 0.1, 0.1],
                shape_blobbing: 0.02,
                operation_type: "smooth_union",
                operation_smoothing: 0.05,
                translation: [1, 2, 3],
                inv_rot_scale: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            },
        ]);
        const [p] = parseSDFPrimitives(text);
        expect(p!.shapeType).toBe(SDF3DShapeType.Box);
        expect(p!.operationType).toBe(SDFOperationType.SmoothUnion);
        expect(p!.shapeData).toEqual([0.4, 0.1, 0.1]);
        expect(p!.shapeBlobbing).toBeCloseTo(0.02, 10);
        expect(p!.operationSmoothing).toBeCloseTo(0.05, 10);
        expect(p!.translation).toEqual([1, 2, 3]);
    });

    it("accepts the legacy numeric enum spelling", () => {
        const text = JSON.stringify([
            { shape_type: 3, shape_data: [0.2, 0, 0], shape_blobbing: 0, operation_type: 2, operation_smoothing: 0, translation: [0, 0, 0], inv_rot_scale: kIdentity },
        ]);
        const [p] = parseSDFPrimitives(text);
        expect(p!.shapeType).toBe(SDF3DShapeType.Torus);
        expect(p!.operationType).toBe(SDFOperationType.Intersection);
    });

    it("round-trips through the writer", () => {
        const primitives = [sphere(0.3, [0.1, 0, -0.2]), { ...sphere(0.1), shapeType: SDF3DShapeType.Capsule, operationType: SDFOperationType.Subtraction }];
        expect(parseSDFPrimitives(serializeSDFPrimitives(primitives))).toEqual(primitives);
    });

    it("rejects malformed entries", () => {
        expect(() => parseSDFPrimitives("{}")).toThrow(/array of primitives/);
        expect(() => parseSDFPrimitives(JSON.stringify([{ shape_type: "blob", shape_data: [0, 0, 0], translation: [0, 0, 0], inv_rot_scale: kIdentity }]))).toThrow(/unknown shape type/);
    });
});

describe("SDF3DPrimitive evaluation", () => {
    const kFltMax = 3.402823466e38;

    it("matches the closed-form sphere distance", () => {
        const p = sphere(0.3, [0.1, -0.2, 0.05]);
        for (const point of [
            [0, 0, 0],
            [0.4, 0.1, -0.3],
            [0.1, -0.2, 0.05],
        ] as [number, number, number][]) {
            const expected = Math.hypot(point[0] - 0.1, point[1] + 0.2, point[2] - 0.05) - 0.3;
            expect(evalSDFPrimitive(p, point, kFltMax)).toBeCloseTo(expected, 12);
        }
    });

    it("folds shapes together with min/max", () => {
        const a = sphere(0.3, [-0.2, 0, 0]);
        const b = sphere(0.3, [0.2, 0, 0]);
        const point: [number, number, number] = [0, 0.1, 0];
        const da = evalSDFPrimitive(a, point, kFltMax);
        const db = evalSDFPrimitive(b, point, kFltMax);
        expect(evalSDFPrimitive(b, point, da)).toBeCloseTo(Math.min(da, db), 12);
        const subtract = { ...b, operationType: SDFOperationType.Subtraction };
        expect(evalSDFPrimitive(subtract, point, da)).toBeCloseTo(Math.max(da, -db), 12);
    });

    it("applies the shader's transposed inverse rotation", () => {
        // A bar rotated 45° about z. Native evaluates transpose(invRotationScale)
        // * (p - translation), which puts the long axis on the +45° diagonal;
        // dropping the transpose would put it on the other one.
        const c = Math.SQRT1_2;
        const bar: SDF3DPrimitive = {
            shapeType: SDF3DShapeType.Box,
            shapeData: [0.42, 0.08, 0.08],
            shapeBlobbing: 0,
            operationType: SDFOperationType.Union,
            operationSmoothing: 0,
            translation: [0, 0, 0],
            invRotationScale: [c, -c, 0, c, c, 0, 0, 0, 1],
        };
        expect(evalSDFPrimitive(bar, [0.25, 0.25, 0], kFltMax)).toBeLessThan(0);
        expect(evalSDFPrimitive(bar, [0.25, -0.25, 0], kFltMax)).toBeGreaterThan(0);
    });

    it("blobbing rounds the shape outwards", () => {
        const plain = sphere(0.3);
        const blobbed = { ...plain, shapeBlobbing: 0.05 };
        expect(evalSDFPrimitive(blobbed, [0.4, 0, 0], kFltMax)).toBeCloseTo(evalSDFPrimitive(plain, [0.4, 0, 0], kFltMax) - 0.05, 12);
    });

    it("bakes corner values at the kernel's sample positions, x fastest", () => {
        const gridWidth = 4;
        const primitives = [sphere(0.3)];
        const values = evaluateSDFPrimitives(primitives, gridWidth);
        const w = gridWidth + 1;
        expect(values.length).toBe(w * w * w);
        for (const [x, y, z] of [
            [0, 0, 0],
            [4, 0, 0],
            [2, 2, 2],
            [1, 3, 4],
        ]) {
            const p: [number, number, number] = [-0.5 + x! / gridWidth, -0.5 + y! / gridWidth, -0.5 + z! / gridWidth];
            expect(values[x! + w * (y! + w * z!)]).toBeCloseTo(Math.hypot(p[0], p[1], p[2]) - 0.3, 6);
        }
        // The centre corner is the deepest point inside the sphere.
        expect(values[2 + w * (2 + w * 2)]).toBeCloseTo(-0.3, 6);
    });
});
