/**
 * Runtime SDF primitive editing (SDFGrid::add/remove/updatePrimitives + bake).
 *
 * The bookkeeping is what has to be exact: IDs are stable handles over a list
 * that stays compact, so a removal shifts every later primitive down one slot
 * while its ID keeps pointing at the same primitive.
 */

import { describe, it, expect, vi } from "vitest";
import { SDF3DShapeType, SDFOperationType, type SDF3DPrimitive } from "../src/Scene/SDFs/SDF3DPrimitive.js";
import { SDFGridPrimitives } from "../src/Scene/SDFs/SDFGridPrimitives.js";

/** Records what the grid would have been given. */
class FakeGrid {
    calls: { values: Float32Array; gridWidth: number }[] = [];
    setValues(values: Float32Array, gridWidth: number): void {
        this.calls.push({ values, gridWidth });
    }
}

function sphere(radius: number): SDF3DPrimitive {
    return {
        shapeType: SDF3DShapeType.Sphere,
        shapeData: [radius, 0, 0],
        shapeBlobbing: 0,
        operationType: SDFOperationType.Union,
        operationSmoothing: 0,
        translation: [0, 0, 0],
        invRotationScale: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    };
}

describe("SDF primitive editing", () => {
    it("assigns stable ids and rebuilds on demand", () => {
        const grid = new FakeGrid();
        const set = new SDFGridPrimitives(grid, 8, [sphere(0.1)]);
        expect(set.primitiveCount).toBe(1);
        expect(set.isDirty).toBe(true);

        expect(set.rebuild()).toBe(true);
        expect(set.isDirty).toBe(false);
        expect(grid.calls.length).toBe(1);
        expect(grid.calls[0]!.gridWidth).toBe(8);
        expect(grid.calls[0]!.values.length).toBe(9 * 9 * 9);
        // A clean set does not touch the grid again.
        expect(set.rebuild()).toBe(false);
        expect(grid.calls.length).toBe(1);

        const base = set.addPrimitives([sphere(0.2), sphere(0.3)]);
        expect(base).toBe(1); // the first set took id 0
        expect(set.primitiveCount).toBe(3);
        expect(set.getPrimitive(2).shapeData[0]).toBeCloseTo(0.3, 10);
        expect(set.isDirty).toBe(true);
    });

    it("keeps the list compact when a primitive is removed", () => {
        const set = new SDFGridPrimitives(new FakeGrid(), 8, [sphere(0.1), sphere(0.2), sphere(0.3)]);
        set.removePrimitives([1]);
        expect(set.primitiveCount).toBe(2);
        // The surviving ids still resolve to their own primitives.
        expect(set.getPrimitive(0).shapeData[0]).toBeCloseTo(0.1, 10);
        expect(set.getPrimitive(2).shapeData[0]).toBeCloseTo(0.3, 10);
        expect(set.getPrimitiveIDs()).toEqual([0, 2]);
        expect(() => set.getPrimitive(1)).toThrow(/invalid/);

        // Ids keep counting up; they are not reused.
        expect(set.addPrimitives([sphere(0.4)])).toBe(3);
        expect(set.getPrimitive(3).shapeData[0]).toBeCloseTo(0.4, 10);
    });

    it("warns instead of throwing on unknown ids", () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const set = new SDFGridPrimitives(new FakeGrid(), 8, [sphere(0.1)]);
        set.removePrimitives([42]);
        set.updatePrimitives([[42, sphere(0.9)]]);
        expect(set.primitiveCount).toBe(1);
        expect(set.getPrimitive(0).shapeData[0]).toBeCloseTo(0.1, 10);
        warn.mockRestore();
    });

    it("refuses to remove baked primitives", () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const set = new SDFGridPrimitives(new FakeGrid(), 8, [sphere(0.1), sphere(0.2), sphere(0.3)]);
        set.bakePrimitives(2);
        expect(set.bakedPrimitiveCount).toBe(2);
        set.removePrimitives([0, 2]);
        // Id 0 is baked and stays; id 2 is not and goes.
        expect(set.getPrimitiveIDs()).toEqual([0, 1]);
        // Baking more than there is clamps to the list length.
        set.bakePrimitives(99);
        expect(set.bakedPrimitiveCount).toBe(2);
        warn.mockRestore();
    });

    it("replaces primitives in place and re-bakes the values", () => {
        const grid = new FakeGrid();
        const set = new SDFGridPrimitives(grid, 4, [sphere(0.1)]);
        set.rebuild();
        const before = grid.calls[0]!.values[0]!;

        set.updatePrimitives([[0, sphere(0.45)]]);
        expect(set.isDirty).toBe(true);
        expect(set.rebuild()).toBe(true);
        // The corner sits at (-0.5, -0.5, -0.5), so a bigger sphere is nearer.
        expect(grid.calls[1]!.values[0]!).toBeCloseTo(before - 0.35, 5);
    });

    it("restarts the ids when the whole list is replaced", () => {
        const set = new SDFGridPrimitives(new FakeGrid(), 8, [sphere(0.1), sphere(0.2)]);
        set.bakePrimitives(1);
        expect(set.setPrimitives([sphere(0.5)])).toBe(0);
        expect(set.primitiveCount).toBe(1);
        expect(set.bakedPrimitiveCount).toBe(0);
        expect(set.getPrimitiveIDs()).toEqual([0]);
    });

    it("copies primitives so callers cannot alias them", () => {
        const source = sphere(0.1);
        const set = new SDFGridPrimitives(new FakeGrid(), 8, [source]);
        source.shapeData[0] = 99;
        expect(set.getPrimitive(0).shapeData[0]).toBeCloseTo(0.1, 10);
    });
});
