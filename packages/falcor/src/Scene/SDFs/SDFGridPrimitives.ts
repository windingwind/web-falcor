/**
 * Runtime SDF primitive editing, mirroring the `SDFGrid` half of the SDFEditor:
 * `setPrimitives`/`addPrimitives`/`removePrimitives`/`updatePrimitives` plus the
 * bake bookkeeping, over the primitive list a `.sdf` file (or an app) supplies.
 *
 * Primitive *IDs* are stable handles; the list behind them stays compact, so
 * removing one shifts the indices of everything after it, exactly as upstream's
 * indirection map does.
 *
 * Web divergence (docs §9): upstream re-bakes incrementally on the GPU and only
 * SDFSBS accepts primitives, with baked ones folded into the stored field. The
 * port re-evaluates the whole list host-side into corner values (see
 * SDF3DPrimitive.ts) and hands them to any grid type, so baking here only fixes
 * which primitives may still be removed — the field is the same either way.
 */

import { Logger } from "../../Utils/Logger.js";
import { RuntimeError } from "../../Core/Error.js";
import { evaluateSDFPrimitives, type SDF3DPrimitive } from "./SDF3DPrimitive.js";

/** What this editor needs of a grid: somewhere to put the baked corner values. */
export interface SDFGridValueSink {
    setValues(cornerValues: Float32Array, gridWidth: number): void;
}

export class SDFGridPrimitives {
    private primitives: SDF3DPrimitive[] = [];
    /** Stable primitive ID -> index into the compact list. */
    private idToIndex = new Map<number, number>();
    private nextID = 0;
    private baked = 0;
    private dirty = false;

    constructor(
        private readonly grid: SDFGridValueSink,
        readonly gridWidth: number,
        primitives: readonly SDF3DPrimitive[] = [],
        /** Corner values the primitives edit (a grid loaded from values); empty space otherwise. */
        private readonly baseValues?: Float32Array,
    ) {
        if (primitives.length > 0) this.setPrimitives(primitives);
    }

    /** Mirrors SDFGrid::setPrimitives: replaces the list and restarts the IDs. */
    setPrimitives(primitives: readonly SDF3DPrimitive[]): number {
        this.primitives = [];
        this.idToIndex.clear();
        this.nextID = 0;
        this.baked = 0;
        return this.addPrimitives(primitives);
    }

    /** Mirrors SDFGrid::addPrimitives; returns the first new primitive's ID. */
    addPrimitives(primitives: readonly SDF3DPrimitive[]): number {
        const start = this.primitives.length;
        this.primitives.push(...primitives.map((p) => clonePrimitive(p)));
        const baseID = this.nextID;
        for (let index = start; index < this.primitives.length; index++) this.idToIndex.set(this.nextID++, index);
        this.dirty = true;
        return baseID;
    }

    /** Mirrors SDFGrid::removePrimitives: unknown and baked IDs warn and stay. */
    removePrimitives(primitiveIDs: readonly number[]): void {
        for (const primitiveID of primitiveIDs) {
            const index = this.idToIndex.get(primitiveID);
            if (index === undefined) {
                Logger.warning(`Primitive with ID ${primitiveID} does not exist!`);
                continue;
            }
            // Upstream marks the grid dirty before the baked check, so a refused
            // removal still triggers a rebuild.
            this.dirty = true;
            if (index < this.baked) {
                Logger.warning(`Primitive with ID ${primitiveID} has been baked, cannot remove it!`);
                continue;
            }
            this.idToIndex.delete(primitiveID);
            this.primitives.splice(index, 1);
            // The list stays compact, so later primitives move down one slot.
            for (const [id, other] of this.idToIndex) {
                if (other > index) this.idToIndex.set(id, other - 1);
            }
        }
    }

    /** Mirrors SDFGrid::updatePrimitives: replaces primitives in place by ID. */
    updatePrimitives(primitives: readonly (readonly [number, SDF3DPrimitive])[]): void {
        for (const [primitiveID, primitive] of primitives) {
            const index = this.idToIndex.get(primitiveID);
            if (index === undefined) {
                Logger.warning(`Primitive with ID ${primitiveID} does not exist!`);
                continue;
            }
            this.dirty = true;
            this.primitives[index] = clonePrimitive(primitive);
        }
    }

    /** Mirrors SDFGrid::getPrimitive. */
    getPrimitive(primitiveID: number): SDF3DPrimitive {
        const index = this.idToIndex.get(primitiveID);
        if (index === undefined) throw new RuntimeError(`SDFGrid.getPrimitive: 'primitiveID' (${primitiveID}) is invalid.`);
        return this.primitives[index]!;
    }

    /** The live primitive IDs, in list order. */
    getPrimitiveIDs(): number[] {
        return [...this.idToIndex.entries()].sort((a, b) => a[1] - b[1]).map(([id]) => id);
    }

    get primitiveCount(): number {
        return this.primitives.length;
    }

    get bakedPrimitiveCount(): number {
        return this.baked;
    }

    /** True when the grid's values no longer match the primitive list. */
    get isDirty(): boolean {
        return this.dirty;
    }

    /**
     * Mirrors SDFGrid::bakePrimitives: the first `batchSize` further primitives
     * become permanent. Upstream folds them into the stored field; here they
     * simply stop being removable (§9).
     */
    bakePrimitives(batchSize: number): void {
        this.baked = Math.min(this.baked + Math.max(0, Math.trunc(batchSize)), this.primitives.length);
    }

    /**
     * Re-evaluates the primitive list into the grid's corner values.
     *
     * @returns true if the grid was rebuilt (it was dirty).
     */
    rebuild(): boolean {
        if (!this.dirty) return false;
        this.grid.setValues(evaluateSDFPrimitives(this.primitives, this.gridWidth, this.baseValues), this.gridWidth);
        this.dirty = false;
        return true;
    }
}

/** Primitives are value types upstream; copying keeps callers from aliasing. */
function clonePrimitive(p: SDF3DPrimitive): SDF3DPrimitive {
    return { ...p, shapeData: [...p.shapeData], translation: [...p.translation], invRotationScale: [...p.invRotationScale] };
}
