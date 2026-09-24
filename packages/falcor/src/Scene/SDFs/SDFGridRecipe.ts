/**
 * Recorded pyscene SDF-grid construction (type + generator calls). Grids are
 * rebuilt deterministically from it (seeded mt19937), which is what the scene
 * cache stores instead of the built voxel data.
 */

import { NDSDFGrid } from "./NDSDFGrid.js";
import { SDFSBS } from "./SDFSBS.js";
import { SDFSVS } from "./SDFSVS.js";
import { SDFSVO } from "./SDFSVO.js";
import { RuntimeError } from "../../Core/Error.js";
import { type SDF3DPrimitive } from "./SDF3DPrimitive.js";
import { SDFGridPrimitives } from "./SDFGridPrimitives.js";

export type SDFGridType = "ndsdf" | "sbs" | "svs" | "svo";

export interface SDFGridRecipe {
    type: SDFGridType;
    narrowBandThickness: number;
    brickWidth: number;
    /** SBS: BC4-compressed bricks (createSBS(compressed=True)). */
    compressed?: boolean;
    /** SBS: grid width for an SBS edited without values (createSBS(defaultGridWidth=...)). */
    defaultGridWidth?: number;
    ops: (
        | { kind: "cheese"; gridWidth: number; seed: number }
        | { kind: "values"; gridWidth: number; values: Float32Array }
        | { kind: "primitives"; gridWidth: number; primitives: SDF3DPrimitive[] }
    )[];
}

export type BuiltSDFGrid = NDSDFGrid | SDFSBS | SDFSVS | SDFSVO;

/** Mirrors the SceneBuilder bridge's SDFGrid.create*() + generateCheeseValues() sequence. */
export function buildSDFGridFromRecipe(recipe: SDFGridRecipe): BuiltSDFGrid {
    const built: BuiltSDFGrid =
        recipe.type === "sbs" ? new SDFSBS(recipe.brickWidth, recipe.compressed ?? false, recipe.defaultGridWidth ?? 256)
        : recipe.type === "svs" ? new SDFSVS()
        : recipe.type === "svo" ? new SDFSVO()
        : new NDSDFGrid(recipe.narrowBandThickness);
    for (const op of recipe.ops) {
        if (op.kind === "cheese") built.generateCheeseValues(op.gridWidth, op.seed);
        else if (op.kind === "values") built.setValues(op.values, op.gridWidth);
        // Primitives bake to corner values here; native evaluates the same loop
        // on the GPU (EvaluateSDFPrimitives.cs.slang), see SDF3DPrimitive.ts.
        // The list stays on the grid so it can be edited afterwards.
        else if (op.kind === "primitives") {
            built.primitives = new SDFGridPrimitives(built, op.gridWidth, op.primitives);
            built.primitives.rebuild();
        }
    }
    const ok = built instanceof SDFSBS ? built.brickCount > 0 : built instanceof SDFSVS || built instanceof SDFSVO ? built.voxelCount > 0 : built.lodCount > 0;
    if (!ok) throw new RuntimeError("SDFGrid: no values set");
    return built;
}
