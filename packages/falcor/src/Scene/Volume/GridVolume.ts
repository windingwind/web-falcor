/**
 * Grid volume host mirroring Scene/Volume/GridVolume.h (single-frame subset:
 * no grid sequences/playback yet). Holds per-slot grids and the medium
 * parameters that feed GridVolumeData.
 */

import { float3 } from "../../Utils/Math/Vector.js";
import type { Grid } from "./Grid.js";

export type GridSlot = "density" | "emission";

export class GridVolume {
    densityScale = 1;
    emissionScale = 1;
    albedo = new float3(1, 1, 1);
    anisotropy = 0;
    emissionTemperature = 0;
    private grids: Partial<Record<GridSlot, Grid>> = {};

    constructor(readonly name: string) {}

    setGrid(slot: GridSlot, grid: Grid): void {
        this.grids[slot] = grid;
    }

    getGrid(slot: GridSlot): Grid | undefined {
        return this.grids[slot];
    }

    get densityGrid(): Grid | undefined {
        return this.grids["density"];
    }

    get emissionGrid(): Grid | undefined {
        return this.grids["emission"];
    }

    /** Mirrors GridVolume::updateBounds (identity volume transform). */
    get bounds(): { min: [number, number, number]; max: [number, number, number] } | null {
        const grids = [this.grids["density"], this.grids["emission"]].filter((g) => g !== undefined);
        if (grids.length === 0) return null;
        const min: [number, number, number] = [Infinity, Infinity, Infinity];
        const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        for (const g of grids) {
            const b = g.worldBounds;
            for (let k = 0; k < 3; k++) {
                min[k] = Math.min(min[k]!, b.min[k]!);
                max[k] = Math.max(max[k]!, b.max[k]!);
            }
        }
        return { min, max };
    }
}
