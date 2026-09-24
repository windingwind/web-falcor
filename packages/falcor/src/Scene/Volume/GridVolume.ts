/**
 * Grid volume host mirroring Scene/Volume/GridVolume.h: per-slot grid
 * *sequences* plus the medium parameters that feed GridVolumeData. A sequence
 * plays back over the scene clock exactly as native does; the frame in effect
 * selects which grid every accessor returns.
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
    /** Mirrors GridVolume::EmissionMode (Direct = 0, Blackbody = 1; the shaders read only Direct). */
    emissionMode = 0;
    private grids: Partial<Record<GridSlot, Grid[]>> = {};
    private frame = 0;
    private rate = 30;
    private start = 0;
    /** Mirrors mPlaybackEnabled (native defaults to playing). */
    playbackEnabled = true;

    constructor(readonly name: string) {}

    setGrid(slot: GridSlot, grid: Grid): void {
        this.setGridSequence(slot, [grid]);
    }

    /** Mirrors GridVolume::setGridSequence. */
    setGridSequence(slot: GridSlot, grids: Grid[]): void {
        this.grids[slot] = grids;
        this.frame = Math.min(this.frame, Math.max(this.gridFrameCount - 1, 0));
    }

    getGridSequence(slot: GridSlot): Grid[] {
        return this.grids[slot] ?? [];
    }

    /** Mirrors GridVolume::getGrid: the sequence clamps to its last frame. */
    getGrid(slot: GridSlot): Grid | undefined {
        const sequence = this.grids[slot];
        if (!sequence || sequence.length === 0) return undefined;
        return sequence[Math.min(this.frame, sequence.length - 1)];
    }

    /** Mirrors GridVolume::getGridFrameCount (the longest slot's sequence). */
    get gridFrameCount(): number {
        return Math.max(0, ...Object.values(this.grids).map((g) => g?.length ?? 0));
    }

    get gridFrame(): number {
        return this.frame;
    }

    /** Mirrors GridVolume::setGridFrame. */
    set gridFrame(value: number) {
        this.frame = Math.min(Math.max(Math.trunc(value), 0), Math.max(this.gridFrameCount - 1, 0));
    }

    get frameRate(): number {
        return this.rate;
    }

    /** Mirrors GridVolume::setFrameRate (clamped to [1, 1000]). */
    set frameRate(value: number) {
        this.rate = Math.min(Math.max(value, 1), 1000);
    }

    get startFrame(): number {
        return this.start;
    }

    /** Mirrors GridVolume::setStartFrame. */
    set startFrame(value: number) {
        this.start = this.gridFrameCount > 0 ? Math.min(Math.max(Math.trunc(value), 0), this.gridFrameCount - 1) : 0;
    }

    /**
     * Mirrors GridVolume::updatePlayback.
     *
     * @returns true if the selected frame changed (the scene must rebind grids).
     */
    updatePlayback(currentTime: number): boolean {
        if (!this.playbackEnabled || this.gridFrameCount === 0) return false;
        const index = (this.start + Math.floor(Math.max(0, currentTime) * this.rate)) % this.gridFrameCount;
        if (index === this.frame) return false;
        this.frame = index;
        return true;
    }

    /** The density grid of the frame in effect. */
    get densityGrid(): Grid | undefined {
        return this.getGrid("density");
    }

    /** The emission grid of the frame in effect. */
    get emissionGrid(): Grid | undefined {
        return this.getGrid("emission");
    }

    /** Mirrors GridVolume::updateBounds (identity volume transform). */
    get bounds(): { min: [number, number, number]; max: [number, number, number] } | null {
        const grids = [this.getGrid("density"), this.getGrid("emission")].filter((g) => g !== undefined);
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
