/**
 * Voxel grid host mirroring Scene/Volume/Grid.h: owns the NanoVDB grid
 * buffer (built from .vdb via the validated TS ports, or extracted from
 * .nvdb) plus the host-side stats native exposes. GPU upload arrives with
 * the Scene volume plumbing (Grid.slang binding).
 */

import type { Device } from "../../Core/API/Device.js";
import { RuntimeError } from "../../Core/Error.js";
import { parseOpenVDBFloatGrid, buildNanoVDBGrid, extractGridFromNVDB } from "./VDBLoader.js";

export class Grid {
    private view: DataView;

    constructor(
        readonly device: Device,
        /** NanoVDB v32.3 grid buffer (the exact bytes PNanoVDB traverses). */
        readonly gridBuffer: Uint8Array,
    ) {
        this.view = new DataView(gridBuffer.buffer, gridBuffer.byteOffset, gridBuffer.byteLength);
        if (this.view.getBigUint64(0, true) !== 0x304244566f6e614en) throw new RuntimeError("Grid: not a NanoVDB buffer");
    }

    static async createFromUrl(device: Device, url: string, gridname: string): Promise<Grid> {
        const res = await fetch(url);
        if (!res.ok) throw new RuntimeError(`Grid: failed to fetch '${url}' (${res.status})`);
        const data = await res.arrayBuffer();
        if (url.toLowerCase().endsWith(".nvdb")) {
            return new Grid(device, extractGridFromNVDB(data, gridname));
        }
        return new Grid(device, buildNanoVDBGrid(parseOpenVDBFloatGrid(data, gridname), gridname));
    }

    private get rootOffset(): number {
        return 672 + Number(this.view.getBigUint64(672 + 24, true));
    }

    get voxelCount(): number {
        return Number(this.view.getBigUint64(672 + 56, true));
    }

    /** Mirrors Grid::getMinIndex: index bbox min rounded down to 8-brick. */
    get minIndex(): [number, number, number] {
        const r = this.rootOffset;
        return [
            this.view.getInt32(r, true) & ~7,
            this.view.getInt32(r + 4, true) & ~7,
            this.view.getInt32(r + 8, true) & ~7,
        ];
    }

    /** Mirrors Grid::getMaxIndex: index bbox max rounded up to 8-brick. */
    get maxIndex(): [number, number, number] {
        const r = this.rootOffset;
        return [
            (this.view.getInt32(r + 12, true) + 7) & ~7,
            (this.view.getInt32(r + 16, true) + 7) & ~7,
            (this.view.getInt32(r + 20, true) + 7) & ~7,
        ];
    }

    get minValue(): number {
        return this.view.getFloat32(this.rootOffset + 32, true);
    }

    get maxValue(): number {
        return this.view.getFloat32(this.rootOffset + 36, true);
    }

    /** World-space AABB of active voxels (GridData::mWorldBBox). */
    get worldBounds(): { min: [number, number, number]; max: [number, number, number] } {
        const o = 560;
        return {
            min: [this.view.getFloat64(o, true), this.view.getFloat64(o + 8, true), this.view.getFloat64(o + 16, true)],
            max: [this.view.getFloat64(o + 24, true), this.view.getFloat64(o + 32, true), this.view.getFloat64(o + 40, true)],
        };
    }
}
