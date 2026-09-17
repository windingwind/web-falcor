/**
 * SDF grid value files (`.sdfg`) mirroring SDFGrid::loadValuesFromFile /
 * writeValuesToFile.
 *
 * The format is a `uint32` grid width followed by `(width + 1)^3` float corner
 * values — signed distances in the unit cube the grid occupies, so positions run
 * over `[-0.5, 0.5]^3` and native clamps the distances to +/- sqrt(3).
 *
 * The companion `.sdf` format (a JSON array of editable SDF3DPrimitives) needs
 * the primitive evaluation the web port does not have yet (docs §8.4).
 */

import { RuntimeError } from "../../Core/Error.js";

/** Native clamps corner values to the unit cube's diagonal. */
export const kSDFMaxDistance = Math.sqrt(3);

export interface SDFGridValues {
    /** Cells per axis; there are (gridWidth + 1)^3 corner values. */
    gridWidth: number;
    values: Float32Array;
}

/** Mirrors SDFGrid::loadValuesFromFile. */
export function parseSDFGridValues(buffer: ArrayBuffer): SDFGridValues {
    if (buffer.byteLength < 4) throw new RuntimeError("SDFGrid: .sdfg file is too short");
    const view = new DataView(buffer);
    const gridWidth = view.getUint32(0, true);
    const widthInValues = gridWidth + 1;
    const total = widthInValues * widthInValues * widthInValues;
    if (gridWidth === 0 || buffer.byteLength < 4 + total * 4) {
        throw new RuntimeError(`SDFGrid: .sdfg file truncated (grid width ${gridWidth} needs ${total} values)`);
    }
    // The payload is not 4-byte aligned in general, so copy it out.
    return { gridWidth, values: new Float32Array(buffer.slice(4, 4 + total * 4)) };
}

/** Mirrors SDFGrid::writeValuesToFile. */
export function encodeSDFGridValues(grid: SDFGridValues): Uint8Array {
    const widthInValues = grid.gridWidth + 1;
    const total = widthInValues * widthInValues * widthInValues;
    if (grid.values.length !== total) throw new RuntimeError(`SDFGrid: expected ${total} corner values, got ${grid.values.length}`);
    const out = new Uint8Array(4 + total * 4);
    new DataView(out.buffer).setUint32(0, grid.gridWidth, true);
    out.set(new Uint8Array(grid.values.buffer, grid.values.byteOffset, total * 4), 4);
    return out;
}

/** Fetches a `.sdfg` (web divergence, docs §9: native reads it from disk). */
export async function loadSDFGridValues(url: string): Promise<SDFGridValues> {
    const res = await fetch(url);
    if (!res.ok) throw new RuntimeError(`SDFGrid: file '${url}' could not be opened (${res.status})`);
    return parseSDFGridValues(await res.arrayBuffer());
}
