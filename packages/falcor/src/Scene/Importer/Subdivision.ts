/**
 * Subdivision surfaces for the USD importer: native's tessellate() (Falcor's
 * USDUtils/Tessellator) through OpenSubdiv's Bfr, compiled to wasm
 * (packages/falcor/wasm, built by scripts/build-opensubdiv-wasm.mjs).
 */
import { Logger } from "../../Utils/Logger.js";
import type { UsdaSubdivMesh } from "./UsdaScene.js";

export interface OsdModule {
    HEAPF32: Float32Array;
    HEAP32: Int32Array;
    _malloc(bytes: number): number;
    _free(ptr: number): void;
    _osd_tessellate(scheme: number, level: number, leftHanded: number, vtxBoundary: number, fvarLinear: number, points: number, numPoints: number, faceCounts: number, numFaces: number, faceIndices: number, numIndices: number, uvs: number, uvInterp: number): number;
    _osd_positions(): number;
    _osd_position_count(): number;
    _osd_normals(): number;
    _osd_uvs(): number;
    _osd_uv_count(): number;
    _osd_indices(): number;
    _osd_coarse_faces(): number;
}

let modulePromise: Promise<OsdModule> | null = null;
/** Loads the OpenSubdiv wasm module (once). */
export const loadOpenSubdiv = (): Promise<OsdModule> => {
    // Literal URLs so bundlers emit both files; the wasm is located explicitly.
    const wasmUrl = new URL("../../../wasm/opensubdiv.wasm", import.meta.url).href;
    modulePromise ??= import(/* @vite-ignore */ new URL("../../../wasm/opensubdiv.mjs", import.meta.url).href).then((m: { default: (opts: object) => Promise<OsdModule> }) =>
        m.default({ locateFile: () => wasmUrl }),
    );
    return modulePromise;
};

/** A refined mesh: shared positions/normals, triangle indices, and texcoords per the input's interpolation. */
export interface TessellatedMesh {
    positions: Float32Array;
    normals: Float32Array;
    indices: Uint32Array;
    /** Per position (vertex/varying), per triangle corner (faceVarying) or per coarse face (uniform). */
    uvs: Float32Array | null;
    uvInterp: string;
    /** Each triangle's originating coarse face. */
    coarseFaces: Uint32Array;
}

const kBoundary: Record<string, number> = { none: 0, edgeOnly: 1, edgeAndCorner: 2 };
const kFVarLinear: Record<string, number> = { none: 0, cornersOnly: 1, cornersPlus1: 2, cornersPlus2: 3, boundaries: 4, all: 5 };
const kUvInterp: Record<string, number> = { vertex: 1, varying: 2, faceVarying: 3, uniform: 4 };

/** Refines `mesh` to `level` like native's tessellate(); null where native falls back to the unrefined mesh. */
export function tessellateUsdMesh(osd: OsdModule, path: string, mesh: UsdaSubdivMesh, level: number): TessellatedMesh | null {
    if (level <= 0 || mesh.points.length === 0 || mesh.faceVertexCounts.length === 0) return null;
    const scheme = { catmullClark: 0, loop: 1, bilinear: 2 }[mesh.scheme];
    if (scheme === undefined) {
        Logger.warning(`Unknown subdivision scheme: '${mesh.scheme}' on mesh '${path}'. Unrefined mesh will be used.`);
        return null;
    }
    if (scheme === 1 && mesh.faceVertexCounts.some((c) => c !== 3)) {
        Logger.warning(`Cannot apply Loop subdivision to non-triangular mesh '${path}'. Unrefined mesh will be used.`);
        return null;
    }
    const boundary = kBoundary[mesh.interpolateBoundary];
    if (boundary === undefined) Logger.warning(`Unsupported vertex boundary interpolation mode '${mesh.interpolateBoundary}' on '${path}'.`);
    const fvar = kFVarLinear[mesh.faceVaryingLinearInterpolation];
    if (fvar === undefined) Logger.warning(`Unsupported face varying linear interpolation mode '${mesh.faceVaryingLinearInterpolation}' on '${path}'.`);
    const uvInterp = mesh.st ? (kUvInterp[mesh.st.interpolation] ?? 0) : 0;

    const allocs: number[] = [];
    const put = (data: ArrayLike<number>, float: boolean) => {
        const ptr = osd._malloc(Math.max(4, data.length * 4));
        allocs.push(ptr);
        (float ? osd.HEAPF32 : osd.HEAP32).set(data, ptr >> 2);
        return ptr;
    };
    try {
        const triangles = osd._osd_tessellate(
            scheme, level, mesh.orientation === "leftHanded" ? 1 : 0, boundary ?? 2, fvar ?? 2,
            put(mesh.points, true), mesh.points.length / 3,
            put(mesh.faceVertexCounts, false), mesh.faceVertexCounts.length,
            put(mesh.faceVertexIndices, false), mesh.faceVertexIndices.length,
            uvInterp ? put(mesh.st!.values, true) : 0, uvInterp,
        );
        if (triangles < 0) return null;
        const f32 = (ptr: number, n: number) => osd.HEAPF32.slice(ptr >> 2, (ptr >> 2) + n);
        const i32 = (ptr: number, n: number) => new Uint32Array(osd.HEAP32.slice(ptr >> 2, (ptr >> 2) + n));
        const count = osd._osd_position_count();
        return {
            positions: f32(osd._osd_positions(), count * 3),
            normals: f32(osd._osd_normals(), count * 3),
            indices: i32(osd._osd_indices(), triangles * 3),
            uvs: uvInterp ? f32(osd._osd_uvs(), osd._osd_uv_count() * 2) : null,
            uvInterp: mesh.st && uvInterp ? mesh.st.interpolation : "none",
            coarseFaces: i32(osd._osd_coarse_faces(), triangles),
        };
    } finally {
        for (const p of allocs) osd._free(p);
    }
}
