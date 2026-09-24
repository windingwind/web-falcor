/**
 * Native's triangulate() (Falcor's USDUtils/Tessellator) on a USD mesh's authored data:
 * fan triangulation (orientation-aware, hole faces skipped), authored normals by their
 * interpolation or generated flat per-face normals, and texcoords by theirs. The result is
 * expanded per triangle corner so every time sample has the same vertex layout.
 */
import type { UsdaSubdivMesh } from "./UsdaScene.js";
import type { TessellatedMesh } from "./Subdivision.js";

/** Per-corner triangle attributes (3 floats per corner for positions/normals, 2 for uvs). */
export interface CornerMesh {
    positions: Float32Array;
    normals: Float32Array;
    uvs: Float32Array | null;
    /** Each corner's source point (for per-point data such as joint influences). */
    pointIndices?: Uint32Array;
}

/** Indexes an attribute array of `interp` for a corner (point p, face-vertex c, face f). */
const pick = (interp: string, p: number, c: number, f: number): number => (interp === "vertex" || interp === "varying" ? p : interp === "faceVarying" ? c : interp === "uniform" ? f : 0);

export function triangulateUsdMesh(mesh: UsdaSubdivMesh, points: number[], normals: number[] | undefined): CornerMesh {
    const next = mesh.orientation === "rightHanded" ? [1, 2] : [2, 1];
    const holes = new Set(mesh.holeIndices);
    const normalInterp = normals && normals.length > 0 ? (mesh.normals?.interpolation ?? "vertex") : "none";
    const st = mesh.st && ["vertex", "varying", "faceVarying", "uniform"].includes(mesh.st.interpolation) ? mesh.st : undefined;
    const pos: number[] = [];
    const nrm: number[] = [];
    const uv: number[] = [];
    const corners: number[] = [];
    for (let f = 0, base = 0; f < mesh.faceVertexCounts.length; base += mesh.faceVertexCounts[f]!, f++) {
        const count = mesh.faceVertexCounts[f]!;
        if (holes.has(f) || count < 3) continue;
        const point = (k: number) => mesh.faceVertexIndices[base + k]!;
        let flat: [number, number, number] | null = null;
        if (normalInterp === "none") {
            // Uniform face normal: the normalized sum of the fan's cross products.
            const p0 = point(0);
            const n: [number, number, number] = [0, 0, 0];
            for (let j = 0; j < count - 2; j++) {
                const [a, b] = [point(j + next[0]!), point(j + next[1]!)];
                const e1 = [0, 1, 2].map((k) => points[a * 3 + k]! - points[p0 * 3 + k]!);
                const e2 = [0, 1, 2].map((k) => points[b * 3 + k]! - points[p0 * 3 + k]!);
                n[0] += e1[1]! * e2[2]! - e1[2]! * e2[1]!;
                n[1] += e1[2]! * e2[0]! - e1[0]! * e2[2]!;
                n[2] += e1[0]! * e2[1]! - e1[1]! * e2[0]!;
            }
            const len = Math.hypot(n[0]!, n[1]!, n[2]!) || 1;
            flat = [n[0]! / len, n[1]! / len, n[2]! / len];
        }
        for (let v = 0; v < count - 2; v++) {
            for (const k of [0, v + next[0]!, v + next[1]!]) {
                const p = point(k);
                pos.push(points[p * 3]!, points[p * 3 + 1]!, points[p * 3 + 2]!);
                corners.push(p);
                if (flat) nrm.push(...flat);
                else {
                    const i = pick(normalInterp, p, base + k, f);
                    nrm.push(normals![i * 3]!, normals![i * 3 + 1]!, normals![i * 3 + 2]!);
                }
                if (st) {
                    const i = pick(st.interpolation, p, base + k, f);
                    uv.push(st.values[i * 2]!, st.values[i * 2 + 1]!);
                }
            }
        }
    }
    return { positions: Float32Array.from(pos), normals: Float32Array.from(nrm), uvs: st ? Float32Array.from(uv) : null, pointIndices: Uint32Array.from(corners) };
}

/** A refined mesh expanded per triangle corner (see TessellatedMesh). */
export function refinedCorners(m: TessellatedMesh): CornerMesh {
    const n = m.indices.length;
    const positions = new Float32Array(n * 3);
    const normals = new Float32Array(n * 3);
    const uvs = m.uvs ? new Float32Array(n * 2) : null;
    const faceUv = new Map<number, number>();
    for (const f of m.coarseFaces) if (!faceUv.has(f)) faceUv.set(f, faceUv.size);
    for (let c = 0; c < n; c++) {
        const p = m.indices[c]!;
        positions.set(m.positions.subarray(p * 3, p * 3 + 3), c * 3);
        normals.set(m.normals.subarray(p * 3, p * 3 + 3), c * 3);
        if (uvs && m.uvs) {
            const i = m.uvInterp === "faceVarying" ? c : m.uvInterp === "uniform" ? faceUv.get(m.coarseFaces[Math.floor(c / 3)]!)! : p;
            uvs.set(m.uvs.subarray(i * 2, i * 2 + 2), c * 2);
        }
    }
    return { positions, normals, uvs };
}
