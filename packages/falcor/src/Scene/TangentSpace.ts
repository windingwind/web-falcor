/**
 * Tangent generation mirroring SceneBuilder's MikkTSpace path (SceneBuilder.cpp
 * MikkTSpaceWrapper::generateTangents). This implements the MikkTSpace core:
 * per-face tangents from UV derivatives (unflipped by mirrored parametrizations,
 * whose handedness goes into the sign instead), angle-weighted per-vertex
 * averaging, and Gram-Schmidt projection against the vertex normal.
 *
 * Exact for meshes whose vertex welding matches the index buffer (MikkTSpace
 * re-welds by position+normal+uv and splits groups at mirrored seams; meshes
 * exercising those paths may diverge until the full port lands — DESIGN.md §8).
 */

import { float3, float4 } from "../Utils/Math/Vector.js";
import type { StaticVertex } from "./SceneData.js";

export function generateTangents(vertices: StaticVertex[], indices: Uint32Array): void {
    const accum = new Float32Array(vertices.length * 3);
    const orient = new Float32Array(vertices.length); // signed vote: >0 preserving, <0 mirrored

    for (let f = 0; f + 2 < indices.length; f += 3) {
        const i0 = indices[f]!;
        const i1 = indices[f + 1]!;
        const i2 = indices[f + 2]!;
        const p0 = vertices[i0]!.position;
        const p1 = vertices[i1]!.position;
        const p2 = vertices[i2]!.position;
        const u0 = vertices[i0]!.texCrd;
        const u1 = vertices[i1]!.texCrd;
        const u2 = vertices[i2]!.texCrd;

        const e1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
        const e2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
        const s1 = u1.x - u0.x;
        const s2 = u2.x - u0.x;
        const t1 = u1.y - u0.y;
        const t2 = u2.y - u0.y;
        // Signed parametric area; MikkTSpace keeps vOs un-divided by it so a
        // mirrored mapping flips handedness (recorded below), not the vector.
        const area = s1 * t2 - s2 * t1;
        const os = [e1[0]! * t2 - e2[0]! * t1, e1[1]! * t2 - e2[1]! * t1, e1[2]! * t2 - e2[2]! * t1];
        const len = Math.hypot(os[0]!, os[1]!, os[2]!);
        if (len === 0) continue;

        for (const [a, b, c] of [
            [i0, i1, i2],
            [i1, i2, i0],
            [i2, i0, i1],
        ] as const) {
            // Angle weight at corner a (MikkTSpace weights per-corner contributions).
            const va = vertices[a]!.position;
            const vb = vertices[b]!.position;
            const vc = vertices[c]!.position;
            const d1 = new float3(vb.x - va.x, vb.y - va.y, vb.z - va.z);
            const d2 = new float3(vc.x - va.x, vc.y - va.y, vc.z - va.z);
            const l1 = Math.hypot(d1.x, d1.y, d1.z);
            const l2 = Math.hypot(d2.x, d2.y, d2.z);
            const cos = l1 > 0 && l2 > 0 ? Math.min(1, Math.max(-1, (d1.x * d2.x + d1.y * d2.y + d1.z * d2.z) / (l1 * l2))) : 1;
            const w = Math.acos(cos);
            accum[a * 3] += (os[0]! / len) * w;
            accum[a * 3 + 1] += (os[1]! / len) * w;
            accum[a * 3 + 2] += (os[2]! / len) * w;
            orient[a] += area > 0 ? w : -w;
        }
    }

    for (let i = 0; i < vertices.length; i++) {
        const n = vertices[i]!.normal;
        let tx = accum[i * 3]!;
        let ty = accum[i * 3 + 1]!;
        let tz = accum[i * 3 + 2]!;
        // Project against the normal and normalize (MikkTSpace tSpace output).
        const dot = tx * n.x + ty * n.y + tz * n.z;
        tx -= dot * n.x;
        ty -= dot * n.y;
        tz -= dot * n.z;
        const len = Math.hypot(tx, ty, tz);
        if (len > 1e-12) {
            vertices[i]!.tangent = new float4(tx / len, ty / len, tz / len, orient[i]! >= 0 ? 1 : -1);
        } else {
            vertices[i]!.tangent = new float4(1, 0, 0, 1);
        }
    }
}
