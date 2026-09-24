/**
 * Tangent generation as SceneBuilder::addMesh does it: MikkTSpace (Falcor/external/mikktspace,
 * compiled to packages/falcor/wasm/mikktspace.wasm by scripts/build-mikktspace-wasm.mjs) gives a
 * tangent per corner, NaN tangents become perp_stark(normal), and identical corners of each
 * original vertex are merged again (compareVertices), which splits vertices at UV seams and
 * mirrored parametrizations. loadMikkTSpace() must run first (SceneBuilder.resolve does); without
 * it the approximate per-vertex path below is used, with a warning.
 */

import { float3, float4 } from "../Utils/Math/Vector.js";
import type { StaticVertex } from "./SceneData.js";
import { Logger } from "../Utils/Logger.js";

/**
 * How a mesh's tangents are produced: "generate" (none supplied), "asset" (supplied, but native
 * regenerates them unless Flags::UseOriginalTangentSpace), "keep" (Mesh::useOriginalTangentSpace),
 * "noTexCrds" (no texture coordinates: native skips MikkTSpace and leaves the tangents zero).
 */
export type TangentSpaceMode = "generate" | "asset" | "keep" | "noTexCrds";

interface MikkExports {
    memory: WebAssembly.Memory;
    mikk_generate(faceCount: number, positions: number, normals: number, texCrds: number, tangents: number): number;
    mikk_alloc(bytes: number): number;
    mikk_free(ptr: number): void;
    _initialize(): void;
}
let mikk: MikkExports | null = null;
let warnedFallback = false;

/** Loads the MikkTSpace wasm module (once). */
export async function loadMikkTSpace(): Promise<void> {
    if (mikk) return;
    const url = new URL("../../wasm/mikktspace.wasm", import.meta.url);
    const bytes =
        url.protocol === "file:"
            ? await (await import(/* @vite-ignore */ "node:fs/promises".toString())).readFile(url)
            : await (await fetch(url)).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, { env: { emscripten_notify_memory_growth: () => {} } });
    mikk = instance.exports as unknown as MikkExports;
    mikk._initialize();
}

/** Mirrors math::perp_stark: a unit vector perpendicular to u. */
function perpStark(u: float3): [number, number, number] {
    const [ax, ay, az] = [Math.abs(u.x), Math.abs(u.y), Math.abs(u.z)];
    const uyx = ax - ay < 0 ? 1 : 0, uzx = ax - az < 0 ? 1 : 0, uzy = ay - az < 0 ? 1 : 0;
    const xm = uyx & uzx;
    const ym = (1 ^ xm) & uzy;
    const zm = 1 ^ (xm | ym);
    const c = [u.y * zm - u.z * ym, u.z * xm - u.x * zm, u.x * ym - u.y * xm];
    const r = Math.fround(1 / Math.sqrt(c[0]! * c[0]! + c[1]! * c[1]! + c[2]! * c[2]!));
    return [c[0]! * r, c[1]! * r, c[2]! * r];
}

/** MikkTSpace tangents per corner (float4 each), or null when the wasm module isn't loaded or fails. */
export function generateCornerTangents(vertices: StaticVertex[], indices: Uint32Array): Float32Array | null {
    if (!mikk || indices.length < 3) return null;
    const corners = indices.length - (indices.length % 3);
    const faceCount = corners / 3;
    const bytes = corners * (3 + 3 + 2 + 4) * 4;
    const base = mikk.mikk_alloc(bytes);
    const [pos, nrm, uv, out] = [base, base + corners * 12, base + corners * 24, base + corners * 32];
    const heap = new Float32Array(mikk.memory.buffer);
    for (let i = 0; i < corners; i++) {
        const v = vertices[indices[i]!]!;
        heap.set([v.position.x, v.position.y, v.position.z], pos / 4 + i * 3);
        heap.set([v.normal.x, v.normal.y, v.normal.z], nrm / 4 + i * 3);
        heap.set([v.texCrd.x, v.texCrd.y], uv / 4 + i * 2);
    }
    const ok = mikk.mikk_generate(faceCount, pos, nrm, uv, out) !== 0;
    const tangents = new Float32Array(mikk.memory.buffer, out, corners * 4).slice();
    mikk.mikk_free(base);
    if (!ok) return null;
    // SceneBuilder::generateTangents: NaN tangents (degenerate triangles) become perp_stark(normal).
    for (let i = 0; i < corners; i++) {
        if ([0, 1, 2, 3].some((c) => Number.isNaN(tangents[i * 4 + c]))) {
            tangents.set([...perpStark(vertices[indices[i]!]!.normal), 1], i * 4);
        }
    }
    return tangents;
}

/** Per-vertex data compareVertices also looks at (skinning), 4 values per vertex. */
export interface VertexExtras {
    boneIDs?: Uint32Array;
    boneWeights?: Float32Array;
}

/**
 * SceneBuilder::addMesh's tangent space and vertex merge: returns the new vertices and indices,
 * and for each new vertex the original vertex it came from (to remap per-vertex side data).
 * Null when MikkTSpace is unavailable.
 */
export function generateTangentsAndMerge(vertices: StaticVertex[], indices: Uint32Array, extras: VertexExtras = {}): { vertices: StaticVertex[]; indices: Uint32Array; source: Uint32Array } | null {
    const tangents = generateCornerTangents(vertices, indices);
    if (!tangents) return null;
    const eps = 1e-6;
    const near = (a: number, b: number) => !(Math.abs(a - b) > eps);
    const same = (a: StaticVertex, b: StaticVertex, ia: number, ib: number) => {
        if (a.position.x !== b.position.x || a.position.y !== b.position.y || a.position.z !== b.position.z) return false;
        if (a.tangent.w !== b.tangent.w) return false;
        if (extras.boneIDs) for (let k = 0; k < 4; k++) if (extras.boneIDs[ia * 4 + k] !== extras.boneIDs[ib * 4 + k]) return false;
        if (!near(a.normal.x, b.normal.x) || !near(a.normal.y, b.normal.y) || !near(a.normal.z, b.normal.z)) return false;
        if (!near(a.tangent.x, b.tangent.x) || !near(a.tangent.y, b.tangent.y) || !near(a.tangent.z, b.tangent.z)) return false;
        if (!near(a.texCrd.x, b.texCrd.x) || !near(a.texCrd.y, b.texCrd.y)) return false;
        if (extras.boneWeights) for (let k = 0; k < 4; k++) if (!near(extras.boneWeights[ia * 4 + k]!, extras.boneWeights[ib * 4 + k]!)) return false;
        return true;
    };
    const out: StaticVertex[] = [];
    const source: number[] = [];
    const next: number[] = [];
    const heads = new Int32Array(vertices.length).fill(-1);
    const newIndices = new Uint32Array(indices.length);
    const corners = indices.length - (indices.length % 3);
    for (let i = 0; i < corners; i++) {
        const orig = indices[i]!;
        const v = { ...vertices[orig]!, tangent: new float4(tangents[i * 4]!, tangents[i * 4 + 1]!, tangents[i * 4 + 2]!, tangents[i * 4 + 3]!) };
        let index = heads[orig]!;
        while (index >= 0 && !same(v, out[index]!, orig, source[index]!)) index = next[index]!;
        if (index < 0) {
            index = out.length;
            out.push(v);
            source.push(orig);
            next.push(heads[orig]!);
            heads[orig] = index;
        }
        newIndices[i] = index;
    }
    return { vertices: out, indices: newIndices, source: Uint32Array.from(source) };
}

/**
 * In-place tangents for meshes whose index buffer already gives each corner its own vertex
 * (USD corner meshes, vertex-cache frames): MikkTSpace when loaded, else the approximation.
 */
export function generateTangents(vertices: StaticVertex[], indices: Uint32Array): void {
    const tangents = generateCornerTangents(vertices, indices);
    if (tangents) {
        indices.forEach((v, i) => (i < tangents.length / 4 ? (vertices[v]!.tangent = new float4(tangents[i * 4]!, tangents[i * 4 + 1]!, tangents[i * 4 + 2]!, tangents[i * 4 + 3]!)) : undefined));
        return;
    }
    if (!warnedFallback) {
        warnedFallback = true;
        Logger.warning("MikkTSpace wasm not loaded (loadMikkTSpace): using approximate per-vertex tangents");
    }
    generateTangentsApprox(vertices, indices);
}

/** The pre-MikkTSpace approximation: angle-weighted per-vertex face tangents, Gram-Schmidt'd. */
function generateTangentsApprox(vertices: StaticVertex[], indices: Uint32Array): void {
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
        // mikktspace.c InitTriInfo: vOs is scaled by fS = mirrored ? -1 : +1,
        // so the face tangent always points along +dP/du.
        const fS = area >= 0 ? 1 : -1;
        os[0]! *= fS;
        os[1]! *= fS;
        os[2]! *= fS;

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
            accum[a * 3]! += (os[0]! / len) * w;
            accum[a * 3 + 1]! += (os[1]! / len) * w;
            accum[a * 3 + 2]! += (os[2]! / len) * w;
            orient[a]! += area > 0 ? w : -w;
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
