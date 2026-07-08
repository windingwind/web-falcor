// Keyframe animation eval: samples per-node translation/rotation/scale channels
// (lerp/slerp) over a retained scene graph and composes global matrices down the
// hierarchy. Ports the core of Falcor's Animation::animate + AnimationController.
import { float3, lerp3, add3, mul3, normalize3 } from "../../Utils/Math/Vector.js";
import { quatf, slerp, matrixFromQuat } from "../../Utils/Math/Quaternion.js";
import { float4x4, mulMat, matrixFromTranslation, matrixFromScaling, transformPoint, transformVector } from "../../Utils/Math/Matrix.js";
import type { StaticVertex } from "../SceneData.js";

export type AnimationPath = "translation" | "rotation" | "scale";

export interface AnimationChannel {
    nodeID: number;
    path: AnimationPath;
    times: Float32Array; // seconds, ascending
    values: Float32Array; // flattened vec3 (translation/scale) or vec4 xyzw quat (rotation)
    interp: "LINEAR" | "STEP";
}

/** Retained scene-graph node with its bind-pose local TRS. Parents precede children. */
export interface SceneNode {
    parent: number; // parent node index, or -1 for a root
    t: float3;
    r: quatf;
    s: float3;
}

/** Decomposes a local matrix into TRS (for retaining animatable nodes whose
 *  transform is given as a matrix; animation channels override the components). */
export function decomposeTRS(m: float4x4): { t: float3; r: quatf; s: float3 } {
    const t = new float3(m.get(0, 3), m.get(1, 3), m.get(2, 3));
    const col = (c: number) => [m.get(0, c), m.get(1, c), m.get(2, c)] as [number, number, number];
    const c0 = col(0);
    const c1 = col(1);
    const c2 = col(2);
    const sx = Math.hypot(...c0) || 1;
    const sy = Math.hypot(...c1) || 1;
    const sz = Math.hypot(...c2) || 1;
    const R = [
        [c0[0] / sx, c1[0] / sy, c2[0] / sz],
        [c0[1] / sx, c1[1] / sy, c2[1] / sz],
        [c0[2] / sx, c1[2] / sy, c2[2] / sz],
    ];
    const tr = R[0]![0]! + R[1]![1]! + R[2]![2]!;
    let x: number, y: number, z: number, w: number;
    if (tr > 0) {
        const s = Math.sqrt(tr + 1) * 2;
        w = s / 4; x = (R[2]![1]! - R[1]![2]!) / s; y = (R[0]![2]! - R[2]![0]!) / s; z = (R[1]![0]! - R[0]![1]!) / s;
    } else if (R[0]![0]! > R[1]![1]! && R[0]![0]! > R[2]![2]!) {
        const s = Math.sqrt(1 + R[0]![0]! - R[1]![1]! - R[2]![2]!) * 2;
        w = (R[2]![1]! - R[1]![2]!) / s; x = s / 4; y = (R[0]![1]! + R[1]![0]!) / s; z = (R[0]![2]! + R[2]![0]!) / s;
    } else if (R[1]![1]! > R[2]![2]!) {
        const s = Math.sqrt(1 + R[1]![1]! - R[0]![0]! - R[2]![2]!) * 2;
        w = (R[0]![2]! - R[2]![0]!) / s; x = (R[0]![1]! + R[1]![0]!) / s; y = s / 4; z = (R[1]![2]! + R[2]![1]!) / s;
    } else {
        const s = Math.sqrt(1 + R[2]![2]! - R[0]![0]! - R[1]![1]!) * 2;
        w = (R[1]![0]! - R[0]![1]!) / s; x = (R[0]![2]! + R[2]![0]!) / s; y = (R[1]![2]! + R[2]![1]!) / s; z = s / 4;
    }
    return { t, r: new quatf(x, y, z, w), s: new float3(sx, sy, sz) };
}

export interface SceneAnimations {
    nodes: SceneNode[];
    channels: AnimationChannel[];
    start: number; // seconds (min first-keyframe time; clips needn't start at 0, e.g. FBX)
    duration: number; // seconds (max last-keyframe time across channels)
}

/** Skinning binding for one mesh (joints reference scene-graph nodes). */
export interface SkinDesc {
    boneNodeIDs: number[]; // scene node index per joint
    inverseBind: float4x4[]; // inverse bind matrix per joint
    boneIDs: Uint32Array; // 4 joint indices per vertex (into boneNodeIDs)
    weights: Float32Array; // 4 blend weights per vertex
}

/** Per-joint skinning matrix = jointGlobal · inverseBind (the mesh-node transform
 *  cancels out for glTF/Assimp skinning, so skinned verts land directly in world). */
export function computeSkinMatrices(skin: SkinDesc, globals: float4x4[]): float4x4[] {
    return skin.boneNodeIDs.map((nodeID, j) => mulMat(globals[nodeID]!, skin.inverseBind[j]!));
}

/** Linear-blend-skins bind-pose vertices to world space (position + normal). */
export function skinVertices(bind: StaticVertex[], skin: SkinDesc, skinMats: float4x4[]): StaticVertex[] {
    return bind.map((v, vi) => {
        let pos = new float3(0, 0, 0);
        let nrm = new float3(0, 0, 0);
        for (let k = 0; k < 4; k++) {
            const w = skin.weights[vi * 4 + k]!;
            if (w === 0) continue;
            const M = skinMats[skin.boneIDs[vi * 4 + k]!]!;
            pos = add3(pos, mul3(transformPoint(M, v.position), w));
            nrm = add3(nrm, mul3(transformVector(M, v.normal), w));
        }
        return { position: pos, normal: normalize3(nrm), tangent: v.tangent, texCrd: v.texCrd };
    });
}

/** Bracketing keyframe indices + interpolation fraction for `t`. */
function findSpan(times: Float32Array, t: number): { i0: number; i1: number; f: number } {
    const n = times.length;
    if (n === 0 || t <= times[0]!) return { i0: 0, i1: 0, f: 0 };
    if (t >= times[n - 1]!) return { i0: n - 1, i1: n - 1, f: 0 };
    let lo = 0;
    let hi = n - 1;
    while (lo + 1 < hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid]! <= t) lo = mid;
        else hi = mid;
    }
    const t0 = times[lo]!;
    const t1 = times[lo + 1]!;
    return { i0: lo, i1: lo + 1, f: t1 > t0 ? (t - t0) / (t1 - t0) : 0 };
}

function sampleVec3(ch: AnimationChannel, time: number): float3 {
    const { i0, i1, f } = findSpan(ch.times, time);
    const a = new float3(ch.values[i0 * 3]!, ch.values[i0 * 3 + 1]!, ch.values[i0 * 3 + 2]!);
    if (ch.interp === "STEP" || i0 === i1) return a;
    const b = new float3(ch.values[i1 * 3]!, ch.values[i1 * 3 + 1]!, ch.values[i1 * 3 + 2]!);
    return lerp3(a, b, f);
}

function sampleQuat(ch: AnimationChannel, time: number): quatf {
    const { i0, i1, f } = findSpan(ch.times, time);
    const a = new quatf(ch.values[i0 * 4]!, ch.values[i0 * 4 + 1]!, ch.values[i0 * 4 + 2]!, ch.values[i0 * 4 + 3]!);
    if (ch.interp === "STEP" || i0 === i1) return a;
    const b = new quatf(ch.values[i1 * 4]!, ch.values[i1 * 4 + 1]!, ch.values[i1 * 4 + 2]!, ch.values[i1 * 4 + 3]!);
    return slerp(a, b, f);
}

/**
 * Every node's global matrix at `time`: animation channels override the bind-pose
 * local TRS component they target; unanimated components keep the bind value. Nodes
 * are composed parents-first (the array is required to be in that order).
 */
export function evaluateGlobals(anim: SceneAnimations, time: number): float4x4[] {
    const n = anim.nodes.length;
    const locals = anim.nodes.map((nd) => ({ t: nd.t, r: nd.r, s: nd.s }));
    for (const ch of anim.channels) {
        if (ch.path === "translation") locals[ch.nodeID]!.t = sampleVec3(ch, time);
        else if (ch.path === "scale") locals[ch.nodeID]!.s = sampleVec3(ch, time);
        else locals[ch.nodeID]!.r = sampleQuat(ch, time);
    }
    // Resolve parents-first via memoized recursion (node order isn't guaranteed
    // topological; glTF is a forest so there are no cycles).
    const global: (float4x4 | undefined)[] = new Array(n);
    const resolve = (i: number): float4x4 => {
        const cached = global[i];
        if (cached) return cached;
        const l = locals[i]!;
        const localM = mulMat(matrixFromTranslation(l.t), mulMat(matrixFromQuat(l.r), matrixFromScaling(l.s)));
        const p = anim.nodes[i]!.parent;
        const g = p >= 0 ? mulMat(resolve(p), localM) : localM;
        global[i] = g;
        return g;
    };
    for (let i = 0; i < n; i++) resolve(i);
    return global as float4x4[];
}
