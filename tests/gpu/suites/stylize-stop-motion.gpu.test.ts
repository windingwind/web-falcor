/**
 * StopMotion (packages/render-passes/src/Stylize/StopMotion.ts):
 *  - CPU reference: per-object jitter (with and without posW ordering, all ids and an id subset) and flicker gain against a
 *    CPU evaluation of the same gather over the same readbacks (exact: the pass only copies pixels and scales by a gain);
 *  - identity: jitterPx 0 and flicker 0 return src bit for bit; frames inside one key are bitwise equal, the next key differs;
 *  - per-object frame rate: nodeHolds { 0: 2 } renders bitwise like Scene.setAnimationTimeQuantization({ 0: 2 / fps }),
 *    node 1 keeps animating every frame, and an empty nodeHolds leaves the scene's quantisation as it was;
 *  - cold vs consecutive: on the animated scene, frame t rendered by a fresh graph equals the same frame rendered after
 *    other frames on one graph (the pass keeps no state).
 */

import { LightType, Properties, Scene, float2, float3, float4, quatf, type Device } from "@web-falcor/falcor";
import type { AnimationChannel, SceneNode } from "../../../packages/falcor/src/Scene/Animation/SceneAnimation.js";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, camInfo, kSize, log, makeScene, readAll, runGraph, type V3 } from "./stylize-helpers.js";

const kFps = 24;
const kEdges: [string, string][] = [["G.vbuffer", "S.vbuffer"], ["G.diffuseOpacity", "M.src"], ["S.id", "M.id"], ["G.posW", "M.posW"]];
const kEdgesNoPos = kEdges.filter(([, b]) => b !== "M.posW");
const kOuts = ["M.dst", "G.diffuseOpacity", "S.id", "G.posW"];

const pcg = (v: number): number => {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
};
const keyOf = (t: number, hold: number) => Math.max(0, Math.floor((t * kFps) / hold + 1e-6));
const gainOf = (flicker: number, key: number, seed: number) =>
    flicker === 0 ? 1 : Math.fround(1 + flicker * (2 * ((pcg((key >>> 0) + pcg((seed ^ 0x9e3779b9) >>> 0)) >>> 8) / 16777216) - 1));

type P = { jitterPx: number; jitterHold?: number; jitterIds?: number[]; flicker?: number; flickerHold?: number; seed?: number; time: number };

/** CPU version of the shader's gather + gain. */
function reference(p: P, src: Float32Array, id: Float32Array, pos: Float32Array | null, cam: { pos: V3; fwd: V3 }): Float32Array {
    const R = Math.round(p.jitterPx), seed = p.seed ?? 0;
    const key = keyOf(p.time, p.jitterHold ?? 2);
    const gain = gainOf(p.flicker ?? 0, keyOf(p.time, p.flickerHold ?? 2), seed);
    const ids = p.jitterIds ?? [];
    const info = (i: number): { j: boolean; id: number; d: number } => {
        const idv = id[i * 4 + 1]!;
        let bg = id[i * 4 + 3] === 0, d = 0;
        if (pos) {
            bg ||= pos[i * 4 + 3] === 0;
            d = Math.fround((pos[i * 4]! - cam.pos[0]) * cam.fwd[0] + (pos[i * 4 + 1]! - cam.pos[1]) * cam.fwd[1] + (pos[i * 4 + 2]! - cam.pos[2]) * cam.fwd[2]);
        }
        if (bg) d = 3.4e38;
        return { j: !bg && (ids.length === 0 || ids.includes(idv)), id: idv >>> 0, d };
    };
    const off = (idv: number): [number, number] => {
        const n = 2 * R + 1, h = pcg(idv + pcg(key + pcg(seed))), h2 = pcg(h);
        return [(h % n) - R, (h2 % n) - R];
    };
    const out = new Float32Array(src.length);
    for (let y = 0; y < kSize; y++) for (let x = 0; x < kSize; x++) {
        const i = y * kSize + x;
        let best = i;
        if (R > 0) {
            const ip = info(i);
            let bestD = ip.j ? 3.4e38 : ip.d;
            for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
                const qx = x - dx, qy = y - dy;
                if (qx < 0 || qy < 0 || qx >= kSize || qy >= kSize) continue;
                const q = info(qy * kSize + qx);
                if (!q.j) continue;
                const o = off(q.id);
                if (qx + o[0] !== x || qy + o[1] !== y) continue;
                if (q.d < bestD) { bestD = q.d; best = qy * kSize + qx; }
            }
        }
        for (let k = 0; k < 3; k++) out[i * 4 + k] = gain === 1 ? src[best * 4 + k]! : Math.fround(src[best * 4 + k]! * gain);
        out[i * 4 + 3] = src[best * 4 + 3]!;
    }
    return out;
}

function countDiff(a: Float32Array, b: Float32Array): number {
    let n = 0;
    for (let i = 0; i < a.length; i += 4) if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2] || a[i + 3] !== b[i + 3]) n++;
    return n;
}

gpuTest("StopMotion.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    const cam = camInfo(scene);
    const g0 = await runGraph(device, scene, [["StyleAovs", "S"], ["StopMotion", "M"]], kEdges, kOuts);
    const [, , id0] = await readAll(device, g0, kOuts);
    const allIds = [...new Set(Array.from({ length: kSize * kSize }, (_, i) => (id0![i * 4 + 3] !== 0 ? id0![i * 4 + 1]! : -1)).filter((v) => v >= 0))];
    const cases: [string, P, boolean][] = [
        ["3px all posW", { jitterPx: 3, time: 0 }, true],
        ["4px all posW key 5", { jitterPx: 4, jitterHold: 3, time: 15 / kFps, seed: 11 }, true],
        ["2px subset no posW flicker", { jitterPx: 2, jitterIds: allIds.slice(0, 3), flicker: 0.2, flickerHold: 1, time: 7 / kFps }, false],
        ["flicker only", { jitterPx: 0, flicker: 0.3, time: 3 / kFps, seed: 2 }, true],
    ];
    for (const [name, p, withPos] of cases) {
        const g = await runGraph(device, scene, [["StyleAovs", "S"], ["StopMotion", "M", p]], withPos ? kEdges : kEdgesNoPos, kOuts);
        const [dst, src, id, pos] = await readAll(device, g, kOuts);
        const ref = reference(p, src!, id!, withPos ? pos! : null, cam);
        const bad = countDiff(dst!, ref), moved = countDiff(dst!, src!);
        log(`StopMotion[${name}]: pixels changed vs src ${moved}, mismatches vs CPU ${bad}`);
        expectEq(moved > 100, true, `${name}: the effect changes the image`);
        expectEq(bad <= 2, true, `${name}: matches the CPU reference (mismatches ${bad})`);
    }
});

gpuTest("StopMotion.identityAndKeys", async ({ device }) => {
    const scene = makeScene(device);
    const g = await runGraph(device, scene, [["StyleAovs", "S"], ["StopMotion", "M", { jitterPx: 0, flicker: 0, time: 5 / kFps, nodeHolds: {} }]], kEdges, kOuts);
    const [dst, src] = await readAll(device, g, kOuts);
    const same = bitwiseEqual(dst!, src!);
    log(`StopMotion identity (jitterPx 0, flicker 0): dst == src bitwise ${same}`);
    expectEq(same, true, "identity is bitwise");
    // Keys: jitterHold 2 / flickerHold 2 -> frames 2 and 3 equal, frame 4 differs.
    const gk = await runGraph(device, scene, [["StyleAovs", "S"], ["StopMotion", "M", { jitterPx: 3, flicker: 0.2, time: 2 / kFps }]], kEdges, kOuts);
    const frames: Float32Array[] = [];
    for (const f of [2, 3, 4]) {
        gk.getPass("M")!.setProperties(new Properties({ time: f / kFps }));
        gk.execute(device.renderContext);
        frames.push((await readAll(device, gk, ["M.dst"]))[0]!);
    }
    const held = bitwiseEqual(frames[0]!, frames[1]!), next = !bitwiseEqual(frames[1]!, frames[2]!);
    log(`StopMotion keys (on twos): frame 3 == frame 2 ${held}, frame 4 differs ${next}`);
    expectEq(held && next, true, "jitter/flicker step on key boundaries only");
});

// ---- animated scene (two sliding boxes on nodes 0 and 1) ----

function quad(c: V3[], n: V3, materialID: number, nodeID?: number) {
    const uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
    const t = new float4(1, 0, 0, 1);
    return {
        vertices: c.map((p, i) => ({ position: new float3(...p), normal: new float3(...n), tangent: t, texCrd: uv[i]! })),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        materialID,
        ...(nodeID !== undefined ? { nodeID } : {}),
    };
}
function box(min: V3, max: V3, materialID: number, nodeID: number) {
    const [x0, y0, z0] = min, [x1, y1, z1] = max;
    return [
        quad([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0, 1, 0], materialID, nodeID),
        quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1], materialID, nodeID),
        quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0, 0, -1], materialID, nodeID),
        quad([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1, 0, 0], materialID, nodeID),
        quad([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1, 0, 0], materialID, nodeID),
    ];
}
const slide = (nodeID: number, from: V3, to: V3): AnimationChannel => ({
    nodeID, path: "translation", interp: "LINEAR", times: new Float32Array([0, 1]), values: new Float32Array([...from, ...to]),
});
const kAnimScenes = new WeakMap<Device, Scene>();
/** Box A (0.9 high) on node 0, box B (1.4 high, the only geometry above y = 1) on node 1; both slide over 1 s. */
function makeAnimScene(device: Device): Scene {
    let scene = kAnimScenes.get(device);
    if (!scene) {
        const s = 6;
        const nodes: SceneNode[] = [
            { parent: -1, t: new float3(-1.4, 0, 0.2), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
            { parent: -1, t: new float3(0.9, 0, -1.0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
        ];
        const meshes = [
            quad([[-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]], [0, 1, 0], 0),
            ...box([-0.45, 0, -0.45], [0.45, 0.9, 0.45], 1, 0),
            ...box([-0.3, 0, -0.3], [0.3, 1.4, 0.3], 2, 1),
        ];
        const channels = [slide(0, [-1.4, 0, 0.2], [0.6, 0, 0.7]), slide(1, [0.9, 0, -1.0], [1.7, 0, -0.1])];
        const mats = ([[0.6, 0.65, 0.6], [0.85, 0.3, 0.1], [0.15, 0.35, 0.85]] as V3[]).map((c) => ({ basic: { baseColor: new float4(c[0], c[1], c[2], 1) } }));
        scene = new Scene(device, meshes, mats, [{ type: LightType.Directional, dirW: new float3(0.2, -1, 0.3), intensity: new float3(2, 2, 2) }],
            undefined, [], nodes, channels);
        kAnimScenes.set(device, scene);
    }
    scene.camera.setPosition(new float3(0.3, 4.0, 5.2));
    scene.camera.setTarget(new float3(0, 0.3, -0.3));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
    return scene;
}

/** Renders frames `ks` (t = k / fps) on one graph; the pass time follows the frame. */
async function clip(device: Device, scene: Scene, props: Record<string, unknown>, ks: number[]): Promise<{ pos: Float32Array[]; dst: Float32Array[] }> {
    scene.animate(ks[0]! / kFps);
    const g = await runGraph(device, scene, [["StyleAovs", "S"], ["StopMotion", "M", { ...props, time: ks[0]! / kFps }]], kEdges, kOuts);
    const pos: Float32Array[] = [], dst: Float32Array[] = [];
    for (const k of ks) {
        scene.animate(k / kFps);
        g.getPass("M")!.setProperties(new Properties({ time: k / kFps }));
        g.execute(device.renderContext);
        const [d, p] = await readAll(device, g, ["M.dst", "G.posW"]);
        dst.push(d!);
        pos.push(p!);
    }
    return { pos, dst };
}

gpuTest("StopMotion.perObjectFrameRate", async ({ device }) => {
    const scene = makeAnimScene(device);
    const ks = [0, 1, 2, 3, 4, 5, 6, 7];
    scene.setAnimationTimeQuantization(null);
    const full = await clip(device, scene, {}, ks);
    scene.setAnimationTimeQuantization({ 0: { hold: 2 / kFps } });
    const direct = await clip(device, scene, {}, ks);
    scene.setAnimationTimeQuantization(null);
    const viaPass = await clip(device, scene, { nodeHolds: { 0: 2 } }, ks);
    const q = scene.getAnimationTimeQuantization();
    expectEq(q.size === 1 && Math.abs(q.get(0)!.hold - 2 / kFps) < 1e-12, true, "pass applied { 0: 2 frames }");
    // An empty nodeHolds leaves the scene's quantisation untouched.
    await clip(device, scene, { nodeHolds: {} }, [0]);
    expectEq(scene.getAnimationTimeQuantization().size, 1, "empty nodeHolds leaves the quantisation as it was");
    scene.setAnimationTimeQuantization(null);
    let fg = 0, maxY = -1e9, moving = 0;
    for (let i = 0; i < kSize * kSize; i++) if (full.pos[0]![i * 4 + 3] !== 0) { fg++; maxY = Math.max(maxY, full.pos[0]![i * 4 + 1]!); }
    for (let k = 1; k < ks.length; k++) if (!bitwiseEqual(full.pos[k]!, full.pos[k - 1]!)) moving++;
    log(`StopMotion animated scene: geometry px ${fg}, max y ${maxY.toFixed(3)}, unquantised frames that move ${moving}/${ks.length - 1}`);
    expectEq(moving, ks.length - 1, "the animated scene moves on every frame");
    let samePass = 0, bBad = 0, bN = 0, aDiff = 0;
    for (let k = 0; k < ks.length; k++) {
        if (bitwiseEqual(viaPass.pos[k]!, direct.pos[k]!)) samePass++;
        const a = viaPass.pos[k]!, b = full.pos[k]!;
        for (let i = 0; i < kSize * kSize; i++) {
            const onB = (p: Float32Array) => p[i * 4 + 3] !== 0 && p[i * 4 + 1]! > 1.0;
            if (onB(a) && onB(b)) { bN++; if (a[i * 4] !== b[i * 4] || a[i * 4 + 2] !== b[i * 4 + 2]) bBad++; }
        }
        if (k % 2 === 1) {
            if (!bitwiseEqual(a, b)) aDiff++;
        }
    }
    log(`StopMotion per-object rate: frames pass == direct API ${samePass}/${ks.length}; box B px ${bN}, differing from unquantised ${bBad}; odd frames differing from unquantised (node 0 held) ${aDiff}/4`);
    expectEq(samePass, ks.length, "nodeHolds == Scene.setAnimationTimeQuantization bitwise");
    expectEq(bN > 100 && bBad === 0, true, "node 1 animates every frame as without holds");
    expectEq(aDiff, 4, "node 0 holds on twos (odd frames show the previous pose)");
});

gpuTest("StopMotion.coldVsConsecutive", async ({ device }) => {
    const scene = makeAnimScene(device);
    const props = { nodeHolds: { 0: 2, 1: 3 }, jitterPx: 2, jitterHold: 2, flicker: 0.15, seed: 5 };
    const cons = await clip(device, scene, props, [1, 3, 5]);
    const cold = await clip(device, scene, props, [5]);
    const control = await clip(device, scene, props, [6]);
    scene.setAnimationTimeQuantization(null);
    const same = bitwiseEqual(cons.dst[2]!, cold.dst[0]!), differs = !bitwiseEqual(control.dst[0]!, cold.dst[0]!);
    log(`StopMotion cold vs consecutive (frame 5 after 1, 3): bitwise ${same}; control frame 6 differs ${differs}`);
    expectEq(same, true, "cold == consecutive bitwise");
    expectEq(differs, true, "control frame differs");
});
