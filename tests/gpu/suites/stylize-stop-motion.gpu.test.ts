/**
 * StopMotion (packages/render-passes/src/Stylize/StopMotion.ts):
 *  - CPU reference: per-object jitter (with and without posW ordering, all ids and an id subset) and flicker gain against a
 *    CPU evaluation of the same gather over the same readbacks (exact: the pass only copies pixels and scales by a gain);
 *  - identity: jitterPx 0 and flicker 0 return src bit for bit; frames inside one key are bitwise equal, the next key differs;
 * The animated-scene tests (per-object frame rate, cold vs consecutive) are in anim-stop-motion.gpu.test.ts.
 */

import { Properties } from "@web-falcor/falcor";
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
