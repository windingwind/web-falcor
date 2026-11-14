/**
 * WorldRevealMask against a CPU reference computed from the same G-buffer / StyleAovs readbacks (floor + two boxes):
 *  - parity: sphere (soft edge), plane with value noise and edge glow, object-space box (hard edge, inverted), a swept
 *    sphere with an "except" id filter; dst and the mask output agree with the reference;
 *  - identity: strength 0 and an empty region return `a` bit for bit;
 *  - sweep / no state: the revealed area grows with `time`; a graph that rendered other times first renders a given time
 *    bitwise like a fresh graph (setProperties({ time }) per frame).
 */

import { Properties } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kSize, log, makeScene, objectClass, px, readAll, runGraph, smoothstep, type V3 } from "./stylize-helpers.js";

const kEdges: [string, string][] = [
    ["G.vbuffer", "S.vbuffer"], ["G.posW", "S.posW"], ["G.normW", "S.normW"],
    ["G.diffuseOpacity", "R.a"], ["S.idColor", "R.b"], ["G.posW", "R.posW"], ["S.posObj", "R.posObj"], ["S.ids", "R.ids"],
];
const kOuts = ["R.dst", "R.mask", "G.diffuseOpacity", "S.idColor", "G.posW", "S.posObj", "S.ids"];

// CPU replica of the shader hash and noise (uint32 arithmetic via Math.imul).
const pcg = (v: number): number => {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
};
const latticeHash = (x: number, y: number, z: number, seed: number) =>
    (pcg((x >>> 0) + pcg((y >>> 0) + pcg((z >>> 0) + pcg(seed)) >>> 0) >>> 0) >>> 8) / 16777216;
function valueNoise(p: V3, seed: number): number {
    const c = p.map(Math.floor) as V3;
    const f = p.map((x, i) => x - c[i]!) as V3;
    const u = f.map((t) => t * t * (3 - 2 * t)) as V3;
    const h = (dx: number, dy: number, dz: number) => latticeHash(c[0] + dx, c[1] + dy, c[2] + dz, seed);
    const l = (a: number, b: number, t: number) => a + (b - a) * t;
    return l(l(l(h(0, 0, 0), h(1, 0, 0), u[0]), l(h(0, 1, 0), h(1, 1, 0), u[0]), u[1]),
        l(l(h(0, 0, 1), h(1, 0, 1), u[0]), l(h(0, 1, 1), h(1, 1, 1), u[0]), u[1]), u[2]);
}

type P = Record<string, unknown>;
const get = <T>(p: P, k: string, d: T): T => (p[k] ?? d) as T;

/** Reference mask (before strength) and signed distance for one pixel; null distance for background. */
function refMask(p: P, pos: V3, bg: boolean, id: number): { m: number; d: number | null } {
    if (bg) return { m: get(p, "background", 0), d: null };
    const sweep = get(p, "speed", 0) * get(p, "time", 0);
    const shape = get(p, "shape", "sphere");
    const c = get<V3>(p, "center", [0, 0, 0]);
    let d: number;
    if (shape === "sphere") d = Math.hypot(pos[0] - c[0], pos[1] - c[1], pos[2] - c[2]) - (get(p, "radius", 1) + sweep);
    else if (shape === "plane") {
        const n = get<V3>(p, "normal", [0, 1, 0]);
        const l = Math.hypot(...n);
        d = (pos[0] * n[0] + pos[1] * n[1] + pos[2] * n[2]) / l - (get(p, "offset", 0) + sweep);
    } else {
        const he = get<V3>(p, "halfExtents", [1, 1, 1]);
        const q = [0, 1, 2].map((i) => Math.abs(pos[i]! - c[i]!) - (he[i]! + sweep));
        d = Math.hypot(...q.map((x) => Math.max(x, 0))) + Math.min(Math.max(q[0]!, q[1]!, q[2]!), 0);
    }
    const amp = get(p, "noiseAmp", 0);
    if (amp !== 0) {
        const s = get(p, "noiseScale", 1);
        d += amp * (2 * valueNoise([pos[0] * s, pos[1] * s, pos[2] * s], get(p, "seed", 0)) - 1);
    }
    const w = get(p, "edgeWidth", 0.1);
    let m = w > 0 ? 1 - smoothstep(-0.5 * w, 0.5 * w, d) : d < 0 ? 1 : 0;
    if (get(p, "invert", false)) m = 1 - m;
    const mode = get(p, "idMode", "none");
    const listed = get<number[]>(p, "ids", []).includes(id);
    if ((mode === "only" && !listed) || (mode === "except" && listed)) m = 0;
    return { m, d };
}

gpuTest("WorldRevealMask.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    // Instance id of the floor (for the id filter), from a first render.
    const g0 = await runGraph(device, scene, [["StyleAovs", "S"], ["WorldRevealMask", "R"]], kEdges, kOuts);
    const [, , , , pos0, , ids0] = await readAll(device, g0, kOuts);
    const floorIds = new Set<number>();
    for (let i = 0; i < kSize * kSize; i++) if (pos0[i * 4 + 3] !== 0 && objectClass(pos0[i * 4]!, pos0[i * 4 + 1]!, pos0[i * 4 + 2]!) === 0) floorIds.add(ids0[i * 4]!);
    expectEq(floorIds.size, 1, "floor is one instance");
    const floorId = [...floorIds][0]!;
    const cases: [string, P][] = [
        ["sphere soft", { shape: "sphere", center: [-0.4, 0.5, -0.3], radius: 1.3, edgeWidth: 0.4 }],
        ["plane noise glow", { shape: "plane", normal: [1, 0.3, 0], offset: 0.2, noiseAmp: 0.35, noiseScale: 3, seed: 7, edgeWidth: 0.25, edgeIntensity: 0.8, edgeColour: [1, 0.5, 0.1] }],
        ["box object hard invert", { shape: "box", space: "object", center: [0.7, 0.8, -0.8], halfExtents: [0.5, 0.6, 0.5], edgeWidth: 0, invert: true, background: 1 }],
        ["sweep except floor", { shape: "sphere", center: [0, 0.4, -0.3], radius: 0.2, speed: 0.6, time: 1.5, edgeWidth: 0.2, idMode: "except", ids: [floorId], strength: 0.75 }],
    ];
    let worst = 0;
    for (const [name, p] of cases) {
        const g = await runGraph(device, scene, [["StyleAovs", "S"], ["WorldRevealMask", "R", p]], kEdges, kOuts);
        const [dst, mask, a, b, pos, pobj, ids] = await readAll(device, g, kOuts);
        const strength = get(p, "strength", 1), w = get(p, "edgeWidth", 0.1), ei = get(p, "edgeIntensity", 0);
        const ec = get<V3>(p, "edgeColour", [1, 0.6, 0.2]);
        let maxErr = 0, bad = 0, skipped = 0, inside = 0, n = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            const bg = pos[i * 4 + 3] === 0;
            const P3 = (get(p, "space", "world") === "object" ? px(pobj, i) : px(pos, i)).slice(0, 3) as V3;
            const r = refMask(p, P3, bg, ids[i * 4]!);
            // Hard edges: skip pixels whose distance is within f32 rounding of the threshold.
            if (r.d !== null && w === 0 && Math.abs(r.d) < 1e-4) { skipped++; continue; }
            n++;
            const m = r.m * strength;
            if (m > 0.5) inside++;
            const edge = !bg && w > 0 ? 4 * r.m * (1 - r.m) : 0;
            let e = Math.abs(mask[i * 4]! - m);
            for (let k = 0; k < 3; k++) {
                const av = a[i * 4 + k]!, bv = b[i * 4 + k]!;
                const ref = (m <= 0 ? av : m >= 1 ? bv : av + (bv - av) * m) + ei * strength * edge * ec[k]!;
                e = Math.max(e, Math.abs(dst[i * 4 + k]! - ref));
            }
            if (e > 2e-3) bad++;
            maxErr = Math.max(maxErr, e);
        }
        log(`WorldRevealMask[${name}]: px ${n}, inside ${inside}, skipped ${skipped}, max err ${maxErr.toExponential(2)}, > 2e-3: ${bad}`);
        expectEq(inside > 200 && inside < n - 200, true, `${name}: both sides visible`);
        expectEq(bad <= Math.ceil(0.001 * n), true, `${name}: dst/mask match the CPU reference (bad ${bad})`);
        worst = Math.max(worst, bad === 0 ? maxErr : 0);
    }
    log(`WorldRevealMask CPU parity: worst max err (cases without outliers) ${worst.toExponential(2)}`);
});

gpuTest("WorldRevealMask.identity", async ({ device }) => {
    const scene = makeScene(device);
    for (const [name, p] of [
        ["strength 0", { shape: "plane", normal: [1, 0, 0], noiseAmp: 0.5, edgeIntensity: 1, strength: 0, background: 1 }],
        ["empty region", { shape: "sphere", center: [50, 50, 50], radius: 0, edgeWidth: 0.1 }],
    ] as [string, P][]) {
        const g = await runGraph(device, scene, [["StyleAovs", "S"], ["WorldRevealMask", "R", p]], kEdges, kOuts);
        const [dst, , a] = await readAll(device, g, kOuts);
        const same = bitwiseEqual(dst, a);
        log(`WorldRevealMask identity [${name}]: dst == a bitwise ${same}`);
        expectEq(same, true, `${name}: dst == a bitwise`);
    }
});

gpuTest("WorldRevealMask.sweepIsStateless", async ({ device }) => {
    const scene = makeScene(device);
    const p = { shape: "sphere", center: [0, 0.4, -0.3], radius: 0, speed: 1, edgeWidth: 0.3, noiseAmp: 0.2, noiseScale: 2 };
    const times = [0.3, 0.9, 1.5, 2.4];
    // Consecutive: one graph, time set per frame.
    const g = await runGraph(device, scene, [["StyleAovs", "S"], ["WorldRevealMask", "R", { ...p, time: times[0] }]], kEdges, kOuts);
    const cover: number[] = [];
    let last: Float32Array | null = null;
    for (const t of times) {
        g.getPass("R")!.setProperties(new Properties({ time: t }));
        g.execute(device.renderContext);
        const [dst, mask] = await readAll(device, g, ["R.dst", "R.mask"]);
        let c = 0;
        for (let i = 0; i < kSize * kSize; i++) c += mask[i * 4]!;
        cover.push(c);
        if (t === 1.5) last = dst!;
    }
    log(`WorldRevealMask sweep coverage ${cover.map((c) => c.toFixed(0)).join(" < ")}`);
    for (let k = 1; k < cover.length; k++) expectEq(cover[k]! > cover[k - 1]!, true, `coverage grows at t = ${times[k]}`);
    // Cold: a fresh graph at t = 1.5.
    const gc = await runGraph(device, scene, [["StyleAovs", "S"], ["WorldRevealMask", "R", { ...p, time: 1.5 }]], kEdges, kOuts);
    const [cold] = await readAll(device, gc, ["R.dst"]);
    const same = bitwiseEqual(cold!, last!);
    log(`WorldRevealMask cold vs consecutive (t = 1.5 after 0.3, 0.9): bitwise ${same}`);
    expectEq(same, true, "cold == consecutive bitwise");
});
