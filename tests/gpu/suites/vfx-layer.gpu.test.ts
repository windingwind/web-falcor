/**
 * VfxLayer:
 *  - CPU parity: the GPU layer (colour, coverage, depth) of fire, electric-arc and plasma emitters (one fire partly hidden
 *    behind a box) equals a CPU evaluation of the same formulas, with a hard and a soft depth test;
 *  - identity: intensity 0 and an empty emitter list give an empty layer that DepthComposite passes through bit for bit;
 *  - animation: the layer changes with `time` and every emitter type is visible;
 *  - stateless: a frame rendered after other shots on one graph equals the same frame on a fresh graph, bit for bit,
 *    with or without a camera cut (the pass keeps no history).
 */

import { packVfxEmitters, type VfxEmitter } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, camInfo, kSize, log, makeScene, read, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, coverage } from "./fx-helpers.js";

const kFar = 3.402823466e38;

const kEmitters: VfxEmitter[] = [
    { type: "fire", position: [-0.7, 1.0, -0.1], size: 0.8, seed: 2, intensity: 3 },
    { type: "electric", position: [-0.2, 0.8, 0.4], target: [0.7, 1.6, -0.4], size: 0.02, amplitude: 0.15, strands: 3, seed: 5 },
    { type: "plasma", position: [0.2, 0.5, 0.9], size: 0.25, seed: 9 },
    { type: "fire", position: [0.7, 0.0, -1.7], size: 1.2, seed: 4, colour: [0.2, 0.5, 1] }, // partly behind box B
];

// ---- CPU mirror of VfxLayer.cs.slang (f64 arithmetic) ----
function pcg(v: number): number {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}
const lattice = (x: number, y: number, seed: number) => (pcg(((x >>> 0) + pcg(((y >>> 0) + pcg(seed)) >>> 0)) >>> 0) >>> 8) / 16777216;
const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
function vnoise(px: number, py: number, seed: number): number {
    const ix = Math.floor(px), iy = Math.floor(py);
    const fx = px - ix, fy = py - iy;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = lattice(ix, iy, seed), b = lattice(ix + 1, iy, seed), c = lattice(ix, iy + 1, seed), d = lattice(ix + 1, iy + 1, seed);
    return lerp(lerp(a, b, sx), lerp(c, d, sx), sy);
}
function fbm(px: number, py: number, seed: number): number {
    let sum = 0, amp = 0.5;
    for (let k = 0; k < 4; k++) {
        sum += amp * vnoise(px, py, (seed + k * 101) >>> 0);
        px *= 2; py *= 2; amp *= 0.5;
    }
    return sum / 0.9375;
}

interface Ctx { vp: number[][]; pos: V3; fwd: V3; ppu: number; time: number; maxR: number }
interface S { e: V3; a: number; depth: number }
const empty = (): S => ({ e: [0, 0, 0], a: 0, depth: kFar });
const vdepth = (c: Ctx, x: ArrayLike<number>) => (x[0]! - c.pos[0]) * c.fwd[0] + (x[1]! - c.pos[1]) * c.fwd[1] + (x[2]! - c.pos[2]) * c.fwd[2];
function project(c: Ctx, x: ArrayLike<number>): [number, number] | null {
    const h = c.vp.map((r) => r[0]! * x[0]! + r[1]! * x[1]! + r[2]! * x[2]! + r[3]!);
    if (h[3]! <= 0) return null;
    return [(h[0]! / h[3]! * 0.5 + 0.5) * kSize, (-h[1]! / h[3]! * 0.5 + 0.5) * kSize];
}

function fire(c: Ctx, p: [number, number], e: Float32Array): S {
    const s = empty();
    const anchor = [e[0]!, e[1]!, e[2]!], top = [0, 1, 2].map((k) => e[k]! + e[4 + k]! * e[7]!);
    const d = vdepth(c, anchor);
    const c0 = project(c, anchor), c1 = project(c, top);
    if (d <= 0 || !c0 || !c1) return s;
    const S = e[7]! * c.ppu / d;
    const ax = [c1[0] - c0[0], c1[1] - c0[1]], al = Math.hypot(ax[0]!, ax[1]!);
    const dir = al > 1e-4 ? [ax[0]! / al, ax[1]! / al] : [0, -1];
    const L = Math.max(al, 0.3 * S), halfW = Math.max(S * e[14]!, 0.5);
    const q = [p[0] - c0[0], p[1] - c0[1]];
    const v = (q[0]! * dir[0]! + q[1]! * dir[1]!) / L, u = (q[0]! * -dir[1]! + q[1]! * dir[0]!) / halfW;
    if (v < -0.2 || v > 1.3 || Math.abs(u) > 1.8) return s;
    const seed = e[12]!, t = c.time * e[13]!;
    const w = fbm(u * 1.5, v * 2 - 1.3 * t, seed);
    const n = fbm(u * 2.5 + 0.8 * w, v * 3 - 2 * t, seed + 7);
    const rw = Math.sqrt(sat(v * 3 + 0.35)) * Math.pow(sat(1 - v), 0.7);
    const hw = rw * lerp(1, 0.35 + 1.3 * n, sat(e[15]!));
    let dens = sat((hw - Math.abs(u)) / Math.max(0.3 * hw, 0.03)) * sat(1 + v * 6);
    dens = dens * dens * (3 - 2 * dens);
    const heat = dens * (1 - 0.65 * sat(v)) * (0.55 + 0.7 * n);
    const white = [1, 0.95, 0.8];
    s.e = [0, 1, 2].map((k) => e[11]! * (e[8 + k]! * sat(1.8 * heat) + white[k]! * sat(2.5 * heat - 1.4))) as V3;
    s.a = dens;
    s.depth = d;
    return s;
}

function arc(c: Ctx, p: [number, number], e: Float32Array): S {
    const s = empty();
    const a0 = [e[0]!, e[1]!, e[2]!], a1 = [e[4]!, e[5]!, e[6]!];
    const d0 = vdepth(c, a0), d1 = vdepth(c, a1);
    const c0 = project(c, a0), c1 = project(c, a1);
    if (d0 <= 0 || d1 <= 0 || !c0 || !c1) return s;
    const seg = [c1[0] - c0[0], c1[1] - c0[1]], L = Math.hypot(seg[0]!, seg[1]!);
    if (L < 1) return s;
    const tg = [seg[0]! / L, seg[1]! / L], nrm = [-tg[1]!, tg[0]!];
    const rel = [p[0] - c0[0], p[1] - c0[1]];
    const tr = (rel[0]! * tg[0]! + rel[1]! * tg[1]!) / L, tc = sat(tr);
    const dz = 1 / lerp(1 / d0, 1 / d1, tc);
    const w = Math.min(Math.max(e[7]! * c.ppu / dz, 0.75), c.maxR);
    const amp = e[14]! * L;
    if (Math.abs(rel[0]! * nrm[0]! + rel[1]! * nrm[1]!) > amp + 4 * w || tr < -4 * w / L || tr > 1 + 4 * w / L) return s;
    const seed = e[12]!, strands = Math.min(Math.max(Math.trunc(e[15]!), 1), 4), jump = Math.floor(c.time * e[13]!);
    let glow = 0, core = 0;
    for (let k = 0; k < strands; k++) {
        const n = fbm(tc * 5 + 13.7 * k, jump * 3.17 + 0.37 * k, seed + 31 * k);
        const disp = amp * Math.sin(Math.PI * tc) * (2 * n - 1) * (k === 0 ? 1 : 1.4);
        const cx = c0[0] + seg[0]! * tc + nrm[0]! * disp, cy = c0[1] + seg[1]! * tc + nrm[1]! * disp;
        const dist = Math.hypot(p[0] - cx, p[1] - cy);
        const bk = k === 0 ? 1 : 0.5;
        glow += bk * Math.pow(0.25 * w / Math.max(dist, 0.25 * w), 1.5) * sat(1 - dist / (4 * w));
        const cr = dist / (0.15 * w + 0.5);
        if (dist < 4 * w) core += bk * Math.exp(-cr * cr);
    }
    s.e = [0, 1, 2].map((k) => e[11]! * (e[8 + k]! * glow + 0.8 * core)) as V3;
    s.a = sat(glow);
    s.depth = dz;
    return s;
}

function orb(c: Ctx, p: [number, number], e: Float32Array): S {
    const s = empty();
    const a = [e[0]!, e[1]!, e[2]!];
    const d = vdepth(c, a), cc = project(c, a);
    if (d <= 0 || !cc) return s;
    const R = Math.max(e[7]! * c.ppu / d, 0.75);
    const o = [(p[0] - cc[0]) / R, (p[1] - cc[1]) / R], r = Math.hypot(o[0]!, o[1]!);
    const ext = Math.max(e[14]!, 0);
    if (r >= 1 + ext) return s;
    const seed = e[12]!, t = c.time * e[13]!;
    if (r < 1) {
        const z = Math.sqrt(1 - r * r);
        const n = fbm(o[0]! * 3 + 0.3 * t, o[1]! * 3 + 2 * z - 0.2 * t, seed);
        const f = Math.pow(1 - Math.abs(2 * n - 1), Math.max(e[15]!, 1));
        const rim = (1 - z) * (1 - z);
        s.e = [0, 1, 2].map((k) => e[11]! * (e[8 + k]! * (1.5 * f + 1.2 * rim + 0.6 * z ** 4) + 0.5 * f ** 3)) as V3;
        s.a = sat(0.35 + rim);
        s.depth = d - e[7]! * z;
    } else {
        const dv = [o[0]! / r, o[1]! / r];
        const g = 0.6 * Math.exp(-4 * (r - 1) / Math.max(ext, 1e-3)) * (0.7 + 0.6 * fbm(dv[0]! * 2, dv[1]! * 2 + 0.25 * t, seed + 3));
        s.e = [0, 1, 2].map((k) => e[11]! * e[8 + k]! * g) as V3;
        s.a = sat(g);
        s.depth = d;
    }
    return s;
}

function layerRef(scene: ReturnType<typeof makeScene>, posW: Float32Array, emitters: VfxEmitter[], time: number, intensity: number, softness: number) {
    const ci = camInfo(scene);
    const vpm = scene.camera.getData().viewProjMatNoJitter;
    const c: Ctx = { vp: [0, 1, 2, 3].map((r) => [vpm.get(r, 0), vpm.get(r, 1), vpm.get(r, 2), vpm.get(r, 3)]), pos: ci.pos, fwd: ci.fwd, ppu: (0.5 * kSize) / Math.tan(0.5 * ci.fovY), time, maxR: 64 };
    const packed = packVfxEmitters(emitters);
    const col = new Float64Array(kSize * kSize * 4), dep = new Float64Array(kSize * kSize).fill(kFar);
    for (let y = 0; y < kSize; y++) for (let x = 0; x < kSize; x++) {
        const k = y * kSize + x, p: [number, number] = [x + 0.5, y + 0.5];
        const sd = posW[k * 4 + 3] !== 0 ? vdepth(c, posW.subarray(k * 4, k * 4 + 3)) : kFar;
        let cov = 0;
        for (let i = 0; i < emitters.length; i++) {
            const e = packed.subarray(i * 16, i * 16 + 16);
            const s = e[3] === 0 ? fire(c, p, e) : e[3] === 1 ? arc(c, p, e) : orb(c, p, e);
            if (!(s.a > 0)) continue;
            let vis = s.depth < sd ? 1 : 0;
            if (softness > 0 && sd < kFar) vis = sat((sd - s.depth) / softness);
            if (vis <= 0) continue;
            for (let ch = 0; ch < 3; ch++) col[k * 4 + ch] += s.e[ch]! * vis;
            cov += s.a * vis;
            dep[k] = Math.min(dep[k]!, s.depth);
        }
        for (let ch = 0; ch < 3; ch++) col[k * 4 + ch] *= intensity;
        col[k * 4 + 3] = Math.min(cov, 1);
    }
    return { col, dep };
}

const vfxGraph = (device: Parameters<typeof buildGraph>[0], scene: Parameters<typeof buildGraph>[1], props: Record<string, unknown>) =>
    buildGraph(device, scene, [["VfxLayer", "V", { emitters: kEmitters, time: 1.37, ...props }], ["DepthComposite", "C", { alphaMode: "additive" }]],
        [["G.posW", "V.posW"], ["G.diffuseOpacity", "C.src"], ["G.posW", "C.posW"], ["V.color", "C.layer"], ["V.depth", "C.layerDepth"]],
        ["V.color", "V.depth", "C.dst", "G.diffuseOpacity"], kSize, kSize);

async function settle(device: Parameters<typeof buildGraph>[0], scene: Parameters<typeof buildGraph>[1]) {
    const g = buildGraph(device, scene, [], [], [], kSize, kSize);
    for (let i = 0; i < 5; i++) { g.execute(device.renderContext); if ((await coverage(device, g)) > 0.2) break; }
}

gpuTest("VfxLayer.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    await settle(device, scene);
    for (const props of [{ intensity: 1, softness: 0, time: 1.37 }, { intensity: 0.7, softness: 0.3, time: 4.05 }]) {
        const g = vfxGraph(device, scene, props);
        g.execute(device.renderContext);
        const [color, depth, pos] = [await read(device, g.getOutput("V.color")!), await read(device, g.getOutput("V.depth")!), await read(device, g.getOutput("G.posW")!)];
        const ref = layerRef(scene, pos, kEmitters, props.time, props.intensity, props.softness);
        let bad = 0, maxErr = 0, lit = 0, depthBad = 0;
        for (let k = 0; k < kSize * kSize; k++) {
            let e = 0;
            for (let ch = 0; ch < 4; ch++) {
                const r = ref.col[k * 4 + ch]!;
                e = Math.max(e, Math.abs(color[k * 4 + ch]! - r) / Math.max(1, Math.abs(r)));
            }
            if (ref.col[k * 4 + 3]! > 0.01) lit++;
            // Value-noise lattice cells (floor) and the arc's discrete jumps can flip at f32 vs f64 boundaries.
            if (e > 2e-3) bad++;
            else maxErr = Math.max(maxErr, e);
            const a = depth[k * 4]!, b = ref.dep[k]!;
            if (!(b < 1e38 ? Math.abs(a - b) <= 1e-4 * b : a >= 1e38)) depthBad++;
        }
        log(`VfxLayer parity ${JSON.stringify(props)}: lit px ${lit}, off-reference px ${bad}, max rel err ${maxErr.toExponential(2)}, depth mismatches ${depthBad}`);
        expectEq(lit > 600, true, "emitters visible");
        expectEq(bad <= 0.005 * kSize * kSize, true, "layer = CPU reference (<= 0.5 % boundary pixels)");
        expectEq(depthBad <= 0.005 * kSize * kSize, true, "layer depth = CPU reference");
    }
});

gpuTest("VfxLayer.identity", async ({ device }) => {
    const scene = makeScene(device);
    await settle(device, scene);
    for (const props of [{ intensity: 0 }, { emitters: [] }]) {
        const g = vfxGraph(device, scene, props);
        g.execute(device.renderContext);
        const [color, depth, dst, src] = [await read(device, g.getOutput("V.color")!), await read(device, g.getOutput("V.depth")!), await read(device, g.getOutput("C.dst")!), await read(device, g.getOutput("G.diffuseOpacity")!)];
        const emptyLayer = color.every((v) => v === 0) && [...depth].every((v, i) => (i % 4 === 0 ? v >= 1e38 : true));
        const same = bitwiseEqual(dst, src);
        log(`VfxLayer identity ${JSON.stringify(props)}: empty layer ${emptyLayer}, composite bitwise = src ${same}`);
        expectEq(emptyLayer && same, true, `identity ${JSON.stringify(props)}`);
    }
});

gpuTest("VfxLayer.animatedPerType", async ({ device }) => {
    const scene = makeScene(device);
    await settle(device, scene);
    const changed: Record<string, number> = {};
    for (const e of kEmitters.slice(0, 3)) {
        const imgs: Float32Array[] = [];
        for (const time of [1.0, 1.5]) {
            const g = vfxGraph(device, scene, { emitters: [e], time });
            g.execute(device.renderContext);
            imgs.push(await read(device, g.getOutput("V.color")!));
        }
        let lit = 0, diff = 0;
        for (let k = 0; k < kSize * kSize; k++) {
            if (imgs[0]![k * 4 + 3]! > 0.01) lit++;
            if (Math.abs(imgs[0]![k * 4]! - imgs[1]![k * 4]!) > 1e-3) diff++;
        }
        changed[e.type] = diff;
        log(`VfxLayer ${e.type}: lit px ${lit}, px changed between t = 1.0 and 1.5: ${diff}`);
        expectEq(lit > 50 && diff > 20, true, `${e.type} visible and animated`);
    }
});

gpuTest("VfxLayer.stateless", async ({ device }) => {
    const scene = makeScene(device);
    const res = await coldVsConsecutive(device, scene, () => vfxGraph(device, scene, {}), ["C.dst", "V.depth"], ["camera-reset", "none"]);
    for (const r of res) log(`VfxLayer cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
    for (const r of res) expectEq(r.equalFrames, r.frames, `cold = consecutive (${r.method}; stateless)`);
});
