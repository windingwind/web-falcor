/**
 * RainWetness, RainLayer and the rainy-night chain (lighting -> wetness -> SSR -> rain layer).
 *  1. RainWetness CPU parity: wet darkening, gloss, puddle mask (planar fBm, up-facing only), puddle F0/roughness/normal
 *     and analytic ripples against a CPU reference of every pixel of the test scene.
 *  2. RainWetness identity: wetness 0 + puddleCoverage 0 copy color/specRough/normW bit for bit.
 *  3. RainLayer CPU parity: drop positions at `time`, streak projection, line coverage, occlusion and the integer
 *     accumulation against a CPU splat of the same drops.
 *  4. RainLayer is stateless and time-driven: same time = same layer (same graph, fresh graph), other time differs;
 *     opacity 0 = empty layer = DepthComposite identity.
 *  5. Cold vs consecutive on the whole rainy-night graph (SSR temporal on): shot X on a fresh graph equals X after shots
 *     A and B on one graph with a camera reset, a prime render or resetGraphHistory in between (bit for bit); without
 *     a cut X differs (control: SSR's history); with SSR temporal off the chain is stateless.
 *  6. RainLayer sky occlusion (RayTraced): under the floor plane (the "roof") drops vanish, drops with open sky above
 *     them are drawn exactly as without occlusion (bit for bit wherever no roofed drop lands); parity against the CPU
 *     splat minus the roofed drops (camera above the floor: only drops ending inside a box or below the floor are
 *     roofed; camera below the floor: the floor is a roof); default layer hashes (defaultHash) for comparison with the
 *     pass before the option existed.
 */

import type { Device, RenderGraph, Scene } from "@web-falcor/falcor";
import { planarBasis } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kBoxA, kBoxB, kSize, log, makeScene, read, type PassDecl } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, kPoseA, setPose, warmUp } from "./fx-helpers.js";

const kFar = 3.402823466e38;
/** One scene for every test of this file (a second Scene on the same device renders empty until the CopyContext fix lands). */
const kSky: [number, number, number] = [0.3, 0.35, 0.45];
const f32 = Math.fround;

// ---------------------------------------------------------------------------------------------------------------- hash

function pcg(v: number): number {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

function hash2(cx: number, cy: number, salt: number, seed: number): number {
    const inner = pcg((Math.imul(cy >>> 0, 19349663) ^ pcg((Math.imul(seed, 83492791) + salt) >>> 0)) >>> 0);
    return (pcg((Math.imul(cx >>> 0, 73856093) ^ inner) >>> 0) >>> 8) / 16777216;
}

function valueNoise(x: number, y: number, salt: number, seed: number): number {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = hash2(ix, iy, salt, seed), b = hash2(ix + 1, iy, salt, seed), c = hash2(ix, iy + 1, salt, seed), d = hash2(ix + 1, iy + 1, salt, seed);
    const l0 = a + (b - a) * sx, l1 = c + (d - c) * sx;
    return l0 + (l1 - l0) * sy;
}

function fbm(x: number, y: number, seed: number): number {
    let v = 0, a = 0.5, n = 0;
    for (let o = 0; o < 4; o++) {
        v += a * valueNoise(x, y, o, seed);
        n += a;
        a *= 0.5;
        x = x * 2.03 + 17.1;
        y = y * 2.03 + 9.7;
    }
    return v / n;
}

const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const smooth = (e0: number, e1: number, x: number) => { const t = sat((x - e0) / Math.max(e1 - e0, 1e-6)); return t * t * (3 - 2 * t); };
const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const frac = (x: number) => x - Math.floor(x);

interface WetParams {
    wetness: number; sideWetness: number; darkening: number; gloss: number; puddleCoverage: number; puddleEdge: number; puddleScale: number;
    puddleUpMin: number; puddleRoughness: number; puddleSpecular: number; rippleStrength: number; rippleScale: number; rippleRate: number; time: number; seed: number;
}

function ripplesCpu(qx: number, qy: number, p: WetParams): [number, number, number] {
    const acc: [number, number, number] = [0, 0, 0];
    const cell = p.rippleScale;
    for (let layer = 0; layer < 2; layer++) {
        const gx = qx / cell + 0.37 * layer, gy = qy / cell + 0.71 * layer;
        const c0x = Math.floor(gx), c0y = Math.floor(gy);
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                const cx = c0x + dx, cy = c0y + dy, salt = 100 + 10 * layer;
                const ox = (gx - (cx + hash2(cx, cy, salt, p.seed))) * cell, oy = (gy - (cy + hash2(cx, cy, salt + 1, p.seed))) * cell;
                const tau = frac(p.time * p.rippleRate + hash2(cx, cy, salt + 2, p.seed));
                const d = Math.hypot(ox, oy), r = tau * cell, w = 0.08 * cell, k = (2 * Math.PI) / (0.12 * cell), x = d - r;
                const env = Math.exp(-(x * x) / (w * w)) * (1 - tau);
                const h = Math.sin(k * x) * env;
                const dh = (k * Math.cos(k * x) - (2 * x) / (w * w) * Math.sin(k * x)) * env;
                const s = 0.08 * cell;
                acc[0] += h * s;
                if (d > 1e-6) { acc[1] += dh * (ox / d) * s; acc[2] += dh * (oy / d) * s; }
            }
        }
    }
    return acc;
}

/** CPU reference of RainWetness for one pixel: [dst rgba, specRough, normW, wetness]. */
function wetCpu(col: ArrayLike<number>, diff: ArrayLike<number>, spec: ArrayLike<number>, pos: ArrayLike<number>, nw: ArrayLike<number>, sr: ArrayLike<number>, p: WetParams, basis: ReturnType<typeof planarBasis>): number[][] {
    let wet = 0, puddle = 0, N = [0, 0, 1], qx = 0, qy = 0;
    if (pos[3] !== 0 && dot3(nw, nw) > 0) {
        const l = Math.sqrt(dot3(nw, nw));
        N = [nw[0]! / l, nw[1]! / l, nw[2]! / l];
        const u = dot3(N, basis.up);
        const facing = u >= 0 ? p.sideWetness + (1 - p.sideWetness) * smooth(0, 1, u) : p.sideWetness * sat(1 + u);
        wet = sat(p.wetness * facing);
        qx = dot3(pos, basis.t1); qy = dot3(pos, basis.t2);
        if (p.puddleCoverage > 0) {
            const lo = 1 - p.puddleCoverage;
            puddle = smooth(p.puddleUpMin, 1, u) * smooth(lo, lo + p.puddleEdge, fbm(qx * p.puddleScale, qy * p.puddleScale, p.seed));
        }
    }
    const w = Math.max(wet, puddle);
    if (w === 0) return [[...Array.from(col)], [...Array.from(sr)], [...Array.from(nw)], [0, 0, 0, 1]];
    const rgb = [0, 1, 2].map((c) => Math.max(col[c]! - p.darkening * w * diff[c]! - p.puddleSpecular * puddle * spec[c]!, 0));
    const rough = sr[3]! * (1 - p.gloss * wet) * (1 - puddle) + p.puddleRoughness * puddle;
    const f0 = [0, 1, 2].map((c) => sr[c]! + (0.02 - sr[c]!) * puddle);
    let n2 = [0, 1, 2].map((c) => N[c]! + (basis.up[c]! - N[c]!) * puddle);
    let rip: [number, number, number] = [0, 0, 0];
    if (puddle > 0 && p.rippleStrength > 0) {
        rip = ripplesCpu(qx, qy, p);
        n2 = [0, 1, 2].map((c) => n2[c]! - p.rippleStrength * puddle * (rip[1] * basis.t1[c]! + rip[2] * basis.t2[c]!));
    }
    const nl = Math.hypot(n2[0]!, n2[1]!, n2[2]!);
    return [[...rgb, col[3]!], [...f0, rough], [n2[0]! / nl, n2[1]! / nl, n2[2]! / nl, nw[3]!], [wet, puddle, rip[0], 1]];
}

const kWet: WetParams = {
    wetness: 0.8, sideWetness: 0.3, darkening: 0.45, gloss: 0.6, puddleCoverage: 0.45, puddleEdge: 0.08, puddleScale: 0.6,
    puddleUpMin: 0.9, puddleRoughness: 0.02, puddleSpecular: 0.8, rippleStrength: 1, rippleScale: 0.5, rippleRate: 1.5, time: 1.3, seed: 3,
};

const kWetEdges: [string, string][] = [
    ["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"],
    ["L.color", "W.color"], ["L.diffuse", "W.diffuse"], ["L.specular", "W.specular"], ["G.posW", "W.posW"], ["G.normW", "W.normW"], ["G.specRough", "W.specRough"],
];

function wetGraph(device: Device, scene: Scene, props: Partial<WetParams>, outputs: string[]): RenderGraph {
    return buildGraph(device, scene, [["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: "None" }], ["RainWetness", "W", props]], kWetEdges, outputs, kSize, kSize);
}

gpuTest("RainWetness.cpuParity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    const ins = ["L.color", "L.diffuse", "L.specular", "G.posW", "G.normW", "G.specRough"];
    const outs = ["W.dst", "W.specRoughOut", "W.normWOut", "W.wetness"];
    const g = wetGraph(device, scene, kWet, [...ins, ...outs]);
    g.execute(device.renderContext);
    const b: Float32Array[] = [];
    for (const o of [...ins, ...outs]) b.push(await read(device, g.getOutput(o)!));
    const [col, dif, spe, pos, nw, sr, dst, srO, nO, wt] = b;
    const basis = planarBasis([0, 1, 0]);
    let maxErr = 0, off = 0, geo = 0, puddlePx = 0, puddleNotUp = 0, sideOk = 0, sidePx = 0, rippled = 0;
    const tol = 2e-3;
    for (let i = 0; i < kSize * kSize; i++) {
        const s = (a: Float32Array) => a.subarray(i * 4, i * 4 + 4);
        const ref = wetCpu(s(col!), s(dif!), s(spe!), s(pos!), s(nw!), s(sr!), kWet, basis);
        const gpu = [s(dst!), s(srO!), s(nO!), s(wt!)];
        let e = 0;
        for (let k = 0; k < 4; k++) for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(gpu[k]![c]! - ref[k]![c]!) / Math.max(1, Math.abs(ref[k]![c]!)));
        if (e > tol) off++;
        else maxErr = Math.max(maxErr, e);
        if (pos![i * 4 + 3] === 0) continue;
        geo++;
        const n = s(nw!), l = Math.hypot(n[0]!, n[1]!, n[2]!), u = n[1]! / l;
        if (wt![i * 4 + 1]! > 0) { puddlePx++; if (u < kWet.puddleUpMin) puddleNotUp++; }
        if (wt![i * 4 + 2] !== 0) rippled++;
        if (Math.abs(u) < 1e-3) { sidePx++; if (Math.abs(wt![i * 4]! - kWet.wetness * kWet.sideWetness) < 1e-5) sideOk++; }
    }
    log(`RainWetness parity: ${geo} geometry px, off-ref (> ${tol}) ${off}, max rel err of the rest ${maxErr.toExponential(2)}; puddle px ${puddlePx} (not up-facing ${puddleNotUp}), rippled ${rippled}; vertical px ${sidePx}, wet = wetness x sideWetness ${sideOk}`);
    expectEq(off <= Math.ceil(0.001 * kSize * kSize), true, "<= 0.1 % of the pixels off the CPU reference");
    expectEq(maxErr < 1e-3, true, "max rel err < 1e-3");
    expectEq(puddlePx > 200 && puddleNotUp === 0, true, "puddles exist and only on up-facing surfaces");
    expectEq(rippled > 50, true, "ripples on the puddles");
    expectEq(sidePx > 100 && sideOk === sidePx, true, "vertical surfaces get wetness x sideWetness");
});

gpuTest("RainWetness.identity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    const outs = ["W.dst", "W.specRoughOut", "W.normWOut", "L.color", "G.specRough", "G.normW"];
    const res: boolean[] = [];
    for (const props of [{ ...kWet, wetness: 0, puddleCoverage: 0 }, { wetness: 0, puddleCoverage: 0 }]) {
        const g = wetGraph(device, scene, props, outs);
        g.execute(device.renderContext);
        const b: Float32Array[] = [];
        for (const o of outs) b.push(await read(device, g.getOutput(o)!));
        res.push(bitwiseEqual(b[0]!, b[3]!) && bitwiseEqual(b[1]!, b[4]!) && bitwiseEqual(b[2]!, b[5]!));
    }
    log(`RainWetness identity (wetness 0, puddleCoverage 0; with other params set / defaults): ${res.join(", ")}`);
    expectEq(res.every((x) => x), true, "dst/specRoughOut/normWOut = color/specRough/normW bit for bit");
});

// ---------------------------------------------------------------------------------------------------------- RainLayer

interface RainParams { count: number; seed: number; box: number; speed: number; speedJitter: number; wind: number[]; exposure: number; width: number; maxLengthPx: number; colour: number[]; intensity: number; opacity: number; near: number; time: number }

function rnd(i: number, salt: number, seed: number): number {
    return (pcg((i ^ pcg((Math.imul(salt, 7919) + pcg(seed)) >>> 0)) >>> 0) >>> 8) / 16777216;
}

/** CPU splat of RainLayer (same algorithm, doubles; accumulation in the same fixed point). */
function rainCpu(scene: Scene, p: RainParams, pos: Float32Array, skip?: (head: number[]) => boolean): { rgba: Float32Array; depth: Float32Array; drawn: number } {
    const N = kSize;
    const cam = scene.camera;
    const cp = cam.getPosition(), ct = cam.getTarget();
    const C = [cp.x, cp.y, cp.z];
    let fwd = [ct.x - cp.x, ct.y - cp.y, ct.z - cp.z];
    const fl = Math.hypot(fwd[0]!, fwd[1]!, fwd[2]!);
    fwd = fwd.map((x) => x / fl);
    const vp = cam.getViewProjMatrix();
    const clip = (x: number[]) => [0, 1, 2, 3].map((r) => vp.get(r, 0) * x[0]! + vp.get(r, 1) * x[1]! + vp.get(r, 2) * x[2]! + vp.get(r, 3));
    const toPx = (c: number[]) => [(c[0]! / c[3]! * 0.5 + 0.5) * N, (-c[1]! / c[3]! * 0.5 + 0.5) * N];
    const vd = (x: number[]) => (x[0]! - C[0]!) * fwd[0]! + (x[1]! - C[1]!) * fwd[1]! + (x[2]! - C[2]!) * fwd[2]!;
    const sceneD = (qx: number, qy: number) => { const k = (qy * N + qx) * 4; return pos[k + 3] !== 0 ? vd([pos[k]!, pos[k + 1]!, pos[k + 2]!]) : kFar; };
    const acc = new Float64Array(N * N * 4), depth = new Float32Array(N * N).fill(kFar);
    const half = Math.max(p.width, 0.1) * 0.5;
    let drawn = 0;
    for (let i = 0; i < p.count; i++) {
        const base = [0, 1, 2].map((s) => rnd(i, s, p.seed) * p.box);
        const sp = p.speed * (1 + p.speedJitter * (rnd(i, 3, p.seed) - 0.5));
        const vel = [p.wind[0]!, -sp + p.wind[1]!, p.wind[2]!];
        const head = [0, 1, 2].map((c) => { const o = C[c]! - 0.5 * p.box; let q = base[c]! + vel[c]! * p.time - o; q -= p.box * Math.floor(q / p.box); return o + q; });
        const tail = [0, 1, 2].map((c) => head[c]! - vel[c]! * p.exposure);
        const dh = vd(head), dtl = vd(tail);
        if (dh < p.near || dtl < p.near) continue;
        const rel = Math.max(...[0, 1, 2].map((c) => Math.abs(head[c]! - C[c]!) / (0.5 * p.box)));
        const fade = sat((1 - rel) * 4);
        if (fade <= 0) continue;
        if (skip && skip(head)) continue;
        const a = toPx(clip(head));
        let b = toPx(clip(tail));
        let ab = [b[0]! - a[0]!, b[1]! - a[1]!];
        let len = Math.hypot(ab[0]!, ab[1]!), dTail = dtl;
        if (len > p.maxLengthPx) { const s = p.maxLengthPx / len; b = [a[0]! + ab[0]! * s, a[1]! + ab[1]! * s]; dTail = dh + (dtl - dh) * s; ab = [b[0]! - a[0]!, b[1]! - a[1]!]; len = p.maxLengthPx; }
        if ((Math.max(a[0]!, b[0]!) < -4 && Math.max(a[1]!, b[1]!) < -4) || Math.min(a[0]!, b[0]!) > N + 4 || Math.min(a[1]!, b[1]!) > N + 4) continue;
        drawn++;
        const w0 = p.opacity * fade;
        const xMajor = Math.abs(ab[0]!) >= Math.abs(ab[1]!);
        const dir = len > 1e-6 ? [ab[0]! / len, ab[1]! / len] : [0, 1];
        const nrm = [-dir[1]!, dir[0]!];
        const a0 = xMajor ? a[0]! : a[1]!, a1 = xMajor ? b[0]! : b[1]!;
        const m0 = Math.floor(Math.min(a0, a1)), m1 = Math.floor(Math.max(a0, a1));
        const r = Math.ceil(half) + 1;
        const span = xMajor ? ab[0]! : ab[1]!;
        for (let m = m0; m <= m1; m++) {
            const t = Math.abs(span) > 1e-6 ? sat((m + 0.5 - a0) / span) : 0;
            const c = [a[0]! + ab[0]! * t, a[1]! + ab[1]! * t];
            const minorC = Math.floor(xMajor ? c[1]! : c[0]!);
            for (let n = minorC - r; n <= minorC + r; n++) {
                const qx = xMajor ? m : n, qy = xMajor ? n : m;
                const pcx = qx + 0.5 - a[0]!, pcy = qy + 0.5 - a[1]!;
                const tt = len > 1e-6 ? sat((pcx * dir[0]! + pcy * dir[1]!) / len) : 0;
                const dist = Math.abs(pcx * nrm[0]! + pcy * nrm[1]!);
                const along = pcx * dir[0]! + pcy * dir[1]!;
                const cov = sat(half + 0.5 - dist) * sat(1 + Math.min(along, len - along));
                const wgt = w0 * cov * (0.35 + 0.65 * (1 - tt));
                if (qx < 0 || qy < 0 || qx >= N || qy >= N || wgt <= 0) continue;
                const d = dh + (dTail - dh) * tt;
                if (d >= sceneD(qx, qy)) continue;
                const k = qy * N + qx;
                for (let ch = 0; ch < 3; ch++) acc[k * 4 + ch]! += Math.floor(p.colour[ch]! * wgt * p.intensity * 4096 + 0.5);
                acc[k * 4 + 3]! += Math.floor(wgt * 4096 + 0.5);
                depth[k] = Math.min(depth[k]!, f32(d));
            }
        }
    }
    const rgba = new Float32Array(N * N * 4);
    for (let k = 0; k < N * N; k++) {
        for (let ch = 0; ch < 3; ch++) rgba[k * 4 + ch] = acc[k * 4 + ch]! / 4096;
        rgba[k * 4 + 3] = Math.min(acc[k * 4 + 3]! / 4096, 1);
    }
    return { rgba, depth, drawn };
}

const kRain: RainParams = { count: 600, seed: 5, box: 6, speed: 8, speedJitter: 0.3, wind: [1.5, 0, -0.5], exposure: 0.03, width: 1.5, maxLengthPx: 40, colour: [0.7, 0.8, 1.0], intensity: 1.2, opacity: 0.5, near: 0.2, time: 0.7 };

function rainGraph(device: Device, scene: Scene, props: Record<string, unknown>, outputs: string[], dc = false): RenderGraph {
    const passes: PassDecl[] = [["RainLayer", "R", props]];
    const edges: [string, string][] = [["G.posW", "R.posW"]];
    if (dc) {
        passes.unshift(["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: "None" }]);
        passes.push(["DepthComposite", "D", { alphaMode: "premultiplied" }]);
        edges.push(["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["L.color", "D.src"], ["G.posW", "D.posW"], ["R.color", "D.layer"], ["R.depth", "D.layerDepth"]);
    }
    return buildGraph(device, scene, passes, edges, outputs, kSize, kSize);
}

gpuTest("RainLayer.cpuParity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    const g = rainGraph(device, scene, kRain, ["R.color", "R.depth"]);
    g.execute(device.renderContext);
    const col = await read(device, g.getOutput("R.color")!), dep = await read(device, g.getOutput("R.depth")!), pos = await read(device, g.getOutput("G.posW")!);
    const ref = rainCpu(scene, kRain, pos);
    let off = 0, lit = 0, maxErr = 0, depthOff = 0, occl = 0;
    for (let k = 0; k < kSize * kSize; k++) {
        let e = 0;
        for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(col[k * 4 + c]! - ref.rgba[k * 4 + c]!));
        if (col[k * 4 + 3]! > 0 || ref.rgba[k * 4 + 3]! > 0) lit++;
        if (e > 2e-3) off++;
        else maxErr = Math.max(maxErr, e);
        const gd = dep[k * 4]!, rd = ref.depth[k]!;
        if ((gd === kFar) !== (rd === kFar) || (gd !== kFar && Math.abs(gd - rd) > 1e-4 * Math.max(1, rd))) depthOff++;
        if (col[k * 4 + 3]! > 0 && pos[k * 4 + 3] !== 0) {
            const cam = scene.camera.getPosition(), t = scene.camera.getTarget();
            const f = [t.x - cam.x, t.y - cam.y, t.z - cam.z], fl = Math.hypot(f[0]!, f[1]!, f[2]!);
            const sd = ((pos[k * 4]! - cam.x) * f[0]! + (pos[k * 4 + 1]! - cam.y) * f[1]! + (pos[k * 4 + 2]! - cam.z) * f[2]!) / fl;
            if (gd > sd * (1 + 1e-5)) occl++;
        }
    }
    log(`RainLayer parity: drops drawn ${ref.drawn}/${kRain.count}, layer px ${lit}, off-ref (> 2e-3) ${off}, max err of the rest ${maxErr.toExponential(2)}, depth mismatches ${depthOff}, streak px behind the scene ${occl}`);
    expectEq(ref.drawn > 100 && lit > 1000, true, "enough streaks on screen");
    expectEq(off <= Math.ceil(0.005 * lit), true, "<= 0.5 % of the layer pixels off the CPU splat");
    expectEq(depthOff <= Math.ceil(0.005 * lit), true, "nearest-streak depth matches");
    expectEq(occl, 0, "no streak pixel behind the scene surface");
});

gpuTest("RainLayer.statelessTimeDrivenAndIdentity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    const outs = ["R.color", "R.depth"];
    const g = rainGraph(device, scene, kRain, outs);
    const frames: Float32Array[] = [];
    for (let f = 0; f < 3; f++) { g.execute(device.renderContext); frames.push(await read(device, g.getOutput("R.color")!)); }
    const g2 = rainGraph(device, scene, kRain, outs);
    g2.execute(device.renderContext);
    const fresh = await read(device, g2.getOutput("R.color")!);
    const g3 = rainGraph(device, scene, { ...kRain, time: kRain.time + 0.05 }, outs);
    g3.execute(device.renderContext);
    const later = await read(device, g3.getOutput("R.color")!);
    const same = bitwiseEqual(frames[0]!, frames[2]!) && bitwiseEqual(frames[0]!, fresh);
    const moved = !bitwiseEqual(frames[0]!, later);
    // identity through DepthComposite
    const id: boolean[] = [];
    for (const props of [{ ...kRain, opacity: 0 }, { ...kRain, count: 0 }]) {
        const gi = rainGraph(device, scene, props, ["D.dst", "L.color", "R.color"], true);
        gi.execute(device.renderContext);
        const d = await read(device, gi.getOutput("D.dst")!), l = await read(device, gi.getOutput("L.color")!), r = await read(device, gi.getOutput("R.color")!);
        id.push(bitwiseEqual(d, l) && r.every((v) => v === 0));
    }
    log(`RainLayer: same time on 3 frames + fresh graph bitwise ${same}; time + 0.05 s differs ${moved}; opacity 0 / count 0 -> empty layer, DepthComposite identity ${id.join(", ")}`);
    expectEq(same, true, "stateless: same time = same layer");
    expectEq(moved, true, "the rain moves with time");
    expectEq(id.every((x) => x), true, "empty layer = identity");
});

// ------------------------------------------------------------------------------------------------ rainy-night chain

function rainyNight(device: Device, scene: Scene, ssrTemporal: boolean, outputs: string[]): RenderGraph {
    const passes: PassDecl[] = [
        ["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: "None" }],
        ["RainWetness", "W", kWet],
        ["HiZBuild", "Z", {}],
        ["SSR", "S", { maxRoughness: 0.6, temporal: ssrTemporal }],
        ["RainLayer", "R", { ...kRain, count: 2000 }],
        ["DepthComposite", "D", { alphaMode: "premultiplied" }],
    ];
    const edges: [string, string][] = [
        ...kWetEdges, ["G.depth", "Z.depth"],
        ["W.dst", "S.color"], ["G.depth", "S.depth"], ["W.normWOut", "S.normW"], ["W.specRoughOut", "S.specRough"], ["Z.hiz", "S.hiz"], ["G.mvec", "S.mvec"],
        ["G.posW", "R.posW"], ["S.dst", "D.src"], ["G.posW", "D.posW"], ["R.color", "D.layer"], ["R.depth", "D.layerDepth"],
    ];
    return buildGraph(device, scene, passes, edges, outputs, kSize, kSize);
}

gpuTest("RainyNight.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device, kSky);
    const outs = ["D.dst", "S.reflection", "W.wetness"];
    const res = await coldVsConsecutive(device, scene, () => rainyNight(device, scene, true, outs), outs, ["camera-reset", "prime", "explicit", "none"]);
    for (const r of res) log(`RainyNight (wetness + SSR temporal + rain) cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
    for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `cold = consecutive after ${r.method}`);
    expectEq(res.find((r) => r.method === "none")!.differs.some((d) => d), true, "without a cut SSR's history changes X (control)");
    const off = await coldVsConsecutive(device, scene, () => rainyNight(device, scene, false, outs), outs, ["none"]);
    log(`RainyNight, SSR temporal off, no cut: X frames bitwise equal ${off[0]!.equalFrames}/${off[0]!.frames}`);
    expectEq(off[0]!.equalFrames, off[0]!.frames, "temporal off: the chain keeps no state");
});

// ---------------------------------------------------------------------------------------------- RainLayer sky occlusion

/** Up-ray (+y) from x hits the test scene: floor quad (y = 0, |x|,|z| <= 6) or one of the boxes (origin below its top). */
function roofed(x: number[]): boolean {
    const under = (lo: number[], hi: number[]) => x[0]! >= lo[0]! && x[0]! <= hi[0]! && x[2]! >= lo[2]! && x[2]! <= hi[2]! && x[1]! < hi[1]!;
    return under([-6, 0, -6], [6, 0, 6]) || under(kBoxA[0], kBoxA[1]) || under(kBoxB[0], kBoxB[1]);
}

/** Camera below the floor plane near its edge (z = 6), looking along +z: the floor is a roof over the near drops. */
const kPoseUnder = { pos: [1.0, -1.0, 3.0] as [number, number, number], target: [1.0, -0.6, 9.0] as [number, number, number] };
/** Larger box and more drops for the under-the-roof pose: roofed drops at z < 6, open-sky drops at z > 6. */
const kRainUnder: RainParams = { ...kRain, box: 10, count: 3000 };

function fnv(a: Float32Array): string {
    const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    let h = 0x811c9dc5;
    for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i]!, 16777619) >>> 0;
    return h.toString(16).padStart(8, "0");
}

gpuTest("RainLayer.skyOcclusion", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    const outs = ["R.color", "R.depth"];
    const render = async (props: Record<string, unknown>) => {
        const g = rainGraph(device, scene, props, outs);
        g.execute(device.renderContext);
        return { col: await read(device, g.getOutput("R.color")!), dep: await read(device, g.getOutput("R.depth")!), pos: await read(device, g.getOutput("G.posW")!) };
    };
    // (a) default hashes (compare with the pass before the option existed) and explicit "None" = default
    const hashes: string[] = [];
    let explicitNone = true;
    for (const [name, pose] of [["A", kPoseA], ["under", kPoseUnder]] as const) {
        setPose(scene, pose);
        const d = await render(kRain), n = await render({ ...kRain, skyOcclusion: "None" });
        hashes.push(`${name}: color=${fnv(d.col)} depth=${fnv(d.dep)}`);
        explicitNone &&= bitwiseEqual(d.col, n.col) && bitwiseEqual(d.dep, n.dep);
    }
    log(`defaultHash RainLayer ${hashes.join(" ")}`);
    // (b) both poses: RayTraced vs the CPU splat without roofed drops, and vs None
    let ok = explicitNone;
    const lines: string[] = [];
    for (const [name, pose, p] of [["above", kPoseA, kRain], ["under", kPoseUnder, kRainUnder]] as const) {
        setPose(scene, pose);
        const un = await render(p), ur = await render({ ...p, skyOcclusion: "RayTraced" });
        const all = rainCpu(scene, p, un.pos), open = rainCpu(scene, p, un.pos, roofed);
        let off = 0, lit = 0, depthOff = 0, maxErr = 0, roofPx = 0, roofLeft = 0, openPx = 0, openChanged = 0;
        for (let k = 0; k < kSize * kSize; k++) {
            let e = 0;
            for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(ur.col[k * 4 + c]! - open.rgba[k * 4 + c]!));
            if (ur.col[k * 4 + 3]! > 0 || open.rgba[k * 4 + 3]! > 0) lit++;
            if (e > 2e-3) off++;
            else maxErr = Math.max(maxErr, e);
            const gd = ur.dep[k * 4]!, rd = open.depth[k]!;
            if ((gd === kFar) !== (rd === kFar) || (gd !== kFar && Math.abs(gd - rd) > 1e-4 * Math.max(1, rd))) depthOff++;
            const allA = all.rgba[k * 4 + 3]!, openA = open.rgba[k * 4 + 3]!;
            if (allA > 0 && openA === 0) { roofPx++; if (ur.col[k * 4 + 3]! !== 0) roofLeft++; }
            let sameRef = all.depth[k] === open.depth[k];
            for (let c = 0; c < 4; c++) sameRef &&= all.rgba[k * 4 + c] === open.rgba[k * 4 + c];
            if (openA > 0 && sameRef) {
                openPx++;
                let same = ur.dep[k * 4] === un.dep[k * 4];
                for (let c = 0; c < 4; c++) same &&= ur.col[k * 4 + c] === un.col[k * 4 + c];
                if (!same) openChanged++;
            }
        }
        lines.push(`${name}: drops drawn ${all.drawn} -> ${open.drawn} (CPU), layer px ${lit}, off-ref (> 2e-3) ${off}, max err ${maxErr.toExponential(2)}, depth mismatches ${depthOff}; roof-only px ${roofPx} (still lit ${roofLeft}); open-sky px ${openPx} (changed vs None ${openChanged})`);
        ok &&= off <= Math.ceil(0.005 * lit) && depthOff <= Math.ceil(0.005 * lit) && openPx > 200 && openChanged === 0 && roofLeft <= Math.ceil(0.005 * roofPx);
        if (name === "under") ok &&= all.drawn - open.drawn > 50 && open.drawn > 50 && roofPx > 200;
    }
    log(`RainLayer sky occlusion: explicit None = default ${explicitNone}; ${lines.join("; ")}`);
    expectEq(ok, true, "roofed drops vanish (CPU parity), open-sky streaks unchanged bit for bit, explicit None = default");
});
