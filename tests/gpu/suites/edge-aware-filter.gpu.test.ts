/**
 * EdgeAwareFilter (packages/render-passes/src/EdgeAwareFilter): float64 CPU references of the three methods on the same
 * readbacks, bit-exact identity parameters, repeatability, parameter validation.
 */

import { createPass } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kSize, log, makeScene, readAll, runGraph, type PassDecl, type V3 } from "./stylize-helpers.js";

const kLit: PassDecl[] = [["DeferredLighting", "L", { shadowMode: "RayTraced" }], ["ToneMapper", "TM", { operator: "Aces", autoExposure: false, exposureCompensation: 0, clamp: true, outputFormat: "RGBA32Float" }]];
const kLitEdges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["L.color", "TM.src"]];
const SKY: V3 = [0.3, 0.35, 0.45];
const W = kSize, H = kSize;
const kLumaW: V3 = [0.2126, 0.7152, 0.0722];

type Params = Record<string, number | string>;

async function filterRun(device: Parameters<typeof makeScene>[0], params: Params, withGuide = false) {
    const scene = makeScene(device, SKY);
    const edges: [string, string][] = [...kLitEdges, ["TM.dst", "F.src"]];
    if (withGuide) edges.push(["G.linearZ", "F.guide"]);
    const outs = ["F.dst", "TM.dst", ...(withGuide ? ["G.linearZ"] : [])];
    const g = await runGraph(device, scene, [...kLit, ["EdgeAwareFilter", "F", params]], edges, outs);
    const [out, src, guide] = await readAll(device, g, outs);
    return { out: out!, src: src!, guide: guide ?? null, g };
}

const lumaEnc = (s: Float32Array, i: number) => {
    const l = kLumaW[0] * s[i * 4]! + kLumaW[1] * s[i * 4 + 1]! + kLumaW[2] * s[i * 4 + 2]!;
    return l > 0 ? Math.pow(l, 1 / 2.2) : 0;
};

/** Guide values as the pass computes them. */
function guideOf(src: Float32Array, guide: Float32Array | null, scale: number): Float64Array {
    const g = new Float64Array(W * H);
    const stride = guide ? guide.length / (W * H) : 0;
    for (let i = 0; i < W * H; i++) g[i] = guide ? scale * guide[i * stride]! : lumaEnc(src, i);
    return g;
}

/** out = max(src + strength * (result - src), 0); alpha from src. */
function compose(src: Float32Array, result: (i: number, c: number) => number, strength: number): Float64Array {
    const o = new Float64Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
        for (let c = 0; c < 3; c++) o[i * 4 + c] = Math.max(src[i * 4 + c]! + strength * (result(i, c) - src[i * 4 + c]!), 0);
        o[i * 4 + 3] = src[i * 4 + 3]!;
    }
    return o;
}

/** Mean over the (2r+1)^2 window truncated at the border, per channel of a planar [ch][W*H] array. */
function boxMean(planes: Float64Array[], r: number): Float64Array[] {
    return planes.map((pl) => {
        const h = new Float64Array(W * H), o = new Float64Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            let s = 0;
            for (let t = Math.max(x - r, 0); t <= Math.min(x + r, W - 1); t++) s += pl[y * W + t]!;
            h[y * W + x] = s;
        }
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            let s = 0;
            for (let t = Math.max(y - r, 0); t <= Math.min(y + r, H - 1); t++) s += h[t * W + x]!;
            const n = (Math.min(x + r, W - 1) - Math.max(x - r, 0) + 1) * (Math.min(y + r, H - 1) - Math.max(y - r, 0) + 1);
            o[y * W + x] = s / n;
        }
        return o;
    });
}

function guidedRef(src: Float32Array, g: Float64Array, p: { radius: number; eps: number; detail: number; strength: number }): Float64Array {
    const ch = (c: number) => Float64Array.from({ length: W * H }, (_, i) => src[i * 4 + c]!);
    const P = [ch(0), ch(1), ch(2)];
    const [mI, mII, ...rest] = boxMean([g, g.map((v) => v * v), ...P, ...P.map((pl) => pl.map((v, i) => v * g[i]!))], p.radius);
    const mP = rest.slice(0, 3), mIP = rest.slice(3, 6);
    const A = [0, 1, 2].map(() => new Float64Array(W * H)), B = [0, 1, 2].map(() => new Float64Array(W * H));
    for (let i = 0; i < W * H; i++) {
        const v = Math.max(mII![i]! - mI![i]! * mI![i]!, 0);
        for (let c = 0; c < 3; c++) {
            A[c]![i] = (mIP[c]![i]! - mI![i]! * mP[c]![i]!) / (v + p.eps);
            B[c]![i] = mP[c]![i]! - A[c]![i]! * mI![i]!;
        }
    }
    const mA = boxMean(A, p.radius), mB = boxMean(B, p.radius);
    return compose(src, (i, c) => { const s = src[i * 4 + c]!; return s + p.detail * (s - (mA[c]![i]! * g[i]! + mB[c]![i]!)); }, p.strength);
}

function bilateralGridRef(src: Float32Array, g: Float64Array, p: { cellSize: number; sigmaR: number; detail: number; strength: number }): Float64Array {
    const gw = Math.ceil(W / p.cellSize), gh = Math.ceil(H / p.cellSize), gd = Math.round(1 / p.sigmaR) + 1;
    const inv = Math.fround(1 / p.sigmaR);
    const n = gw * gh * gd, idx = (x: number, y: number, z: number) => (z * gh + y) * gw + x;
    let grid = new Float64Array(4 * n);
    const gc = (i: number) => Math.min(Math.max(g[i]!, 0), 1);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const z = Math.min(Math.floor(Math.fround(Math.fround(gc(i)) * inv) + 0.5), gd - 1);
        const k = idx(Math.floor(x / p.cellSize), Math.floor(y / p.cellSize), z) * 4;
        for (let c = 0; c < 3; c++) grid[k + c] += Math.floor(Math.fround(Math.min(Math.max(src[i * 4 + c]!, 0), 64) * 1024 + 0.5));
        grid[k + 3] += 1024;
    }
    grid = grid.map((v) => v / 1024);
    const kB = [1, 4, 6, 4, 1].map((v) => v / 16);
    for (const axis of [0, 1, 2]) {
        const o = new Float64Array(4 * n), d = [axis === 0 ? 1 : 0, axis === 1 ? 1 : 0, axis === 2 ? 1 : 0];
        for (let z = 0; z < gd; z++) for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
            for (let t = -2; t <= 2; t++) {
                const qx = x + t * d[0]!, qy = y + t * d[1]!, qz = z + t * d[2]!;
                if (qx < 0 || qy < 0 || qz < 0 || qx >= gw || qy >= gh || qz >= gd) continue;
                for (let c = 0; c < 4; c++) o[idx(x, y, z) * 4 + c] += kB[t + 2]! * grid[idx(qx, qy, qz) * 4 + c]!;
            }
        }
        grid = o;
    }
    const base = new Float64Array(W * H * 3);
    const cl = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const u = [cl((x + 0.5) / p.cellSize - 0.5, gw - 1), cl((y + 0.5) / p.cellSize - 0.5, gh - 1), cl(gc(i) * inv, gd - 1)];
        const c0 = u.map(Math.floor), c1 = c0.map((v, a) => Math.min(v + 1, [gw, gh, gd][a]! - 1)), f = u.map((v, a) => v - c0[a]!);
        const acc = [0, 0, 0, 0];
        for (let k = 0; k < 8; k++) {
            const b = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
            const w = b.reduce((m, bb, a) => m * (bb ? f[a]! : 1 - f[a]!), 1);
            const j = idx(b[0] ? c1[0]! : c0[0]!, b[1] ? c1[1]! : c0[1]!, b[2] ? c1[2]! : c0[2]!) * 4;
            for (let c = 0; c < 4; c++) acc[c] += w * grid[j + c]!;
        }
        for (let c = 0; c < 3; c++) base[i * 3 + c] = acc[3]! > 1e-6 ? acc[c]! / acc[3]! : src[i * 4 + c]!;
    }
    return compose(src, (i, c) => { const s = src[i * 4 + c]!; return s + p.detail * (s - base[i * 3 + c]!); }, p.strength);
}

type Level = { w: number; h: number; d: Float64Array };

function llRef(src: Float32Array, p: { pyramidLevels: number; intensitySamples: number; sigmaR: number; detail: number; strength: number }): Float64Array {
    const dims: [number, number][] = [[W, H]];
    while (dims.length <= p.pyramidLevels) {
        const [w, h] = dims[dims.length - 1]!;
        if (w < 4 || h < 4) break;
        dims.push([Math.ceil(w / 2), Math.ceil(h / 2)]);
    }
    const n = dims.length - 1, K = p.intensitySamples;
    const kB = [1, 4, 6, 4, 1].map((v) => v / 16);
    const down = (a: Level): Level => {
        const w = Math.ceil(a.w / 2), h = Math.ceil(a.h / 2), d = new Float64Array(w * h);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            let s = 0;
            for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) {
                const qx = Math.min(Math.max(2 * x + i, 0), a.w - 1), qy = Math.min(Math.max(2 * y + j, 0), a.h - 1);
                s += kB[i + 2]! * kB[j + 2]! * a.d[qy * a.w + qx]!;
            }
            d[y * w + x] = s;
        }
        return { w, h, d };
    };
    const up = (c: Level, x: number, y: number) => {
        const cx = Math.min(Math.max(x * 0.5 - 0.25, 0), c.w - 1), cy = Math.min(Math.max(y * 0.5 - 0.25, 0), c.h - 1);
        const x0 = Math.floor(cx), y0 = Math.floor(cy), x1 = Math.min(x0 + 1, c.w - 1), y1 = Math.min(y0 + 1, c.h - 1), fx = cx - x0, fy = cy - y0;
        const v = (xx: number, yy: number) => c.d[yy * c.w + xx]!;
        return (v(x0, y0) * (1 - fx) + v(x1, y0) * fx) * (1 - fy) + (v(x0, y1) * (1 - fx) + v(x1, y1) * fx) * fy;
    };
    const pyr = (l0: Level) => { const ps = [l0]; for (let l = 0; l < n; l++) ps.push(down(ps[l]!)); return ps; };
    const gin = pyr({ w: W, h: H, d: Float64Array.from({ length: W * H }, (_, i) => lumaEnc(src, i)) });
    const acc = dims.slice(0, n).map(([w, h]) => new Float64Array(w * h));
    for (let k = 0; k < K; k++) {
        const ref = k / (K - 1);
        const rk = pyr({ w: W, h: H, d: gin[0]!.d.map((g) => { const d = g - ref, t = d / p.sigmaR; return g + p.detail * d * Math.exp(-0.5 * t * t); }) });
        for (let l = 0; l < n; l++) {
            const L = rk[l]!;
            for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
                const i = y * L.w + x;
                const w = Math.max(1 - Math.abs(Math.min(Math.max(gin[l]!.d[i]!, 0), 1) - ref) * (K - 1), 0);
                acc[l]![i] += w * (L.d[i]! - up(rk[l + 1]!, x, y));
            }
        }
    }
    let r: Level = gin[n]!;
    for (let l = n - 1; l >= 0; l--) {
        const [w, h] = dims[l]!, d = new Float64Array(w * h);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = acc[l]![y * w + x]! + up(r, x, y);
        r = { w, h, d };
    }
    return compose(src, (i, c) => {
        const s = src[i * 4 + c]!, l = kLumaW[0] * src[i * 4]! + kLumaW[1] * src[i * 4 + 1]! + kLumaW[2] * src[i * 4 + 2]!;
        const u = Math.max(r.d[i]!, 0), l2 = u > 0 ? Math.pow(u, 2.2) : 0;
        return l > 1e-5 ? s * (l2 / l) : s + (l2 - l);
    }, p.strength);
}

function compare(gpu: Float32Array, ref: Float64Array, tol: number) {
    let maxErr = 0, bad = 0;
    for (let i = 0; i < W * H; i++) {
        let e = 0;
        for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(gpu[i * 4 + c]! - ref[i * 4 + c]!));
        if (!(e <= tol)) bad++;
        if (!(e <= maxErr)) maxErr = e;
    }
    return { maxErr, badShare: bad / (W * H) };
}

/** Mean absolute change of the output against the input (sanity: the filter does something). */
function meanChange(a: Float32Array, b: Float32Array): number {
    let s = 0;
    for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) s += Math.abs(a[i * 4 + c]! - b[i * 4 + c]!);
    return s / (3 * W * H);
}

gpuTest("EdgeAwareFilter.guidedMatchesCpuReference", async ({ device }) => {
    const cases: [Params, boolean][] = [
        [{ method: "guided", radius: 4, eps: 0.01, detail: -1, strength: 1 }, false],
        [{ method: "guided", radius: 6, eps: 0.002, detail: 1.5, strength: 0.7 }, false],
        [{ method: "guided", radius: 3, eps: 0.05, detail: -0.6, strength: 1, guideScale: 0.2 }, true],
    ];
    for (const [p, withGuide] of cases) {
        const { out, src, guide } = await filterRun(device, p, withGuide);
        const ref = guidedRef(src, guideOf(src, guide, Number(p.guideScale ?? 1)), p as never);
        const r = compare(out, ref, 1e-3);
        log(`guided ${JSON.stringify(p)} guide=${withGuide}: max err ${r.maxErr.toExponential(2)}, mean change ${meanChange(out, src).toExponential(2)}`);
        expectEq(r.maxErr < 1e-3, true, "guided filter matches the CPU reference (max abs err < 1e-3)");
        expectEq(meanChange(out, src) > 1e-3, true, "the filter changes the image");
    }
});

gpuTest("EdgeAwareFilter.bilateralGridMatchesCpuReference", async ({ device }) => {
    // Pixels whose range coordinate g / sigmaR lies within float rounding of a cell boundary may splat into the
    // neighbouring cell on the GPU; such a pixel moves its cell's average by up to ~1 / cellSize^2. So: almost all
    // pixels within 1e-3, and a bound on the worst pixel.
    for (const p of [{ method: "bilateralGrid", cellSize: 8, sigmaR: 0.1, detail: -1, strength: 1 }, { method: "bilateralGrid", cellSize: 6, sigmaR: 0.05, detail: 2, strength: 1 }]) {
        const { out, src } = await filterRun(device, p);
        const ref = bilateralGridRef(src, guideOf(src, null, 1), p);
        const r = compare(out, ref, 1e-3);
        log(`bilateralGrid ${JSON.stringify(p)}: max err ${r.maxErr.toExponential(2)}, share > 1e-3 ${(100 * r.badShare).toFixed(3)} %, mean change ${meanChange(out, src).toExponential(2)}`);
        expectEq(r.badShare < 5e-3 && r.maxErr < 0.05, true, "bilateral grid matches the CPU reference (99.5 % of pixels < 1e-3, all < 0.05)");
        expectEq(meanChange(out, src) > 1e-3, true, "the filter changes the image");
    }
});

gpuTest("EdgeAwareFilter.localLaplacianMatchesCpuReference", async ({ device }) => {
    for (const p of [{ method: "localLaplacian", pyramidLevels: 4, intensitySamples: 6, sigmaR: 0.2, detail: 1, strength: 1 },
        { method: "localLaplacian", pyramidLevels: 5, intensitySamples: 8, sigmaR: 0.1, detail: -1, strength: 0.8 }]) {
        const { out, src } = await filterRun(device, p);
        const ref = llRef(src, p);
        const r = compare(out, ref, 1e-3);
        log(`localLaplacian ${JSON.stringify(p)}: max err ${r.maxErr.toExponential(2)}, share > 1e-3 ${(100 * r.badShare).toFixed(3)} %, mean change ${meanChange(out, src).toExponential(2)}`);
        expectEq(r.maxErr < 1e-3, true, "local Laplacian matches the CPU reference (max abs err < 1e-3)");
        expectEq(meanChange(out, src) > 1e-3, true, "the filter changes the image");
    }
});

gpuTest("EdgeAwareFilter.identityIsBitExact", async ({ device }) => {
    for (const p of [{ method: "guided", detail: 0 }, { method: "guided", strength: 0 }, { method: "bilateralGrid", detail: 0 }, { method: "localLaplacian", strength: 0 }]) {
        const { out, src } = await filterRun(device, p);
        expectEq(bitwiseEqual(out, src), true, `identity ${JSON.stringify(p)}`);
    }
});

gpuTest("EdgeAwareFilter.repeatableAndStateless", async ({ device }) => {
    for (const method of ["guided", "bilateralGrid", "localLaplacian"]) {
        const { out, g, src } = await filterRun(device, { method, detail: 0.8 });
        g.execute(device.renderContext);
        g.execute(device.renderContext);
        const [again, src2] = await readAll(device, g, ["F.dst", "TM.dst"]);
        expectEq(bitwiseEqual(src, src2!), true, `${method}: input unchanged`);
        expectEq(bitwiseEqual(out, again!), true, `${method}: repeated execution is bitwise equal`);
        const fresh = await filterRun(device, { method, detail: 0.8 });
        expectEq(bitwiseEqual(out, fresh.out), true, `${method}: a fresh pass instance gives the same image`);
    }
});

gpuTest("EdgeAwareFilter.rejectsBadParameters", async ({ device }) => {
    const bad: Params[] = [{ method: "median" }, { radius: 0 }, { cellSize: 3.5 }, { sigmaR: 0 }, { detail: -2 }, { levels: 4 }];
    for (const p of bad) {
        let threw = false;
        try { createPass(device, "EdgeAwareFilter", p); } catch { threw = true; }
        expectEq(threw, true, `rejects ${JSON.stringify(p)}`);
    }
});
