/**
 * SelectiveBloom and StylizedFocus:
 *  - CPU parity: both passes on synthetic frames (HDR colour, raw-id texture, mask, world positions) against a CPU
 *    evaluation of the same formulas: id lists, invert, mask, threshold/knee, odd frame sizes; focus modes depth, point,
 *    screen, ids and mask with blur and desaturation;
 *  - selection: an unselected bright object adds no glow; glow has finite support around the selected object;
 *  - scene: StyleAovs.id -> SelectiveBloom.id on the test scene, glow only from the selected instance;
 *  - identity: intensity 0 / empty selection (bloom), strength 0 / everything in focus (focus) reproduce src bit for bit;
 *    in-focus pixels are bit-identical to src;
 *  - stateless: a frame after frames of other inputs on one graph equals a fresh graph bit for bit.
 */

import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, log, makeScene, read, type V3 } from "./stylize-helpers.js";
import { buildGraph, rgbaTexture } from "./fx-helpers.js";

type Dev = Parameters<typeof buildGraph>[0];
type Img = Float64Array | Float32Array;

const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const smoothstep = (a: number, b: number, x: number) => { const t = sat((x - a) / (b - a)); return t * t * (3 - 2 * t); };

/** Separable Gaussian of the shader (65 taps, stride max(3 sigma / 32, 1), linear lookups, clamp to edge); rgb only. */
function blurRef(img: Img, W: number, H: number, sigma: number): Float64Array {
    const T = 32, stride = Math.max((3 * sigma) / T, 1);
    const pass = (inp: Img, horiz: boolean) => {
        const out = new Float64Array(W * H * 4);
        const n = horiz ? W : H;
        const ws: number[] = [];
        let wsum = 0;
        for (let j = -T; j <= T; j++) { const d = j * stride; const w = Math.exp((-0.5 * d * d) / (sigma * sigma)); ws.push(w); wsum += w; }
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const acc = [0, 0, 0];
            const x0 = (horiz ? x : y) + 0.5;
            for (let j = -T; j <= T; j++) {
                const xc = x0 + j * stride - 0.5, f0 = Math.floor(xc), f = xc - f0;
                const i0 = Math.min(Math.max(f0, 0), n - 1), i1 = Math.min(Math.max(f0 + 1, 0), n - 1);
                const a = horiz ? (y * W + i0) * 4 : (i0 * W + x) * 4, b = horiz ? (y * W + i1) * 4 : (i1 * W + x) * 4;
                for (let k = 0; k < 3; k++) acc[k] += ws[j + T]! * (inp[a + k]! + (inp[b + k]! - inp[a + k]!) * f);
            }
            const o = (y * W + x) * 4;
            for (let k = 0; k < 3; k++) out[o + k] = acc[k]! / wsum;
            out[o + 3] = 1;
        }
        return out;
    };
    return pass(pass(img, true), false);
}

function compare(gpu: Float32Array, ref: Img, tol = 1e-4) {
    let bad = 0, maxErr = 0;
    const n = gpu.length / 4;
    for (let k = 0; k < n; k++) {
        let e = 0;
        for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(gpu[k * 4 + c]! - ref[k * 4 + c]!) / Math.max(1, Math.abs(ref[k * 4 + c]!)));
        if (e > tol) bad++;
        else maxErr = Math.max(maxErr, e);
    }
    return { bad, maxErr, n };
}

// ---------------------------------------------------------------------------------------------------------------------
// Synthetic frames: a dim gradient with discs; each disc has its own id (background id texture alpha 0).

type Disc = [cx: number, cy: number, r: number, c: V3, id: number];
function frames(W: number, H: number, discs: Disc[]) {
    const src = new Float32Array(W * H * 4), id = new Float32Array(W * H * 4), mask = new Float32Array(W * H * 4), pos = new Float32Array(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const g = 0.15 + 0.35 * (x / W) * (1 - y / H);
        src.set([g, 0.9 * g, 0.7 * g + 0.05 * Math.sin(x * 0.7) * Math.cos(y * 0.5) + 0.05, 0.6], i);
        id.set([0, 0, 0, 0], i);
        mask.set([sat(x / W + 0.2 * Math.sin(y * 0.1)), 0, 0, 1], i);
        // world positions: a sloped plane in front of the test camera; a background strip on the right (alpha 0)
        if (x < W - 8) pos.set([-3 + (6 * x) / W, 0.2 * Math.sin(y * 0.2), 2 - (8 * y) / H, 1], i);
        for (const [cx, cy, r, c, d] of discs) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) { src.set([c[0], c[1], c[2], 0.6], i); id.set([0.5, d, 0, 1], i); }
    }
    return { src, id, mask, pos };
}
const kDiscs: Disc[] = [[30, 40, 6, [6, 4, 2], 3], [90, 30, 8, [3, 3, 6], 7], [64, 92, 5, [9, 9, 9], 12], [20, 100, 4, [0.4, 0.3, 0.2], 7]];

// ---------------------------------------------------------------------------------------------------------------------
// SelectiveBloom CPU mirror

interface SB { ids: number[]; invert: boolean; intensity: number; threshold: number; knee: number; radius: number; tint: V3 }
const kSB: SB = { ids: [], invert: false, intensity: 1, threshold: 0, knee: 0, radius: 0.02, tint: [1, 1, 1] };

function bloomRef(p: SB, src: Float32Array, id: Float32Array | null, mask: Float32Array | null, W: number, H: number) {
    const W2 = Math.max(1, (W + 1) >> 1), H2 = Math.max(1, (H + 1) >> 1);
    const sel = (i: number) => {
        if (!id && !mask) return 0;
        let s = 1;
        if (id) s = id[i * 4 + 3] !== 0 && p.ids.includes(Math.floor(id[i * 4 + 1]! + 0.5)) ? 1 : 0;
        if (mask) s *= sat(mask[i * 4]!);
        return p.invert ? 1 - s : s;
    };
    const any = p.invert ? !!id || !!mask : (!!id && p.ids.length > 0) || (!!mask && !id);
    const intensity = any ? p.intensity : 0;
    const dst = new Float64Array(W * H * 4), gl = new Float64Array(W * H * 4);
    let G: Float64Array | null = null;
    if (intensity !== 0) {
        const B = new Float64Array(W2 * H2 * 4);
        for (let qy = 0; qy < H2; qy++) for (let qx = 0; qx < W2; qx++) {
            const c = [0, 0, 0];
            for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
                const px = Math.min(2 * qx + i, W - 1), py = Math.min(2 * qy + j, H - 1), k0 = py * W + px, s = sel(k0);
                for (let k = 0; k < 3; k++) c[k] += src[k0 * 4 + k]! * s;
            }
            for (let k = 0; k < 3; k++) c[k] = Math.max(c[k]! * 0.25, 0);
            const br = Math.max(c[0]!, c[1]!, c[2]!);
            let soft = Math.min(Math.max(br - p.threshold + p.knee, 0), 2 * p.knee);
            soft = (soft * soft) / (4 * p.knee + 1e-5);
            const w = Math.max(soft, br - p.threshold) / Math.max(br, 1e-5);
            for (let k = 0; k < 3; k++) B[(qy * W2 + qx) * 4 + k] = c[k]! * w;
            B[(qy * W2 + qx) * 4 + 3] = 1;
        }
        G = blurRef(B, W2, H2, Math.max(p.radius * H2, 0.5));
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        let g = [0, 0, 0];
        if (G) {
            const u = (x + 0.5) / W, v = (y + 0.5) / H;
            const fx0 = u * W2 - 0.5, fy0 = v * H2 - 0.5, ix = Math.floor(fx0), iy = Math.floor(fy0), fx = fx0 - ix, fy = fy0 - iy;
            const cx = (t: number) => Math.min(Math.max(t, 0), W2 - 1), cy = (t: number) => Math.min(Math.max(t, 0), H2 - 1);
            const t = (xx: number, yy: number, k: number) => G![(cy(yy) * W2 + cx(xx)) * 4 + k]!;
            g = [0, 1, 2].map((k) => {
                const a = t(ix, iy, k) + (t(ix + 1, iy, k) - t(ix, iy, k)) * fx, b = t(ix, iy + 1, k) + (t(ix + 1, iy + 1, k) - t(ix, iy + 1, k)) * fx;
                return intensity * p.tint[k]! * (a + (b - a) * fy);
            });
        }
        for (let k = 0; k < 3; k++) { dst[i + k] = src[i + k]! + g[k]!; gl[i + k] = g[k]!; }
        dst[i + 3] = src[i + 3]!;
        gl[i + 3] = 1;
    }
    return { dst, gl };
}

function sbGraph(device: Dev, W: number, H: number, props: Partial<SB>, f: { src: Float32Array; id?: Float32Array | null; mask?: Float32Array | null }) {
    const scene = makeScene(device);
    const inputs: Record<string, ReturnType<typeof rgbaTexture>> = { "B.src": rgbaTexture(device, W, H, f.src) };
    if (f.id) inputs["B.id"] = rgbaTexture(device, W, H, f.id);
    if (f.mask) inputs["B.mask"] = rgbaTexture(device, W, H, f.mask);
    return buildGraph(device, scene, [["SelectiveBloom", "B", props]], [], ["B.dst", "B.glow"], W, H, inputs);
}

gpuTest("SelectiveBloom.matchesCpuReference", async ({ device }) => {
    const cases: { name: string; W: number; H: number; props: Partial<SB>; id: boolean; mask: boolean }[] = [
        { name: "ids [3, 12]", W: 128, H: 128, props: { ids: [3, 12] }, id: true, mask: false },
        { name: "id 7, soft knee, tint, odd size", W: 127, H: 115, props: { ids: [7], threshold: 1, knee: 0.5, radius: 0.05, intensity: 0.7, tint: [1, 0.6, 0.3] }, id: true, mask: false },
        { name: "invert ids [3], wide radius", W: 128, H: 112, props: { ids: [3], invert: true, threshold: 2, radius: 0.15 }, id: true, mask: false },
        { name: "mask only", W: 120, H: 128, props: { radius: 0.03 }, id: false, mask: true },
        { name: "ids x mask", W: 128, H: 128, props: { ids: [3, 7, 12], radius: 0.01 }, id: true, mask: true },
    ];
    for (const c of cases) {
        const f = frames(c.W, c.H, kDiscs);
        const g = sbGraph(device, c.W, c.H, c.props, { src: f.src, id: c.id ? f.id : null, mask: c.mask ? f.mask : null });
        g.execute(device.renderContext);
        const dst = await read(device, g.getOutput("B.dst")!), gl = await read(device, g.getOutput("B.glow")!);
        const ref = bloomRef({ ...kSB, ...c.props } as SB, f.src, c.id ? f.id : null, c.mask ? f.mask : null, c.W, c.H);
        const a = compare(dst, ref.dst), b = compare(gl, ref.gl);
        let lit = 0;
        for (let k = 0; k < a.n; k++) if (ref.gl[k * 4]! + ref.gl[k * 4 + 1]! + ref.gl[k * 4 + 2]! > 1e-3) lit++;
        log(`SelectiveBloom parity [${c.name}] ${c.W}x${c.H}: glow px ${lit}, dst off-reference ${a.bad} (max rel err ${a.maxErr.toExponential(2)}), glow off-reference ${b.bad} (max ${b.maxErr.toExponential(2)})`);
        expectEq(lit > 200, true, `${c.name}: glow visible`);
        expectEq(a.bad <= 0.001 * a.n && b.bad <= 0.001 * b.n, true, `${c.name}: GPU = CPU reference`);
    }
});

/** Full-resolution reach of the glow: 32 taps x stride half-res pixels, + 2 half-res pixels for the box and bilinear steps. */
const supportPx = (radius: number, H: number) => { const H2 = (H + 1) >> 1, s = Math.max(radius * H2, 0.5); return 2 * (32 * Math.max((3 * s) / 32, 1) + 2); };

gpuTest("SelectiveBloom.onlySelectedObjectsGlow", async ({ device }) => {
    // Very bright unselected disc (id 12) far from a dim selected disc (id 3): the glow comes from id 3 alone. The blur
    // has finite support (32 taps x stride half-res pixels, + bilinear), so the unselected disc, farther away than that,
    // gets exactly zero glow.
    const W = 128, H = 128;
    const f = frames(W, H, [[30, 40, 6, [2, 2, 2], 3], [96, 96, 6, [500, 500, 500], 12]]);
    const radius = 0.03;
    const g = sbGraph(device, W, H, { ids: [3], radius }, { src: f.src, id: f.id });
    g.execute(device.renderContext);
    const gl = await read(device, g.getOutput("B.glow")!);
    const reach = supportPx(radius, H);
    let near = 0, far = 0, farMax = 0, other = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const v = gl[(y * W + x) * 4]!;
        if (Math.hypot(x - 30, y - 40) <= 6) near = Math.max(near, v);
        if (Math.hypot(x - 96, y - 96) <= 6) other = Math.max(other, v);
        if (Math.hypot(x - 30, y - 40) > 6 + reach) { if (v !== 0) far++; farMax = Math.max(farMax, v); }
    }
    log(`SelectiveBloom selection: glow on selected disc ${near.toFixed(3)}, on the unselected 500-disc ${other}; pixels beyond the ${reach.toFixed(1)} px support with glow ${far} (max ${farMax})`);
    expectEq(near > 0.3 && other === 0 && far === 0, true, "glow only from the selected id");
});

gpuTest("SelectiveBloom.sceneIds", async ({ device }) => {
    // StyleAovs (instance ids) -> SelectiveBloom.id on the test scene; src = constant 2. The selected instance is the
    // visible instance with the second-largest coverage (the largest is the ground): every pixel of it glows, and
    // pixels far from it do not.
    const W = 256, H = 256;
    const scene = makeScene(device);
    const src = new Float32Array(W * H * 4).fill(2);
    const mk = (ids: number[]) => buildGraph(device, scene, [["StyleAovs", "A", { idSource: "instance" }], ["SelectiveBloom", "B", { ids, radius: 0.01 }]],
        [["G.vbuffer", "A.vbuffer"], ["A.id", "B.id"]], ["B.glow", "A.id"], W, H, { "B.src": rgbaTexture(device, W, H, src) });
    const runUntilIds = async (g: ReturnType<typeof mk>) => {
        let ids = new Float32Array(0);
        for (let t = 0; t < 5; t++) { g.execute(device.renderContext); ids = await read(device, g.getOutput("A.id")!); if (ids.some((v, i) => i % 4 === 3 && v !== 0)) break; }
        return ids;
    };
    const ids0 = await runUntilIds(mk([]));
    const count = new Map<number, number>();
    for (let i = 0; i < W * H; i++) if (ids0[i * 4 + 3] !== 0) { const d = Math.round(ids0[i * 4 + 1]!); count.set(d, (count.get(d) ?? 0) + 1); }
    const ranked = [...count.entries()].sort((a, b) => b[1] - a[1]);
    const pick = ranked[1]![0];
    const g = mk([pick]);
    const ids = await runUntilIds(g);
    const gl = await read(device, g.getOutput("B.glow")!);
    const sel: number[] = [];
    for (let i = 0; i < W * H; i++) if (ids[i * 4 + 3] !== 0 && Math.round(ids[i * 4 + 1]!) === pick) sel.push(i);
    let minSel = Infinity, farLit = 0, far = 0;
    for (const i of sel) minSel = Math.min(minSel, gl[i * 4]!);
    const reach = supportPx(0.01, H) + 1;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let dmin = Infinity;
        for (const i of sel) { const d = Math.hypot((i % W) - x, Math.floor(i / W) - y); if (d < dmin) dmin = d; if (dmin <= reach) break; }
        if (dmin > reach) { far++; if (gl[(y * W + x) * 4] !== 0) farLit++; }
    }
    log(`SelectiveBloom scene: instances ${JSON.stringify(ranked.slice(0, 6))}; selected ${pick} covers ${sel.length} px, min glow there ${minSel.toFixed(3)}; ${farLit}/${far} far pixels glow`);
    expectEq(sel.length > 50 && minSel > 0.02 && farLit === 0 && far > 1000, true, "scene id selection");
});

gpuTest("SelectiveBloom.identity", async ({ device }) => {
    const W = 128, H = 128;
    const f = frames(W, H, kDiscs);
    const cases: [string, Partial<SB>, boolean, boolean][] = [
        ["intensity 0", { ids: [3, 7], intensity: 0 }, true, false],
        ["empty id list", { ids: [] }, true, false],
        ["no id, no mask", { ids: [3] }, false, false],
        ["empty id list x mask", { ids: [] }, true, true],
    ];
    for (const [name, props, id, mask] of cases) {
        const g = sbGraph(device, W, H, props, { src: f.src, id: id ? f.id : null, mask: mask ? f.mask : null });
        g.execute(device.renderContext);
        const same = bitwiseEqual(await read(device, g.getOutput("B.dst")!), f.src);
        log(`SelectiveBloom identity [${name}]: dst bitwise = src ${same}`);
        expectEq(same, true, `identity ${name}`);
    }
});

// ---------------------------------------------------------------------------------------------------------------------
// StylizedFocus CPU mirror

interface SF { mode: string; strength: number; focusDistance: number; focusPoint: V3; screenCenter: [number, number]; focusRange: number; falloff: number; backgroundFocus: number; ids: number[]; blurRadius: number; desaturate: number }
const kSF: SF = { mode: "depth", strength: 1, focusDistance: 10, focusPoint: [0, 0, 0], screenCenter: [0.5, 0.5], focusRange: 1, falloff: 5, backgroundFocus: 0, ids: [], blurRadius: 0.01, desaturate: 0.8 };

function focusRef(p: SF, f: ReturnType<typeof frames>, W: number, H: number, cam: V3) {
    const falloff = Math.max(p.falloff, 1e-6), range = Math.max(p.focusRange, 0);
    const fw = (x: number, y: number) => {
        const i = (y * W + x) * 4;
        if (p.mode === "mask") return sat(f.mask[i]!);
        if (p.mode === "ids") return f.id[i + 3] !== 0 && p.ids.includes(Math.round(f.id[i + 1]!)) ? 1 : 0;
        if (p.mode === "screen") {
            const dx = ((x + 0.5) / W - p.screenCenter[0]) * (W / H), dy = (y + 0.5) / H - p.screenCenter[1];
            return 1 - smoothstep(range, range + falloff, Math.hypot(dx, dy));
        }
        if (f.pos[i + 3] === 0) return p.backgroundFocus;
        const px = f.pos[i]!, py = f.pos[i + 1]!, pz = f.pos[i + 2]!;
        const d = p.mode === "depth" ? Math.abs(Math.hypot(px - cam[0], py - cam[1], pz - cam[2]) - p.focusDistance) : Math.hypot(px - p.focusPoint[0], py - p.focusPoint[1], pz - p.focusPoint[2]);
        return 1 - smoothstep(range, range + falloff, d);
    };
    const blurOn = p.strength > 0 && p.blurRadius > 0;
    const s2 = p.blurRadius * H;
    const B1 = blurOn ? blurRef(f.src, W, H, Math.max(s2 / 3, 0.25)) : null, B2 = blurOn ? blurRef(f.src, W, H, Math.max(s2, 0.25)) : null;
    const dst = new Float64Array(W * H * 4), fo = new Float64Array(W * H * 4), inFocus = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4, wgt = fw(x, y), a = p.strength * (1 - wgt);
        fo.set([wgt, wgt, wgt, 1], i);
        if (!(a > 0)) { for (let k = 0; k < 4; k++) dst[i + k] = f.src[i + k]!; inFocus[y * W + x] = 1; continue; }
        let c = [0, 1, 2].map((k) => f.src[i + k]!);
        if (blurOn) {
            const t = sat(a);
            c = [0, 1, 2].map((k) => (t < 0.5 ? f.src[i + k]! + (B1![i + k]! - f.src[i + k]!) * 2 * t : B1![i + k]! + (B2![i + k]! - B1![i + k]!) * (2 * t - 1)));
        }
        const L = 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!, s = sat(a * p.desaturate);
        for (let k = 0; k < 3; k++) dst[i + k] = c[k]! + (L - c[k]!) * s;
        dst[i + 3] = f.src[i + 3]!;
    }
    return { dst, fo, inFocus };
}

function sfGraph(device: Dev, W: number, H: number, props: Partial<SF>, f: ReturnType<typeof frames>) {
    const scene = makeScene(device);
    const inputs = { "F.src": rgbaTexture(device, W, H, f.src), "F.posW": rgbaTexture(device, W, H, f.pos), "F.id": rgbaTexture(device, W, H, f.id), "F.mask": rgbaTexture(device, W, H, f.mask) };
    return { g: buildGraph(device, scene, [["StylizedFocus", "F", props]], [], ["F.dst", "F.focus"], W, H, inputs), cam: (() => { const c = scene.camera.getPosition(); return [c.x, c.y, c.z] as V3; })() };
}

gpuTest("StylizedFocus.matchesCpuReference", async ({ device }) => {
    const cases: { name: string; W: number; H: number; props: Partial<SF> }[] = [
        { name: "depth", W: 128, H: 128, props: { mode: "depth", focusDistance: 5, focusRange: 0.5, falloff: 2 } },
        { name: "point, background in focus, odd size", W: 127, H: 101, props: { mode: "point", focusPoint: [0, 0, -1], focusRange: 0.8, falloff: 1.5, backgroundFocus: 1, blurRadius: 0.03, desaturate: 1 } },
        { name: "screen, half strength", W: 128, H: 96, props: { mode: "screen", screenCenter: [0.35, 0.6], focusRange: 0.15, falloff: 0.2, strength: 0.6, blurRadius: 0.05 } },
        { name: "ids [3, 12], wide blur", W: 128, H: 128, props: { mode: "ids", ids: [3, 12], blurRadius: 0.12, desaturate: 0.5 } },
        { name: "mask, desaturate only", W: 120, H: 128, props: { mode: "mask", blurRadius: 0 } },
    ];
    for (const c of cases) {
        const f = frames(c.W, c.H, kDiscs);
        const { g, cam } = sfGraph(device, c.W, c.H, c.props, f);
        g.execute(device.renderContext);
        const dst = await read(device, g.getOutput("F.dst")!), fo = await read(device, g.getOutput("F.focus")!);
        const ref = focusRef({ ...kSF, ...c.props } as SF, f, c.W, c.H, cam);
        const a = compare(dst, ref.dst), b = compare(fo, ref.fo);
        let changed = 0, inF = 0, inFsame = 0;
        for (let k = 0; k < a.n; k++) {
            if (Math.abs(ref.dst[k * 4]! - f.src[k * 4]!) > 1e-3) changed++;
            if (ref.inFocus[k]) { inF++; if ([0, 1, 2, 3].every((q) => Object.is(dst[k * 4 + q], f.src[k * 4 + q]))) inFsame++; }
        }
        log(`StylizedFocus parity [${c.name}] ${c.W}x${c.H}: abstracted px ${changed}, dst off-reference ${a.bad} (max rel err ${a.maxErr.toExponential(2)}), focus off-reference ${b.bad}; in-focus px ${inF}, bitwise = src ${inFsame}`);
        expectEq(changed > 200 && (c.props.mode === "mask" || inF > 50), true, `${c.name}: both regions present`);
        expectEq(a.bad <= 0.001 * a.n && b.bad <= 0.001 * b.n, true, `${c.name}: GPU = CPU reference`);
        expectEq(inFsame, inF, `${c.name}: in-focus pixels unchanged`);
    }
});

gpuTest("StylizedFocus.identity", async ({ device }) => {
    const W = 128, H = 128;
    const f = frames(W, H, kDiscs);
    const cases: [string, Partial<SF>][] = [
        ["strength 0", { mode: "depth", strength: 0, blurRadius: 0.05 }],
        ["screen range covers the frame", { mode: "screen", focusRange: 2, falloff: 0.1 }],
        ["point, everything in range, background in focus", { mode: "point", focusRange: 100, backgroundFocus: 1 }],
    ];
    for (const [name, props] of cases) {
        const { g } = sfGraph(device, W, H, props, f);
        g.execute(device.renderContext);
        const same = bitwiseEqual(await read(device, g.getOutput("F.dst")!), f.src);
        log(`StylizedFocus identity [${name}]: dst bitwise = src ${same}`);
        expectEq(same, true, `identity ${name}`);
    }
});

gpuTest("FocusPasses.stateless", async ({ device }) => {
    const W = 128, H = 128;
    const a = frames(W, H, kDiscs), b = frames(W, H, [[64, 64, 20, [8, 1, 1], 3]]);
    const runs: [string, () => { g: ReturnType<typeof buildGraph>; out: string; inputs: (fr: ReturnType<typeof frames>) => [string, Float32Array][] }][] = [
        ["SelectiveBloom", () => ({ g: sbGraph(device, W, H, { ids: [3, 12], radius: 0.05 }, { src: b.src, id: b.id }), out: "B.dst", inputs: (fr) => [["B.src", fr.src], ["B.id", fr.id]] })],
        ["StylizedFocus", () => ({ g: sfGraph(device, W, H, { mode: "depth", focusDistance: 5, blurRadius: 0.04 }, b).g, out: "F.dst", inputs: (fr) => [["F.src", fr.src], ["F.posW", fr.pos], ["F.id", fr.id], ["F.mask", fr.mask]] })],
    ];
    for (const [name, mk] of runs) {
        const cold = mk();
        for (const [ref, data] of cold.inputs(a)) cold.g.setInput(ref, rgbaTexture(device, W, H, data));
        cold.g.execute(device.renderContext);
        const x0 = await read(device, cold.g.getOutput(cold.out)!);
        const warm = mk();
        warm.g.execute(device.renderContext);
        warm.g.execute(device.renderContext);
        for (const [ref, data] of warm.inputs(a)) warm.g.setInput(ref, rgbaTexture(device, W, H, data));
        warm.g.execute(device.renderContext);
        const same = bitwiseEqual(await read(device, warm.g.getOutput(warm.out)!), x0);
        log(`${name} stateless: frame A after frames of B on one graph bitwise = fresh graph ${same}`);
        expectEq(same, true, `${name} stateless`);
    }
});
