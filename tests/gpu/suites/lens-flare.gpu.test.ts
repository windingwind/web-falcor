/**
 * LensFlare:
 *  - CPU parity: bright pass, ghosts, halo, streak and composite on synthetic HDR frames (bright spots on a dim
 *    gradient), even and odd frame sizes, chromatic offset, external `bright` input; dst and the flare output match a
 *    CPU evaluation of the same formulas;
 *  - mirror: the first ghost of an off-centre spot lands at the point mirrored through the image centre;
 *  - identity: intensity 0, and a frame with nothing above the threshold (knee 0), reproduce src bit for bit;
 *  - stateless: re-running a frame after another input gives the same result bit for bit.
 */

import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, log, makeScene, read, type V3 } from "./stylize-helpers.js";
import { buildGraph, rgbaTexture } from "./fx-helpers.js";

interface P { intensity: number; threshold: number; knee: number; ghostCount: number; ghostSpacing: number; ghostIntensity: number; ghostTint: V3; haloRadius: number; haloFalloff: number; haloIntensity: number; haloTint: V3; chromatic: number; streakIntensity: number; streakLength: number; streakTint: V3 }
const kDef: P = { intensity: 1, threshold: 1, knee: 0.5, ghostCount: 4, ghostSpacing: 0.3, ghostIntensity: 1, ghostTint: [1, 0.9, 0.8], haloRadius: 0.45, haloFalloff: 5, haloIntensity: 0.5, haloTint: [0.8, 0.9, 1], chromatic: 0.006, streakIntensity: 0, streakLength: 0.4, streakTint: [0.55, 0.7, 1] };

/** Dim gradient (< 0.6) with bright spots; alpha 0.7. */
function frame(w: number, h: number, spots: [number, number, number, V3][], base = true): Float32Array {
    const a = new Float32Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const g = base ? 0.2 + 0.4 * (x / w) * (y / h) : 0;
        a.set([g, 0.8 * g, 0.6 * g, 0.7], i);
        for (const [cx, cy, r, c] of spots) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) a.set([c[0], c[1], c[2], 0.7], i);
    }
    return a;
}

const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const len2 = (x: number, y: number) => Math.hypot(x, y);

/** CPU mirror of LensFlare.cs.slang. */
function flareRef(p: P, src: Float32Array, bright: Float32Array | null, W: number, H: number) {
    const W2 = Math.max(1, (W + 1) >> 1), H2 = Math.max(1, (H + 1) >> 1);
    const B = new Float64Array(W2 * H2 * 3), F = new Float64Array(W2 * H2 * 3);
    const s = bright ?? src;
    for (let qy = 0; qy < H2; qy++) for (let qx = 0; qx < W2; qx++) {
        const c = [0, 0, 0];
        for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
            const px = Math.min(2 * qx + i, W - 1), py = Math.min(2 * qy + j, H - 1);
            for (let k = 0; k < 3; k++) c[k] += s[(py * W + px) * 4 + k]!;
        }
        for (let k = 0; k < 3; k++) c[k] = Math.max(c[k]! * 0.25, 0);
        const br = Math.max(c[0]!, c[1]!, c[2]!);
        let soft = Math.min(Math.max(br - p.threshold + p.knee, 0), 2 * p.knee);
        soft = (soft * soft) / (4 * p.knee + 1e-5);
        const w = Math.max(soft, br - p.threshold) / Math.max(br, 1e-5);
        for (let k = 0; k < 3; k++) B[(qy * W2 + qx) * 3 + k] = c[k]! * w;
    }
    const bil = (T: Float64Array, u: number, v: number, k: number) => {
        const x = u * W2 - 0.5, y = v * H2 - 0.5, ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
        const cx = (i: number) => Math.min(Math.max(i, 0), W2 - 1), cy = (i: number) => Math.min(Math.max(i, 0), H2 - 1);
        const t = (xx: number, yy: number) => T[(cy(yy) * W2 + cx(xx)) * 3 + k]!;
        const a = t(ix, iy) + (t(ix + 1, iy) - t(ix, iy)) * fx, b = t(ix, iy + 1) + (t(ix + 1, iy + 1) - t(ix, iy + 1)) * fx;
        return a + (b - a) * fy;
    };
    const ca = (u: number, v: number, ox: number, oy: number) => [bil(B, u + ox, v + oy, 0), bil(B, u, v, 1), bil(B, u - ox, v - oy, 2)];
    const row = (x: number, y: number, k: number) => {
        const xc = x - 0.5, i0 = Math.floor(xc), f = xc - i0;
        const at = (i: number) => (i >= 0 && i < W2 ? B[(y * W2 + i) * 3 + k]! : 0);
        return at(i0) + (at(i0 + 1) - at(i0)) * f;
    };
    const T = 24;
    for (let qy = 0; qy < H2; qy++) for (let qx = 0; qx < W2; qx++) {
        const fu = 1 - (qx + 0.5) / W2, fv = 1 - (qy + 0.5) / H2;
        const gx = (0.5 - fu) * p.ghostSpacing, gy = (0.5 - fv) * p.ghostSpacing, gl = len2(gx, gy);
        const dx = gl > 1e-6 ? gx / gl : 0, dy = gl > 1e-6 ? gy / gl : 0;
        const ox = dx * p.chromatic, oy = dy * p.chromatic;
        const gh = [0, 0, 0], ha = [0, 0, 0], st = [0, 0, 0];
        for (let i = 0; i < p.ghostCount; i++) {
            const su = fu + gx * i, sv = fv + gy * i;
            if (su < 0 || sv < 0 || su > 1 || sv > 1) continue;
            const w = sat(1 - len2(0.5 - su, 0.5 - sv) / 0.70710678);
            const c = ca(su, sv, ox, oy);
            for (let k = 0; k < 3; k++) gh[k] += c[k]! * w * w;
        }
        const hu = fu + dx * p.haloRadius, hv = fv + dy * p.haloRadius;
        if (gl > 1e-6 && hu >= 0 && hv >= 0 && hu <= 1 && hv <= 1) {
            const w = Math.pow(sat(1 - len2(0.5 - hu, 0.5 - hv) / 0.70710678), p.haloFalloff);
            const c = ca(hu, hv, ox, oy);
            for (let k = 0; k < 3; k++) ha[k] = c[k]! * w;
        }
        if (p.streakIntensity !== 0) {
            const stride = Math.max((p.streakLength * W2) / T, 1);
            let ws = 0;
            for (let j = -T; j <= T; j++) {
                const w = Math.exp((-4 * Math.abs(j)) / T);
                for (let k = 0; k < 3; k++) st[k] += w * row(qx + 0.5 + j * stride, qy, k);
                ws += w;
            }
            for (let k = 0; k < 3; k++) st[k] /= ws;
        }
        for (let k = 0; k < 3; k++) F[(qy * W2 + qx) * 3 + k] = gh[k]! * p.ghostIntensity * p.ghostTint[k]! + ha[k]! * p.haloIntensity * p.haloTint[k]! + st[k]! * p.streakIntensity * p.streakTint[k]!;
    }
    const dst = new Float64Array(W * H * 4), fl = new Float64Array(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        for (let k = 0; k < 3; k++) {
            const f = p.intensity === 0 ? 0 : p.intensity * bil(F, (x + 0.5) / W, (y + 0.5) / H, k);
            dst[i + k] = src[i + k]! + f;
            fl[i + k] = f;
        }
        dst[i + 3] = src[i + 3]!;
        fl[i + 3] = 1;
    }
    return { dst, fl };
}

type Dev = Parameters<typeof buildGraph>[0];
function lfGraph(device: Dev, W: number, H: number, props: Partial<P>, src: Float32Array, bright: Float32Array | null) {
    const scene = makeScene(device);
    const inputs: Record<string, ReturnType<typeof rgbaTexture>> = { "L.src": rgbaTexture(device, W, H, src) };
    if (bright) inputs["L.bright"] = rgbaTexture(device, W, H, bright);
    return buildGraph(device, scene, [["LensFlare", "L", props]], [], ["L.dst", "L.flare"], W, H, inputs);
}

function compare(gpu: Float32Array, ref: Float64Array) {
    let bad = 0, maxErr = 0;
    const n = gpu.length / 4;
    for (let k = 0; k < n; k++) {
        let e = 0;
        for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(gpu[k * 4 + c]! - ref[k * 4 + c]!) / Math.max(1, Math.abs(ref[k * 4 + c]!)));
        if (e > 1e-4) bad++;
        else maxErr = Math.max(maxErr, e);
    }
    return { bad, maxErr, n };
}

const kSpots: [number, number, number, V3][] = [[30, 40, 2, [8, 6, 4]], [100, 20, 3, [3, 3, 5]], [64, 90, 0, [40, 40, 40]]];

gpuTest("LensFlare.matchesCpuReference", async ({ device }) => {
    const cases: { name: string; W: number; H: number; props: Partial<P>; bright: boolean }[] = [
        { name: "default+streak", W: 128, H: 128, props: { streakIntensity: 2 }, bright: false },
        { name: "8 ghosts, wide CA, hard threshold, odd size", W: 127, H: 95, props: { ghostCount: 8, ghostSpacing: 0.5, chromatic: 0.02, knee: 0, threshold: 2, haloRadius: 0.3, haloFalloff: 2, intensity: 0.6, ghostTint: [0.6, 1, 0.7] }, bright: false },
        { name: "bright input, streak only", W: 128, H: 96, props: { threshold: 0, knee: 0, ghostIntensity: 0, haloIntensity: 0, streakIntensity: 4, streakLength: 0.8 }, bright: true },
    ];
    for (const c of cases) {
        const src = frame(c.W, c.H, kSpots);
        const bright = c.bright ? frame(c.W, c.H, [[64, 60, 1, [5, 5, 5]]], false) : null;
        const g = lfGraph(device, c.W, c.H, c.props, src, bright);
        g.execute(device.renderContext);
        const [dst, fl] = [await read(device, g.getOutput("L.dst")!), await read(device, g.getOutput("L.flare")!)];
        const ref = flareRef({ ...kDef, ...c.props }, src, bright, c.W, c.H);
        const a = compare(dst, ref.dst), b = compare(fl, ref.fl);
        let lit = 0;
        for (let k = 0; k < a.n; k++) if (ref.fl[k * 4]! + ref.fl[k * 4 + 1]! + ref.fl[k * 4 + 2]! > 1e-3) lit++;
        log(`LensFlare parity [${c.name}] ${c.W}x${c.H}: flare px ${lit}, dst off-reference ${a.bad} (max rel err ${a.maxErr.toExponential(2)}), flare off-reference ${b.bad} (max ${b.maxErr.toExponential(2)})`);
        expectEq(lit > 200, true, `${c.name}: flare visible`);
        expectEq(a.bad <= 0.001 * a.n && b.bad <= 0.001 * b.n, true, `${c.name}: GPU = CPU reference`);
    }
});

gpuTest("LensFlare.ghostMirror", async ({ device }) => {
    // One spot, one ghost (ghost 0 = the mirrored image), no halo/streak: the flare peak is at the mirrored point.
    const W = 128, H = 128;
    const src = frame(W, H, [[28, 36, 2, [10, 10, 10]]], false);
    const g = lfGraph(device, W, H, { ghostCount: 1, haloIntensity: 0, chromatic: 0, threshold: 1, knee: 0 }, src, null);
    g.execute(device.renderContext);
    const fl = await read(device, g.getOutput("L.flare")!);
    let best = -1, bx = 0, by = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const v = fl[(y * W + x) * 4]!; if (v > best) { best = v; bx = x; by = y; } }
    log(`LensFlare ghost 0 of a spot at (28, 36): peak at (${bx}, ${by}), expected about (${W - 1 - 28}, ${H - 1 - 36})`);
    expectEq(Math.abs(bx - (W - 1 - 28)) <= 2 && Math.abs(by - (H - 1 - 36)) <= 2 && best > 0.5, true, "ghost mirrored through the centre");
});

gpuTest("LensFlare.identity", async ({ device }) => {
    const W = 128, H = 128;
    const hot = frame(W, H, kSpots), dim = frame(W, H, []);
    for (const [name, props, src] of [["intensity 0", { intensity: 0, streakIntensity: 2 }, hot], ["nothing above threshold (knee 0)", { knee: 0, threshold: 1, streakIntensity: 2 }, dim]] as const) {
        const g = lfGraph(device, W, H, props, src, null);
        g.execute(device.renderContext);
        const dst = await read(device, g.getOutput("L.dst")!);
        const same = bitwiseEqual(dst, src);
        log(`LensFlare identity [${name}]: dst bitwise = src ${same}`);
        expectEq(same, true, `identity ${name}`);
    }
});

gpuTest("LensFlare.stateless", async ({ device }) => {
    const W = 128, H = 128;
    const a = frame(W, H, kSpots), b = frame(W, H, [[90, 90, 4, [20, 4, 2]]]);
    const props = { streakIntensity: 2, ghostCount: 6 };
    const cold = await read(device, (() => { const g = lfGraph(device, W, H, props, a, null); g.execute(device.renderContext); return g; })().getOutput("L.dst")!);
    const g = lfGraph(device, W, H, props, b, null);
    g.execute(device.renderContext);
    g.execute(device.renderContext);
    g.setInput("L.src", rgbaTexture(device, W, H, a));
    g.execute(device.renderContext);
    const again = await read(device, g.getOutput("L.dst")!);
    const same = bitwiseEqual(again, cold);
    log(`LensFlare stateless: frame A after frames of B on one graph bitwise = fresh graph ${same}`);
    expectEq(same, true, "stateless");
});
