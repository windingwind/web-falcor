/**
 * LightCookiePattern (procedural cookies for DeferredLighting's `cookie` input):
 *  - CPU parity: clouds, window and leaves patterns (several parameter sets, odd sizes, tint) against a CPU evaluation of
 *    the same formulas (PCG hash, periodic value noise, window/mullion distances, fleck discs);
 *  - tiling: every pattern is periodic (texel columns 0 and size-1 continue each other like inner neighbours);
 *  - wind: clouds at time t equal clouds at time 0 shifted by wind x t (an integer number of texels);
 *  - chain: pattern -> DeferredLighting.cookie on the test scene; (lit with cookie) / (lit without) = the CPU pattern
 *    projected by the directional light;
 *  - identity: strength 0 gives an exactly white cookie and a bitwise unchanged DeferredLighting output;
 *  - stateless: re-running after other times/modes on one graph = fresh graph, bit for bit.
 */

import { Properties, RenderGraph, createPass, type Device, type Scene } from "@web-falcor/falcor";
import { LightCookiePattern, type LightCookiePatternParams } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kSize, bitwiseEqual, log, makeScene, read, type V3 } from "./stylize-helpers.js";

let graphCount = 0;
/** One scene for every test of this file (a second Scene on the same device can render an empty G-buffer before the
 * staged-upload fix); env lighting is off where the lighting ratio is checked, so the sky only fills the background. */
const kSky: V3 = [0.3, 0.35, 0.4];

const kDef: LightCookiePatternParams = {
    mode: "clouds", size: 512, seed: 0, strength: 0.8, shadowColor: [0, 0, 0], time: 0, coverage: 0.5, softness: 0.15, scale: 4, octaves: 5,
    wind: [0.01, 0.004], rect: [0.2, 0.15, 0.8, 0.85], panes: [2, 3], frame: 0.04, mullion: 0.03, spotSize: 0.35, sway: 0.05, swaySpeed: 0.3,
};

// ---------------------------------------------------------------------------------------------------------------------
// CPU mirror of LightCookiePattern.cs.slang

const pcg = (v: number) => {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
};
const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const smoothstep = (a: number, b: number, x: number) => { const t = sat((x - a) / (b - a)); return t * t * (3 - 2 * t); };

function patternRef(props: Partial<LightCookiePatternParams>): { data: Float64Array; size: number } {
    const P = { ...kDef, ...props } as LightCookiePatternParams;
    const r = LightCookiePattern.resolved(P);
    const N = Math.round(P.size);
    const hash2 = (cx: number, cy: number, k: number) => (pcg((cx + pcg((cy + pcg((Math.imul(r.seed, 4) + k) >>> 0)) >>> 0)) >>> 0) >>> 8) / 16777216;
    const wrap = (c: number, p: number) => ((c % p) + p) % p;
    const noise = (x: number, y: number, period: number, oct: number) => {
        const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
        const q = (f: number) => f * f * f * (f * (f * 6 - 15) + 10);
        const ux = q(fx), uy = q(fy);
        const h = (dx: number, dy: number) => hash2(wrap(ix + dx, period), wrap(iy + dy, period), oct);
        const a = h(0, 0), b = h(1, 0), c = h(0, 1), d = h(1, 1);
        const top = a + (b - a) * ux, bot = c + (d - c) * ux;
        return top + (bot - top) * uy;
    };
    const clouds = (u: number, v: number) => {
        let qx = u - r.wind[0]! * r.time, qy = v - r.wind[1]! * r.time;
        qx -= Math.floor(qx); qy -= Math.floor(qy);
        let n = 0, amp = 1, sum = 0, period = r.scale;
        for (let o = 0; o < r.octaves; o++) { n += amp * noise(qx * period, qy * period, period, o); sum += amp; amp *= 0.5; period *= 2; }
        n /= Math.max(sum, 1e-6);
        const t = 1 - r.coverage;
        return smoothstep(t - r.softness, t + r.softness, n);
    };
    const windowOcc = (u: number, v: number) => {
        const rx = (u - r.rect[0]!) / Math.max(r.rect[2]! - r.rect[0]!, 1e-6), ry = (v - r.rect[1]!) / Math.max(r.rect[3]! - r.rect[1]!, 1e-6);
        if (rx < 0 || ry < 0 || rx > 1 || ry > 1) return 1;
        const s = r.softness;
        let open = smoothstep(r.frame - s, r.frame + s, Math.min(rx, 1 - rx, ry, 1 - ry));
        for (const [n, t0] of [[r.panes[0]!, rx], [r.panes[1]!, ry]] as const) {
            if (n < 2) continue;
            const t = t0 * n, k = Math.min(Math.max(Math.round(t), 1), n - 1);
            open *= smoothstep(0.5 * r.mullion - s, 0.5 * r.mullion + s, Math.abs(t - k) / n);
        }
        return 1 - open;
    };
    const leaves = (u: number, v: number) => {
        const M = r.scale, px = u * M, py = v * M, cx = Math.floor(px), cy = Math.floor(py);
        let light = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const ccx = cx + dx, ccy = cy + dy, wx = wrap(ccx, M), wy = wrap(ccy, M);
            const h1 = hash2(wx, wy, 11), h2 = hash2(wx, wy, 12), h3 = hash2(wx, wy, 13), h4 = hash2(wx, wy, 14);
            const ph = 6.2831853 * (r.time * r.swaySpeed + h3);
            const ex = ccx + 0.5 + 0.8 * (h1 - 0.5) + r.sway * Math.sin(ph), ey = ccy + 0.5 + 0.8 * (h2 - 0.5) + r.sway * Math.cos(1.3 * ph);
            const rad = r.spotSize * (0.5 + 0.5 * h4);
            light = Math.max(light, 1 - smoothstep(rad * (1 - r.softness), rad, Math.hypot(px - ex, py - ey)));
        }
        return 1 - light;
    };
    const data = new Float64Array(N * N * 4);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const u = (x + 0.5) / N, v = (y + 0.5) / N;
        const occ = r.strength === 0 ? 0 : r.mode === 0 ? clouds(u, v) : r.mode === 1 ? windowOcc(u, v) : leaves(u, v);
        const a = r.strength * occ, i = (y * N + x) * 4;
        for (let k = 0; k < 3; k++) data[i + k] = a === 0 ? 1 : 1 + (r.shadowColor[k]! - 1) * a;
        data[i + 3] = 1;
    }
    return { data, size: N };
}

// ---------------------------------------------------------------------------------------------------------------------

/** Pattern-only graph (no scene: the pass does not need one). */
function patternGraph(device: Device, props: Partial<LightCookiePatternParams>): RenderGraph {
    const g = new RenderGraph(device, `CookiePatternTest${++graphCount}`);
    g.onResize(64, 64);
    g.addPass(createPass(device, "LightCookiePattern", props), "C");
    g.markOutput("C.dst");
    return g;
}

async function renderPattern(device: Device, props: Partial<LightCookiePatternParams>): Promise<Float32Array> {
    const g = patternGraph(device, props);
    g.execute(device.renderContext);
    return read(device, g.getOutput("C.dst")!);
}

const kCases: [string, Partial<LightCookiePatternParams>][] = [
    ["clouds default", { size: 128 }],
    ["clouds dense, sharp, 3 octaves, seed 7, t = 37.5, blue shadow", { size: 96, coverage: 0.75, softness: 0.05, scale: 3, octaves: 3, seed: 7, time: 37.5, wind: [0.013, -0.021], shadowColor: [0.1, 0.15, 0.3] }],
    ["window 2x3", { mode: "window", size: 128, softness: 0.004 }],
    ["window 4x1, full rect, wide frame, odd size", { mode: "window", size: 101, rect: [0, 0, 1, 1], panes: [4, 1], frame: 0.1, mullion: 0.06, softness: 0.01, strength: 1 }],
    ["leaves default", { mode: "leaves", size: 128, scale: 6 }],
    ["leaves big flecks, sway, t = 3.2, warm shade", { mode: "leaves", size: 120, scale: 5, spotSize: 0.6, sway: 0.2, swaySpeed: 0.7, time: 3.2, softness: 0.5, seed: 3, shadowColor: [0.2, 0.25, 0.1] }],
];

gpuTest("LightCookiePattern.matchesCpuReference", async ({ device }) => {
    for (const [name, props] of kCases) {
        const gpu = await renderPattern(device, props);
        const ref = patternRef(props);
        let bad = 0, maxErr = 0, lo = Infinity, hi = -Infinity;
        const n = ref.size * ref.size;
        for (let k = 0; k < n; k++) {
            let e = 0;
            for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(gpu[k * 4 + c]! - ref.data[k * 4 + c]!));
            if (e > 1e-4) bad++;
            else maxErr = Math.max(maxErr, e);
            lo = Math.min(lo, ref.data[k * 4 + 1]!); hi = Math.max(hi, ref.data[k * 4 + 1]!);
        }
        log(`LightCookiePattern parity [${name}] ${ref.size}^2: range g [${lo.toFixed(3)}, ${hi.toFixed(3)}], off-reference ${bad}/${n} (max abs err ${maxErr.toExponential(2)})`);
        expectEq(hi - lo > 0.5, true, `${name}: pattern has contrast`);
        expectEq(bad <= 0.001 * n, true, `${name}: GPU = CPU reference`);
    }
});

gpuTest("LightCookiePattern.tilesSeamlessly", async ({ device }) => {
    // Edge jump (column N-1 -> column 0, row N-1 -> row 0) is no larger than the largest jump between inner neighbours.
    for (const [name, props] of kCases.filter(([n]) => !n.startsWith("window"))) {
        const N = props.size!;
        const img = await renderPattern(device, props);
        const at = (x: number, y: number) => img[(y * N + x) * 4 + 1]!;
        let inner = 0, edge = 0;
        for (let y = 0; y < N; y++) for (let x = 0; x < N - 1; x++) inner = Math.max(inner, Math.abs(at(x + 1, y) - at(x, y)), Math.abs(at(y, x + 1) - at(y, x)));
        for (let i = 0; i < N; i++) edge = Math.max(edge, Math.abs(at(0, i) - at(N - 1, i)), Math.abs(at(i, 0) - at(i, N - 1)));
        log(`LightCookiePattern tiling [${name}]: max edge jump ${edge.toFixed(4)}, max inner jump ${inner.toFixed(4)}`);
        expectEq(edge <= inner * 1.0001 + 1e-6, true, `${name}: periodic`);
    }
});

gpuTest("LightCookiePattern.windScrolls", async ({ device }) => {
    // wind x time = (5, -3) texels exactly: pattern(t) at texel (x, y) = pattern(0) at (x - 5, y + 3) (wrapped).
    const N = 128, wind = [0.01, -0.006], t = 5 / (N * 0.01);
    const a = await renderPattern(device, { size: N, wind, time: 0 }), b = await renderPattern(device, { size: N, wind, time: t });
    let maxErr = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const sx = (x - 5 + N) % N, sy = (y + 3) % N;
        maxErr = Math.max(maxErr, Math.abs(b[(y * N + x) * 4 + 1]! - a[(sy * N + sx) * 4 + 1]!));
    }
    log(`LightCookiePattern wind: shift by (5, -3) texels at t = ${t}, max err ${maxErr.toExponential(2)}`);
    expectEq(maxErr < 2e-3, true, "clouds scroll with the wind");
});

// ---------------------------------------------------------------------------------------------------------------------
// Chain: LightCookiePattern -> DeferredLighting.cookie

async function lit(device: Device, scene: Scene, cookie: Partial<LightCookiePatternParams> | null, lightProps: Record<string, unknown>) {
    const g = new RenderGraph(device, `CookiePatternChain${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    g.addPass(createPass(device, "DeferredLighting", lightProps), "L");
    g.addEdge("G.vbuffer", "L.vbuffer");
    g.addEdge("G.viewW", "L.viewW");
    if (cookie) {
        g.addPass(createPass(device, "LightCookiePattern", cookie), "C");
        g.addEdge("C.dst", "L.cookie");
    }
    g.markOutput("L.color");
    g.markOutput("G.posW");
    g.setScene(scene);
    for (let tries = 0; ; tries++) {
        g.execute(device.renderContext);
        const pos = await read(device, g.getOutput("G.posW")!);
        let hits = 0;
        for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
        if (hits > 0.2 * kSize * kSize) break;
        if (tries >= 4) throw new Error("G-buffer still empty");
    }
    g.execute(device.renderContext);
    return { color: await read(device, g.getOutput("L.color")!), pos: await read(device, g.getOutput("G.posW")!) };
}

function basis(n: V3): [V3, V3] {
    const sgn = n[2] >= 0 ? 1 : -1;
    const a = -1 / (sgn + n[2]);
    const b = n[0] * n[1] * a;
    return [[1 + sgn * n[0] * n[0] * a, sgn * b, -sgn * n[0]], [b, sgn + n[1] * n[1] * a, -n[1]]];
}

gpuTest("LightCookiePattern.drivesDeferredLighting", async ({ device }) => {
    const scene = makeScene(device, kSky);
    const lightProps = { brdf: "LambertGGX", shadowMode: "None", envLighting: false, emissive: false, cookieWorldSize: 3.1, cookieOffset: [0.13, 0.4] };
    const d = scene.getLight(0).dirW!;
    const len = Math.hypot(d.x, d.y, d.z);
    const [T, B] = basis([d.x / len, d.y / len, d.z / len]);
    const base = await lit(device, scene, null, lightProps);
    for (const [name, props] of [["clouds", { size: 64, time: 12 }], ["leaves", { mode: "leaves", size: 64, scale: 5 }], ["window", { mode: "window", size: 64, softness: 0.01 }]] as [string, Partial<LightCookiePatternParams>][]) {
        const withC = await lit(device, scene, props, lightProps);
        const ref = patternRef(props);
        const N = ref.size;
        const look = (u: number, v: number, c: number) => {
            const tx = u * N - 0.5, ty = v * N - 0.5, x0 = Math.floor(tx), y0 = Math.floor(ty), fx = tx - x0, fy = ty - y0;
            const at = (x: number, y: number) => ref.data[((((y % N) + N) % N) * N + (((x % N) + N) % N)) * 4 + c]!;
            const a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx, b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
            return a + (b - a) * fy;
        };
        let n = 0, maxErr = 0, dark = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            if (base.pos[i * 4 + 3] === 0) continue;
            const p: V3 = [base.pos[i * 4]!, base.pos[i * 4 + 1]!, base.pos[i * 4 + 2]!];
            let u = (p[0] * T[0] + p[1] * T[1] + p[2] * T[2]) / 3.1 + 0.13, v = (p[0] * B[0] + p[1] * B[1] + p[2] * B[2]) / 3.1 + 0.4;
            u -= Math.floor(u); v -= Math.floor(v);
            for (let k = 0; k < 3; k++) {
                const b0 = base.color[i * 4 + k]!;
                if (b0 < 1e-2) continue;
                n++;
                const ratio = withC.color[i * 4 + k]! / b0;
                if (ratio < 0.5) dark++;
                maxErr = Math.max(maxErr, Math.abs(ratio - look(u, v, k)));
            }
        }
        log(`LightCookiePattern -> DeferredLighting [${name}]: channels ${n} (${dark} shadowed below 0.5), max |ratio - cpu| ${maxErr.toExponential(2)}`);
        expectEq(n > 10000 && dark > 100, true, `${name}: pattern visible in the lighting`);
        expectEq(maxErr < 3e-3, true, `${name}: lighting ratio = CPU pattern`);
    }
});

gpuTest("LightCookiePattern.identity", async ({ device }) => {
    const white = await renderPattern(device, { size: 64, strength: 0, coverage: 1 });
    const allOne = white.every((v) => v === 1);
    log(`LightCookiePattern identity: strength 0 -> all texels exactly 1: ${allOne}`);
    expectEq(allOne, true, "white cookie");
    const scene = makeScene(device, kSky);
    for (const props of [{ brdf: "LambertGGX", shadowMode: "RayTraced" }, { brdf: "Material", shadowMode: "RayTraced" }]) {
        const a = await lit(device, scene, null, props);
        for (const mode of ["clouds", "window", "leaves"]) {
            const b = await lit(device, scene, { mode, size: 64, strength: 0 }, { ...props, cookieWorldSize: 2.3 });
            const same = bitwiseEqual(a.color, b.color);
            log(`LightCookiePattern identity [${props.brdf}, ${mode}, strength 0]: DeferredLighting bitwise = no cookie ${same}`);
            expectEq(same, true, `identity ${mode}`);
        }
    }
});

gpuTest("LightCookiePattern.stateless", async ({ device }) => {
    const X = { mode: "leaves", size: 96, time: 4.5, sway: 0.2 };
    const fresh = patternGraph(device, X);
    fresh.execute(device.renderContext);
    const x0 = await read(device, fresh.getOutput("C.dst")!);
    const g = patternGraph(device, { size: 96, time: 100 });
    g.execute(device.renderContext);
    g.getPass("C")!.setProperties(new Properties({ mode: "window", time: 1 }));
    g.execute(device.renderContext);
    g.getPass("C")!.setProperties(new Properties(X));
    g.execute(device.renderContext);
    const same = bitwiseEqual(await read(device, g.getOutput("C.dst")!), x0);
    log(`LightCookiePattern stateless: X after other modes/times on one graph bitwise = fresh graph ${same}`);
    expectEq(same, true, "stateless");
});
