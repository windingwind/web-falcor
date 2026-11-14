/**
 * SSGI (screen-space one-bounce diffuse GI, visibility bitmask) and its painted bounce.
 *  1. Path-traced reference: in an open Cornell-like box (red left wall, green right wall, white floor and back wall, two
 *     directional lights, no shadows, no sky) the SSGI incident radiance on the floor is compared with a CPU one-bounce
 *     path-traced reference (cosine-sampled rays to the walls, wall radiance = analytic Lambert direct light, checked
 *     against DeferredLighting.diffuse on screen): luminance correlation, overall ratio, and the sign of the colour
 *     bleeding (red - green) per pixel and per floor half.
 *  2. Identity: intensity 0 gives the input colour bit for bit on every frame of a moving shot (physical and painted).
 *  3. Composite parity: indirect / dst = CPU composite of the gi output, albedo and normal (physical and painted: hue
 *     rotation, saturation, tint, sky/ground hemisphere, AO weight); a neutral painted setting equals physical.
 *  4. Temporal accumulation reduces noise; with temporal off the output is the same every frame (no state); softness
 *     reduces noise further.
 *  5. Cold vs consecutive: shot X on a fresh graph = shot X after shots A and B on one graph with a camera reset, a prime
 *     render or resetGraphHistory in between (bit for bit; full and half resolution, painted); without a cut X differs
 *     (control).
 *  6. No banding: unknown band/ramp/level/quantise properties are rejected; a sweep of extreme painted parameters over a
 *     smooth gradient input keeps the output continuous (no luminance histogram with <= 8 dominant levels), while a
 *     6-level quantised control is flagged by the same metric.
 */

import { LightType, Scene, ResourceBindFlags, ResourceFormat, createPass, float2, float3, float4, type Device, type RenderGraph, type Texture } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kSize, log, makeScene, read, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, renderShot, warmUp, type Pose } from "./fx-helpers.js";

// ---------------------------------------------------------------------------
// Scenes

type Vtx = { position: float3; normal: float3; tangent: float4; texCrd: float2 };
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = (a: V3): V3 => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** Quad with the winding chosen so that its geometric normal is `n` (front face towards n). */
function quad(c: V3[], n: V3, materialID: number) {
    if (dot(cross(sub(c[1]!, c[0]!), sub(c[2]!, c[0]!)), n) < 0) c = [c[0]!, c[3]!, c[2]!, c[1]!];
    const uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
    const vertices: Vtx[] = c.map((p, i) => ({ position: new float3(...p), normal: new float3(...n), tangent: new float4(1, 0, 0, 1), texCrd: uv[i]! }));
    return { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), materialID };
}

const kAlb: V3[] = [[0.8, 0.8, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.8, 0.8, 0.8]];
/** Light travel directions and intensities (no shadows in the test chain). */
const kLights: { dir: V3; I: number }[] = [{ dir: [0.6, -1, -0.3], I: 2 }, { dir: [-0.6, -1, -0.3], I: 2 }];
/** Inner wall planes of the box x, z in [-1, 1], y in [0, 2]: normal (inwards), material. */
const kWalls: { axis: 0 | 2; at: number; n: V3; mat: number }[] = [
    { axis: 0, at: -1, n: [1, 0, 0], mat: 1 },
    { axis: 0, at: 1, n: [-1, 0, 0], mat: 2 },
    { axis: 2, at: -1, n: [0, 0, 1], mat: 3 },
];
const kCornellCam: Pose = { pos: [0, 1.2, 3.6], target: [0, 0.8, 0] };

const kScenes = new WeakMap<Device, Map<string, Scene>>();
function cached(device: Device, key: string, mk: () => Scene): Scene {
    let m = kScenes.get(device);
    if (!m) kScenes.set(device, (m = new Map()));
    let s = m.get(key);
    if (!s) m.set(key, (s = mk()));
    return s;
}

function setCam(scene: Scene, p: Pose) {
    scene.camera.setPosition(new float3(...p.pos));
    scene.camera.setTarget(new float3(...p.target));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
}

/** Open (no ceiling, no front) box with coloured side walls; each wall has an outer shell 0.02 behind it. */
function cornellScene(device: Device): Scene {
    const scene = cached(device, "cornell", () => {
        const meshes = [quad([[-1, 0, 1], [1, 0, 1], [1, 0, -1], [-1, 0, -1]], [0, 1, 0], 0)];
        for (const w of kWalls) {
            for (const shell of [0, 1]) {
                const at = w.at - (shell ? 0.02 * Math.sign(w.n[w.axis]) : 0);
                const n: V3 = shell ? [-w.n[0], -w.n[1], -w.n[2]] : w.n;
                const c: V3[] = w.axis === 0
                    ? [[at, 0, -1], [at, 0, 1], [at, 2, 1], [at, 2, -1]]
                    : [[-1, 0, at], [1, 0, at], [1, 2, at], [-1, 2, at]];
                meshes.push(quad(c, n, w.mat));
            }
        }
        const mats = kAlb.map((c) => ({ basic: { baseColor: new float4(c[0], c[1], c[2], 1), specular: new float4(0, 1, 0, 0) } }));
        return new Scene(device, meshes, mats, kLights.map((l) => ({ type: LightType.Directional, dirW: new float3(...l.dir), intensity: new float3(l.I, l.I, l.I) })));
    });
    setCam(scene, kCornellCam);
    return scene;
}

/** Floor + sphere for the banding sweep (smooth normals and a smooth bounce). */
const kSphC: V3 = [0, 0.5, 0], kSphR = 0.5;
function sphereScene(device: Device): Scene {
    const scene = cached(device, "sphere", () => {
        const nLon = 128, nLat = 64, vertices: Vtx[] = [], idx: number[] = [];
        for (let a = 0; a <= nLat; a++) {
            const th = (Math.PI * a) / nLat;
            for (let b = 0; b <= nLon; b++) {
                const ph = (2 * Math.PI * b) / nLon;
                const n: V3 = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
                vertices.push({ position: new float3(kSphC[0] + n[0] * kSphR, kSphC[1] + n[1] * kSphR, kSphC[2] + n[2] * kSphR), normal: new float3(...n), tangent: new float4(-Math.sin(ph), 0, Math.cos(ph), 1), texCrd: new float2(b / nLon, a / nLat) });
            }
        }
        for (let a = 0; a < nLat; a++) for (let b = 0; b < nLon; b++) {
            const i0 = a * (nLon + 1) + b, i1 = i0 + 1, i2 = i0 + nLon + 1, i3 = i2 + 1;
            idx.push(i0, i1, i2, i1, i3, i2);
        }
        const f = 6;
        const meshes = [quad([[-f, 0, f], [f, 0, f], [f, 0, -f], [-f, 0, -f]], [0, 1, 0], 0), { vertices, indices: new Uint32Array(idx), materialID: 1 }];
        const mats = [[0.7, 0.7, 0.7], [0.75, 0.75, 0.75]].map((c) => ({ basic: { baseColor: new float4(c[0]!, c[1]!, c[2]!, 1), specular: new float4(0, 1, 0, 0) } }));
        return new Scene(device, meshes, mats, [{ type: LightType.Directional, dirW: new float3(0.3, -1, -0.4), intensity: new float3(3, 3, 3) }]);
    });
    setCam(scene, { pos: [0, 1.0, 1.6], target: [0, 0.45, 0] });
    return scene;
}

// ---------------------------------------------------------------------------
// Graphs

const kBase: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"]];
const kSsgi: [string, string][] = [["G.depth", "S.depth"], ["G.normW", "S.normW"], ["G.diffuseOpacity", "S.diffuseOpacity"], ["G.mvec", "S.mvec"]];

function ssgiGraph(device: Device, scene: Scene, props: Record<string, unknown>, outputs: string[], opts: { radiance?: string; hiz?: boolean; inputs?: Record<string, Texture>; color?: boolean } = {}): RenderGraph {
    const passes: [string, string, Record<string, unknown>][] = [["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: "None" }]];
    const edges = [...kBase, ...kSsgi];
    if (opts.color !== false) edges.push(["L.color", "S.color"]);
    if (opts.radiance) edges.push([opts.radiance, "S.radiance"]);
    if (opts.hiz) {
        passes.push(["HiZBuild", "Z", {}]);
        edges.push(["G.depth", "Z.depth"], ["Z.hiz", "S.hiz"]);
    }
    passes.push(["SSGI", "S", props]);
    return buildGraph(device, scene, passes, edges, outputs, kSize, kSize, opts.inputs ?? {});
}

async function readOuts(device: Device, g: RenderGraph, outs: string[]): Promise<Float32Array[]> {
    const r: Float32Array[] = [];
    for (const o of outs) r.push(await read(device, g.getOutput(o)!));
    return r;
}

const lum = (b: ArrayLike<number>, i: number) => 0.2126 * b[i * 4]! + 0.7152 * b[i * 4 + 1]! + 0.0722 * b[i * 4 + 2]!;

// ---------------------------------------------------------------------------
// 1. Path-traced reference

/** Analytic direct diffuse radiance of a surface with normal n and albedo a (Lambert, no shadows). */
function direct(n: V3, a: V3): V3 {
    let e = 0;
    for (const l of kLights) e += l.I * Math.max(dot(n, nrm([-l.dir[0], -l.dir[1], -l.dir[2]])), 0);
    return a.map((x) => (x / Math.PI) * e) as V3;
}

/** One-bounce incident radiance / pi at a floor point (cosine-sampled rays to the inner walls; misses are black). */
function referenceGi(P: V3, sqrtN: number): V3 {
    const acc: V3 = [0, 0, 0];
    for (let a = 0; a < sqrtN; a++) for (let b = 0; b < sqrtN; b++) {
        const u1 = (a + 0.5) / sqrtN, u2 = (b + 0.5) / sqrtN;
        const r = Math.sqrt(u1), ph = 2 * Math.PI * u2;
        const d: V3 = [r * Math.cos(ph), Math.sqrt(Math.max(1 - u1, 0)), r * Math.sin(ph)];
        let best = Infinity, L: V3 | null = null;
        for (const w of kWalls) {
            const dd = d[w.axis];
            if (Math.abs(dd) < 1e-9) continue;
            const t = (w.at - P[w.axis]) / dd;
            if (!(t > 1e-6) || t >= best) continue;
            const H: V3 = [P[0] + d[0] * t, P[1] + d[1] * t, P[2] + d[2] * t];
            const o = w.axis === 0 ? 2 : 0;
            if (H[1] < 0 || H[1] > 2 || H[o] < -1 || H[o] > 1) continue;
            best = t;
            L = direct(w.n, kAlb[w.mat]!);
        }
        if (L) for (let c = 0; c < 3; c++) acc[c] += L[c]!;
    }
    return acc.map((x) => x / (sqrtN * sqrtN)) as V3;
}

interface RefStats { n: number; corr: number; ratio: number; signAgree: number; leftRG: number; rightRG: number; refLeftRG: number; refRightRG: number }

function compareToReference(gi: Float32Array, pos: Float32Array, ref: Map<number, V3>): RefStats {
    const xs: number[] = [], ys: number[] = [];
    let sg = 0, sr = 0, agree = 0, nSign = 0, maxC = 0;
    const halves = { l: [0, 0, 0], r: [0, 0, 0] }; // sum(R-G) gpu, sum ref, count
    for (const [, v] of ref) maxC = Math.max(maxC, Math.abs(v[0] - v[1]));
    for (const [i, v] of ref) {
        const g: V3 = [gi[i * 4]!, gi[i * 4 + 1]!, gi[i * 4 + 2]!];
        const lg = 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2], lr = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
        xs.push(lg); ys.push(lr); sg += lg; sr += lr;
        const cr = v[0] - v[1], cg = g[0] - g[1];
        if (Math.abs(cr) > 0.2 * maxC) { nSign++; if (Math.sign(cr) === Math.sign(cg)) agree++; }
        const h = pos[i * 4]! < -0.3 ? halves.l : pos[i * 4]! > 0.3 ? halves.r : null;
        if (h) { h[0] += cg; h[1] += cr; h[2]++; }
    }
    const n = xs.length, mx = sg / n, my = sr / n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let k = 0; k < n; k++) { sxy += (xs[k]! - mx) * (ys[k]! - my); sxx += (xs[k]! - mx) ** 2; syy += (ys[k]! - my) ** 2; }
    return {
        n, corr: sxy / Math.sqrt(sxx * syy), ratio: sg / sr, signAgree: agree / Math.max(nSign, 1),
        leftRG: halves.l[0]! / halves.l[2]!, rightRG: halves.r[0]! / halves.r[2]!, refLeftRG: halves.l[1]! / halves.l[2]!, refRightRG: halves.r[1]! / halves.r[2]!,
    };
}

const fmt = (s: RefStats) => `n ${s.n}, lum corr ${s.corr.toFixed(3)}, ratio ${s.ratio.toFixed(3)}, bleeding sign agree ${(100 * s.signAgree).toFixed(1)} %, R-G left ${s.leftRG.toFixed(4)} (ref ${s.refLeftRG.toFixed(4)}) right ${s.rightRG.toFixed(4)} (ref ${s.refRightRG.toFixed(4)})`;

gpuTest("SSGI.cornellMatchesPathTracedReference", async ({ device }) => {
    const scene = cornellScene(device);
    await warmUp(device, scene);
    cornellScene(device);
    scene.camera.resetHistory();
    const outs = ["S.gi", "L.diffuse", "G.posW", "G.normW"];
    const exact = { temporal: false, sliceCount: 16, stepCount: 24, radius: 4, thickness: 0.3, denoiseIterations: 0, maxRadiusPx: 1 };
    const g = ssgiGraph(device, scene, exact, outs, { radiance: "L.diffuse" });
    g.execute(device.renderContext);
    const [gi, dif, pos, nor] = await readOuts(device, g, outs);

    // Wall radiance model of the reference vs DeferredLighting.diffuse on screen.
    let dErr = 0, dN = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        if (pos![i * 4 + 3] === 0) continue;
        const P: V3 = [pos![i * 4]!, pos![i * 4 + 1]!, pos![i * 4 + 2]!];
        let surf: { n: V3; a: V3 } | null = null;
        if (Math.abs(P[1]) < 1e-4) surf = { n: [0, 1, 0], a: kAlb[0]! };
        else for (const w of kWalls) if (Math.abs(P[w.axis] - w.at) < 1e-4 && dot([nor![i * 4]!, nor![i * 4 + 1]!, nor![i * 4 + 2]!], w.n) > 0.99) surf = { n: w.n, a: kAlb[w.mat]! };
        if (!surf) continue;
        const ref = direct(surf.n, surf.a);
        for (let c = 0; c < 3; c++) dErr = Math.max(dErr, Math.abs(dif![i * 4 + c]! - ref[c]!) / Math.max(ref[c]!, 1e-3));
        dN++;
    }
    log(`SSGI reference: wall/floor direct radiance model vs DeferredLighting.diffuse: ${dN} px, max rel err ${dErr.toExponential(2)}`);
    expectEq(dN > 2000 && dErr < 1e-3, true, "reference direct-light model matches DeferredLighting.diffuse");

    const ref = new Map<number, V3>();
    for (let y = 0; y < kSize; y++) for (let x = 0; x < kSize; x++) {
        const i = y * kSize + x;
        if (pos![i * 4 + 3] === 0) continue;
        const P: V3 = [pos![i * 4]!, pos![i * 4 + 1]!, pos![i * 4 + 2]!];
        if (Math.abs(P[1]) > 1e-4 || Math.abs(P[0]) > 0.95 || P[2] < -0.95 || P[2] > 0.95) continue;
        ref.set(i, referenceGi(P, 16));
    }
    const s1 = compareToReference(gi!, pos!, ref);
    log(`SSGI vs path-traced one bounce (16 slices x 24 steps, full res, no denoise/temporal): ${fmt(s1)}`);
    expectEq(s1.n > 500, true, "enough floor pixels");
    expectEq(s1.corr >= 0.8, true, "luminance correlation >= 0.8");
    expectEq(s1.ratio >= 0.6 && s1.ratio <= 1.4, true, "overall indirect within 0.6-1.4 of the reference");
    expectEq(s1.signAgree >= 0.9, true, "bleeding colour (red - green) has the reference sign on >= 90 % of the strongly tinted pixels");
    expectEq(s1.leftRG > 0 && s1.rightRG < 0 && s1.refLeftRG > 0 && s1.refRightRG < 0, true, "red bleeds onto the left floor half, green onto the right");

    // Default sampling (4 slices x 8 steps, denoise, 16 temporal frames), search radius not clamped by the frame share;
    // without and with Hi-Z depth, then at half resolution.
    for (const [label, props, hiz] of [["4x8, 16 frames", {}, false], ["4x8 + Hi-Z, 16 frames", {}, true], ["4x8 + Hi-Z, half res, 16 frames", { resolution: "half" }, true]] as [string, Record<string, unknown>, boolean][]) {
        cornellScene(device);
        scene.camera.resetHistory();
        const g2 = ssgiGraph(device, scene, { radius: 4, thickness: 0.3, maxRadiusPx: 1, ...props }, ["S.gi"], { radiance: "L.diffuse", hiz });
        const fr = await renderShot(device, scene, g2, kCornellCam, 16, ["S.gi"]);
        const s2 = compareToReference(fr[15]![0]!, pos!, ref);
        log(`SSGI vs path-traced one bounce (${label}): ${fmt(s2)}`);
        expectEq(s2.corr >= 0.7 && s2.ratio >= 0.5 && s2.ratio <= 1.5 && s2.signAgree >= 0.85, true, `${label}: close to the reference`);
    }
});

// ---------------------------------------------------------------------------
// 2. Identity

gpuTest("SSGI.intensityZeroIsIdentity", async ({ device }) => {
    const scene = cornellScene(device);
    await warmUp(device, scene);
    for (const mode of ["physical", "painted"]) {
        cornellScene(device);
        scene.camera.resetHistory();
        const outs = ["S.dst", "L.color", "S.indirect"];
        const g = ssgiGraph(device, scene, { intensity: 0, mode, radius: 4 }, outs);
        const frames = await renderShot(device, scene, g, kCornellCam, 4, outs, 0.05);
        const same = frames.every((row) => bitwiseEqual(row[0]!, row[1]!));
        const active = frames.some((row) => row[2]!.some((v, i) => i % 4 !== 3 && v > 1e-4));
        log(`SSGI ${mode} intensity 0: dst = color bit for bit on ${frames.length} moving frames ${same}; indirect computed ${active}`);
        expectEq(active, true, `${mode}: the shot has indirect light`);
        expectEq(same, true, `${mode}: intensity 0 = identity`);
    }
});

// ---------------------------------------------------------------------------
// 3. Composite parity

const kPaint = { hueShift: 35, saturation: 1.6, tint: [1, 0.9, 0.8] as V3, skyColor: [0.5, 0.6, 0.9] as V3, groundColor: [0.4, 0.3, 0.2] as V3, hemiIntensity: 0.4, hemiOcclusion: 0.5, upVector: [0, 1, 0] as V3 };

function paintCpu(b: V3, p: typeof kPaint): V3 {
    const Y = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2];
    const I = 0.595716 * b[0] - 0.274453 * b[1] - 0.321263 * b[2];
    const Q = 0.211456 * b[0] - 0.522591 * b[1] + 0.311135 * b[2];
    const h = (p.hueShift * Math.PI) / 180, cs = Math.cos(h), sn = Math.sin(h);
    const I2 = p.saturation * (cs * I - sn * Q), Q2 = p.saturation * (sn * I + cs * Q);
    const rgb: V3 = [Y + 0.9563 * I2 + 0.621 * Q2, Y - 0.2721 * I2 - 0.6474 * Q2, Y - 1.107 * I2 + 1.7046 * Q2];
    return rgb.map((x, c) => Math.max(x, 0) * p.tint[c]!) as V3;
}

gpuTest("SSGI.compositeParity", async ({ device }) => {
    const scene = cornellScene(device);
    await warmUp(device, scene);
    const outs = ["S.dst", "S.indirect", "S.gi", "L.color", "G.diffuseOpacity", "G.normW", "G.posW"];
    const res: Record<string, Float32Array[]> = {};
    for (const [name, props] of [["physical", {}], ["painted", { mode: "painted", ...kPaint }],
        ["neutral", { mode: "painted", hueShift: 0, saturation: 1, tint: [1, 1, 1], hemiIntensity: 0 }]] as [string, Record<string, unknown>][]) {
        cornellScene(device);
        scene.camera.resetHistory();
        const g = ssgiGraph(device, scene, { temporal: false, radius: 4, intensity: 1.5, ...props }, outs);
        g.execute(device.renderContext);
        res[name] = await readOuts(device, g, outs);
    }
    for (const name of ["physical", "painted"]) {
        const [dst, ind, gi, col, alb, nor, pos] = res[name]!;
        let err = 0, n = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            if (pos![i * 4 + 3] === 0) continue;
            n++;
            const b: V3 = [gi![i * 4]!, gi![i * 4 + 1]!, gi![i * 4 + 2]!];
            let add: V3;
            if (name === "physical") add = b.map((x, c) => alb![i * 4 + c]! * x) as V3;
            else {
                const N = nrm([nor![i * 4]!, nor![i * 4 + 1]!, nor![i * 4 + 2]!]);
                const k = 0.5 + 0.5 * dot(N, kPaint.upVector);
                const ao = gi![i * 4 + 3]!, occ = 1 + (ao - 1) * kPaint.hemiOcclusion;
                const pb = paintCpu(b, kPaint);
                add = pb.map((x, c) => alb![i * 4 + c]! * (x + (kPaint.groundColor[c]! + (kPaint.skyColor[c]! - kPaint.groundColor[c]!) * k) * kPaint.hemiIntensity * occ)) as V3;
            }
            for (let c = 0; c < 3; c++) {
                err = Math.max(err, Math.abs(ind![i * 4 + c]! - add[c]!) / Math.max(Math.abs(add[c]!), 1e-2));
                err = Math.max(err, Math.abs(dst![i * 4 + c]! - (col![i * 4 + c]! + 1.5 * add[c]!)) / Math.max(Math.abs(col![i * 4 + c]!) + 1.5 * Math.abs(add[c]!), 1e-2));
            }
        }
        log(`SSGI composite parity ${name}: ${n} px, max rel err ${err.toExponential(2)}`);
        expectEq(n > 2000 && err < 1e-4, true, `${name}: indirect and dst = CPU composite`);
    }
    // Hue: the painted bounce on the red-tinted floor is rotated by the hue shift (YIQ angle) relative to physical.
    const giP = res.physical![2]!;
    let dh = 0, nh = 0, nErr = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        if (res.physical![6]![i * 4 + 3] === 0) continue;
        const b: V3 = [giP[i * 4]!, giP[i * 4 + 1]!, giP[i * 4 + 2]!];
        const I = 0.595716 * b[0] - 0.274453 * b[1] - 0.321263 * b[2], Q = 0.211456 * b[0] - 0.522591 * b[1] + 0.311135 * b[2];
        const p = paintCpu(b, { ...kPaint, tint: [1, 1, 1] });
        const I2 = 0.595716 * p[0] - 0.274453 * p[1] - 0.321263 * p[2], Q2 = 0.211456 * p[0] - 0.522591 * p[1] + 0.311135 * p[2];
        if (Math.hypot(I, Q) > 0.02 && Math.min(...p) > 0) { let d = Math.atan2(Q2, I2) - Math.atan2(Q, I); d = Math.atan2(Math.sin(d), Math.cos(d)); dh += d; nh++; }
        for (let c = 0; c < 4; c++) nErr = Math.max(nErr, Math.abs(res.neutral![1]![i * 4 + c]! - res.physical![1]![i * 4 + c]!));
    }
    log(`SSGI painted hue rotation on tinted pixels: mean ${(dh / Math.max(nh, 1) * 180 / Math.PI).toFixed(2)} deg over ${nh} px (set 35); neutral painted vs physical indirect max abs diff ${nErr.toExponential(2)}`);
    expectEq(nh > 200 && Math.abs(dh / nh * 180 / Math.PI - 35) < 0.5, true, "hue rotated by hueShift");
    expectEq(nErr < 1e-5, true, "neutral painted = physical");
});

// ---------------------------------------------------------------------------
// 4. Temporal, stateless, softness

function noise(buf: Float32Array, pos: Float32Array): number {
    let s = 0, n = 0;
    for (let y = 0; y < kSize; y++) for (let x = 0; x + 1 < kSize; x++) {
        const i = y * kSize + x, j = i + 1;
        if (pos[i * 4 + 3] === 0 || pos[j * 4 + 3] === 0) continue;
        for (let c = 0; c < 3; c++) s += Math.abs(buf[i * 4 + c]! - buf[j * 4 + c]!);
        n++;
    }
    return n ? s / n : 0;
}

gpuTest("SSGI.temporalAccumulationAndStateless", async ({ device }) => {
    const scene = cornellScene(device);
    await warmUp(device, scene);
    scene.camera.resetHistory();
    const props = { radius: 4, denoiseIterations: 0, temporalAlpha: 0.9 };
    const g = ssgiGraph(device, scene, props, ["S.gi", "G.posW"]);
    const fr = await renderShot(device, scene, g, kCornellCam, 12, ["S.gi", "G.posW"]);
    const pos = fr[0]![1]!;
    const n0 = noise(fr[0]![0]!, pos), n11 = noise(fr[11]![0]!, pos);
    cornellScene(device);
    const g2 = ssgiGraph(device, scene, { ...props, temporal: false }, ["S.gi"]);
    const st = await renderShot(device, scene, g2, kCornellCam, 3, ["S.gi"]);
    const stateless = bitwiseEqual(st[0]![0]!, st[2]![0]!);
    const g3 = ssgiGraph(device, scene, { ...props, temporal: false, mode: "painted", softness: 1 }, ["S.gi"]);
    const so = await renderShot(device, scene, g3, kCornellCam, 1, ["S.gi"]);
    const nSoft = noise(so[0]![0]!, pos), nRaw = noise(st[0]![0]!, pos);
    log(`SSGI temporal: neighbour noise frame 0 ${n0.toExponential(3)} -> frame 11 ${n11.toExponential(3)}; temporal off: frames 0 and 2 bitwise equal ${stateless}; softness 1 noise ${nSoft.toExponential(3)} vs raw ${nRaw.toExponential(3)}`);
    expectEq(n0 > 0 && n11 < 0.7 * n0, true, "temporal accumulation reduces the noise");
    expectEq(stateless, true, "temporal off: no state");
    expectEq(nSoft < 0.5 * nRaw, true, "softness smooths the bounce");
});

// ---------------------------------------------------------------------------
// 5. Cold vs consecutive

gpuTest("SSGI.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device);
    const outs = ["S.dst", "S.gi"];
    const mk = (props: Record<string, unknown>) => () => ssgiGraph(device, scene, { radius: 1.5, ...props }, outs, { hiz: true });
    for (const [label, props] of [["physical full", {}], ["painted half", { mode: "painted", resolution: "half", softness: 0.5 }]] as [string, Record<string, unknown>][]) {
        const res = await coldVsConsecutive(device, scene, mk(props), outs, ["camera-reset", "prime", "explicit", "none"]);
        for (const r of res) log(`SSGI ${label} cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
        for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `${label}: cold = consecutive after ${r.method}`);
        expectEq(res.find((r) => r.method === "none")!.differs.some((d) => d), true, `${label}: without a cut the history changes X (control)`);
    }
    const off = await coldVsConsecutive(device, scene, mk({ temporal: false }), outs, ["none"]);
    log(`SSGI temporal off, no cut: X frames bitwise equal ${off[0]!.equalFrames}/${off[0]!.frames}`);
    expectEq(off[0]!.equalFrames, off[0]!.frames, "temporal off: stateless (no cut needed)");
});

// ---------------------------------------------------------------------------
// 6. No banding

gpuTest("SSGI.rejectsQuantisedProperties", async ({ device }) => {
    const bad = ["bands", "bandCount", "rampTexture", "levels", "quantize", "quantise", "posterize", "toonSteps"];
    const rejected = bad.filter((k) => { try { createPass(device, "SSGI", { [k]: 4 }); return false; } catch { return true; } });
    let okDefault = true;
    try { createPass(device, "SSGI", { mode: "painted", hueShift: -180, saturation: 10 }); } catch { okDefault = false; }
    log(`SSGI rejected properties: ${rejected.join(", ")} (${rejected.length}/${bad.length}); continuous painted params accepted ${okDefault}`);
    expectEq(rejected.length, bad.length, "every band/ramp/level/quantise property is rejected");
    expectEq(okDefault, true, "valid painted parameters are accepted");
});

/**
 * Banding metric on the sphere pixels with signal: distinct values and the share of the 8 fullest of 1024 equal luminance
 * bins over [p1, p99] (a banded image with <= 8 levels puts ~100 % of its pixels into <= 8 bins).
 */
function banding(img: Float32Array, mask: boolean[]): { n: number; distinct: number; top8: number } {
    const ls: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) ls.push(lum(img, i));
    const mx = Math.max(...ls);
    const v = ls.filter((x) => x > 1e-3 * mx).sort((a, b) => a - b);
    const lo = v[Math.floor(0.01 * v.length)]!, hi = v[Math.floor(0.99 * v.length)]!;
    const kBins = 1024, bins = new Array(kBins).fill(0);
    for (const x of v) bins[Math.min(kBins - 1, Math.max(0, Math.floor(((x - lo) / Math.max(hi - lo, 1e-12)) * kBins)))]++;
    bins.sort((a, b) => b - a);
    return { n: v.length, distinct: new Set(v).size, top8: bins.slice(0, 8).reduce((a, b) => a + b, 0) / Math.max(v.length, 1) };
}

gpuTest("SSGI.paintedSweepStaysContinuous", async ({ device }) => {
    const scene = sphereScene(device);
    await warmUp(device, scene);
    // Smooth gradient input: the bounced radiance is an external colour ramp, the base colour is black.
    const grad = new Float32Array(kSize * kSize * 4), black = new Float32Array(kSize * kSize * 4);
    for (let y = 0; y < kSize; y++) for (let x = 0; x < kSize; x++) {
        const i = (y * kSize + x) * 4, u = x / (kSize - 1), v = y / (kSize - 1);
        grad.set([2 * u, 2 * v, 2 * (1 - u), 1], i);
        black[i + 3] = 1;
    }
    const tex = (d: Float32Array) => device.createTexture2D(kSize, kSize, ResourceFormat.RGBA32Float, 1, 1, d, ResourceBindFlags.ShaderResource);
    const inputs = { "S.radiance": tex(grad), "S.color": tex(black) };
    const configs: Record<string, unknown>[] = [
        { mode: "physical" },
        { hueShift: -180, saturation: 0 },
        { hueShift: 180, saturation: 10, tint: [4, 0.01, 1] },
        { hueShift: 90, saturation: 3, hemiIntensity: 5, hemiOcclusion: 1, softness: 1 },
        { hueShift: 0, saturation: 1, hemiIntensity: 0, softness: 0, denoiseIterations: 0 },
        { hueShift: -45, saturation: 0.2, hemiIntensity: 20, hemiOcclusion: 0, skyColor: [1, 1, 1], groundColor: [0, 0, 0] },
        { hueShift: 170, saturation: 6, hemiIntensity: 2, resolution: "half", intensity: 100 },
        { hueShift: 10, saturation: 2, intensity: 0.001, sliceCount: 1, stepCount: 1 },
    ];
    let mask: boolean[] | null = null;
    let worst = { top8: 0, distinct: Infinity };
    let allOk = true;
    for (const c of configs) {
        sphereScene(device);
        scene.camera.resetHistory();
        const g = ssgiGraph(device, scene, { mode: "painted", radius: 1, temporal: false, ...c }, ["S.dst", "G.posW"], { inputs, color: false });
        g.execute(device.renderContext);
        const [dst, pos] = await readOuts(device, g, ["S.dst", "G.posW"]);
        mask ??= Array.from({ length: kSize * kSize }, (_, i) => pos![i * 4 + 3] !== 0 && Math.abs(Math.hypot(pos![i * 4]! - kSphC[0], pos![i * 4 + 1]! - kSphC[1], pos![i * 4 + 2]! - kSphC[2]) - kSphR) < 1e-3);
        const b = banding(dst!, mask);
        const ok = b.n > 300 && b.distinct > 200 && b.top8 < 0.5;
        allOk &&= ok;
        worst = { top8: Math.max(worst.top8, b.top8), distinct: Math.min(worst.distinct, b.distinct) };
        log(`SSGI sweep ${JSON.stringify(c)}: px ${b.n}, distinct ${b.distinct}, top-8-of-1024-bin share ${(100 * b.top8).toFixed(1)} % ${ok ? "continuous" : "BANDED"}`);
        if (c === configs[2]) {
            // Control: the same image quantised to 6 luminance levels must be flagged by the metric.
            const q = dst!.slice();
            const ls = mask.map((m, i) => (m ? lum(q, i) : 0)), mx = Math.max(...ls);
            for (let i = 0; i < mask.length; i++) if (mask[i] && ls[i]! > 0) {
                const s = (Math.min(5, Math.floor((ls[i]! / mx) * 6)) + 0.5) / 6 * mx / ls[i]!;
                for (let ch = 0; ch < 3; ch++) q[i * 4 + ch] = q[i * 4 + ch]! * s;
            }
            const bq = banding(q, mask);
            log(`SSGI sweep control (6-level quantised copy): distinct ${bq.distinct}, top-8 share ${(100 * bq.top8).toFixed(1)} %`);
            expectEq(bq.top8 >= 0.95, true, "the metric flags a 6-level quantised image");
        }
    }
    log(`SSGI sweep: ${configs.length} extreme configs, worst top-8 share ${(100 * worst.top8).toFixed(1)} %, min distinct ${worst.distinct}`);
    expectEq(allOk, true, "every extreme painted configuration stays continuous");
});
