/**
 * ProceduralNightSky: a procedural night/space background with three layers that can be combined: stars, nebula and
 * aurora curtains. A sky pass for scenes where PhysicalSky (daylight only) does not fit, and an env map generator.
 * Written from scratch for web-falcor (integer-hash value noise; no third-party shader code).
 *
 * The sky depends on the view direction only (it sits at infinity). Up axis +y. Per pixel view ray as in PhysicalSky.
 *   background : gradient zenithColour -> horizonColour by elevation, groundColour below the horizon (mode "replace").
 *   stars      : three jittered 3D-grid layers on the unit sphere (grid starGrid x 1, 2.03, 4.07; occupancy starDensity x
 *                1, 0.3, 0.08) (one candidate star per cell, kept when it lies in the
 *                shell around the sphere so the 27-cell neighbourhood always finds it); each star is a Gaussian spot of
 *                starSize pixels with a long-tailed brightness, a colour between blue-white and warm, and a sinusoidal
 *                twinkle driven by `time`.
 *   nebula     : domain-warped fBm value noise on the sphere: emission between nebulaColourA/B above a density
 *                threshold, plus dust lanes that also dim the stars behind them.
 *   aurora     : `auroraCurtains` curtains (azimuth 0 = towards -z, auroraDistance ahead, auroraSpacing apart), each sampled at `auroraLayers` altitudes (planes above the viewer): a band
 *                around a noise-displaced line, vertical rays from 1D noise along the curtain, a colour ramp from
 *                auroraColourLow (bottom) to auroraColourHigh (top), animated by `time` x auroraSpeed.
 * `upAxis` (default +y) sets the sky's up direction (the camera is rotated into the +y-up sky frame).
 * Layers fade towards the horizon (horizonFade) and are absent below it. `rotation` + rotationSpeed x time turns stars and
 * nebula about the up axis. Everything is scaled by skyScale.
 *
 * mode: "+"-separated subset of stars, nebula, aurora (e.g. "stars+aurora"); "none" leaves only the background.
 * composite: "replace" = background + layers behind the scene; "add" = the layers are added onto the `color` input
 *            (overlay on PhysicalSky's output or an existing background).
 *
 * Inputs : color? (scene/background colour), posW? (w = 0 marks background; geometry pixels pass `color` through)
 * Outputs: dst    (RGBA32Float) the sky behind the scene;
 *          layer  (RGBA32Float, optional) premultiplied sky layer for DepthComposite: replace -> (sky, 1), add -> (sky, 0);
 *                 zero where posW marks geometry;
 *          layerDepth (optional) r = kSkyDepth (3e38: behind every surface, in front of DepthComposite's empty background).
 * Stateless: `time` is a property; a frame depends only on the properties and the camera.
 * Identity: composite "add" with mode "none" (or skyScale 0) returns `color` bit for bit.
 * CPU reference: evalNightSky() (same formulas); createNightSkyEnvMap() bakes a lat-long EnvMap for lighting.
 */

import {
    ComputePass,
    EnvMap,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    RuntimeError,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture, parseColour } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Sky/ProceduralNightSky.cs.slang";
/** Layer depth of the sky (view-depth units): behind any surface, in front of an empty background (3.4e38). */
export const kSkyDepth = 3.0e38;
const kComposites = ["replace", "add"] as const;
const kLayers = ["stars", "nebula", "aurora"] as const;

export interface NightSkyParams {
    mode: string;
    composite: string;
    skyScale: number;
    zenithColour: number[];
    horizonColour: number[];
    groundColour: number[];
    horizonFade: number;
    /** World up axis of the sky (default +y; e.g. [0, 0, 1] for z-up scenes). */
    upAxis: number[];
    time: number;
    seed: number;
    rotation: number;
    rotationSpeed: number;
    starIntensity: number;
    starDensity: number;
    starGrid: number;
    starSize: number;
    starTwinkle: number;
    starColourVar: number;
    nebulaIntensity: number;
    nebulaScale: number;
    nebulaThreshold: number;
    nebulaWarp: number;
    nebulaDust: number;
    nebulaColourA: number[];
    nebulaColourB: number[];
    auroraIntensity: number;
    auroraCurtains: number;
    auroraLayers: number;
    auroraHeight: number;
    auroraThickness: number;
    auroraDistance: number;
    auroraSpacing: number;
    auroraAzimuth: number;
    auroraWave: number;
    auroraFreq: number;
    auroraRayFreq: number;
    auroraWidth: number;
    auroraExtent: number;
    auroraSpeed: number;
    auroraColourLow: number[];
    auroraColourHigh: number[];
}

export const kDefaultNightSky: NightSkyParams = {
    mode: "stars",
    composite: "replace",
    skyScale: 1,
    zenithColour: [0.002, 0.003, 0.009],
    horizonColour: [0.008, 0.011, 0.022],
    groundColour: [0.002, 0.002, 0.0025],
    upAxis: [0, 1, 0],
    horizonFade: 0.08,
    time: 0,
    seed: 0,
    rotation: 0,
    rotationSpeed: 0,
    starIntensity: 1,
    starDensity: 0.6,
    starGrid: 24,
    starSize: 1,
    starTwinkle: 0.25,
    starColourVar: 0.6,
    nebulaIntensity: 0.12,
    nebulaScale: 2.2,
    nebulaThreshold: 0.42,
    nebulaWarp: 1.4,
    nebulaDust: 0.6,
    nebulaColourA: [0.9, 0.25, 0.55],
    nebulaColourB: [0.15, 0.45, 1.0],
    auroraIntensity: 0.25,
    auroraCurtains: 2,
    auroraLayers: 24,
    auroraHeight: 1,
    auroraThickness: 0.7,
    auroraDistance: 3,
    auroraSpacing: 1.6,
    auroraAzimuth: 0,
    auroraWave: 1.2,
    auroraFreq: 0.3,
    auroraRayFreq: 5,
    auroraWidth: 0.3,
    auroraExtent: 14,
    auroraSpeed: 0.15,
    auroraColourLow: [0.1, 1.0, 0.35],
    auroraColourHigh: [0.55, 0.12, 0.75],
};

const kColourKeys = ["zenithColour", "horizonColour", "groundColour", "nebulaColourA", "nebulaColourB", "auroraColourLow", "auroraColourHigh"] as const;

/** Enabled layers of a mode string: bit 0 stars, 1 nebula, 2 aurora. Throws on unknown names. */
export function nightSkyLayerMask(mode: string): number {
    const m = String(mode).trim();
    if (m === "none" || m === "") return 0;
    let mask = 0;
    for (const part of m.split("+").map((s) => s.trim())) {
        const i = (kLayers as readonly string[]).indexOf(part);
        if (i < 0) throw new RuntimeError(`ProceduralNightSky: unknown mode '${part}' (use a '+'-list of ${kLayers.join(", ")}, or 'none')`);
        mask |= 1 << i;
    }
    return mask;
}

/** Parameters with defaults filled in, colours parsed (hex sRGB or linear triples) and ranges clamped. */
export function resolveNightSkyParams(p: Partial<NightSkyParams>): NightSkyParams {
    const r = { ...kDefaultNightSky, ...p } as NightSkyParams;
    for (const k of kColourKeys) r[k] = parseColour(r[k]);
    if (!(kComposites as readonly string[]).includes(r.composite)) throw new RuntimeError(`ProceduralNightSky: composite must be one of ${kComposites.join(", ")}`);
    nightSkyLayerMask(r.mode);
    r.seed = Math.max(0, Math.floor(Number(r.seed))) >>> 0;
    r.auroraCurtains = Math.min(Math.max(Math.round(r.auroraCurtains), 1), 4);
    r.auroraLayers = Math.min(Math.max(Math.round(r.auroraLayers), 2), 64);
    r.starGrid = Math.max(Number(r.starGrid), 1);
    r.horizonFade = Math.max(Number(r.horizonFade), 1e-4);
    r.upAxis = [0, 1, 2].map((i) => Number(r.upAxis?.[i] ?? 0));
    if (!r.upAxis.some((x) => x !== 0)) r.upAxis = [0, 1, 0];
    return r;
}

// ---- integer hash and value noise (bit-identical integer part on CPU and GPU) ----

/** pcg3d (Jarzynski & Olano 2020), 32-bit unsigned arithmetic. */
export function pcg3d(x: number, y: number, z: number): [number, number, number] {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    y = (Math.imul(y, 1664525) + 1013904223) >>> 0;
    z = (Math.imul(z, 1664525) + 1013904223) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0; y = (y + Math.imul(z, x)) >>> 0; z = (z + Math.imul(x, y)) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0; y = (y ^ (y >>> 16)) >>> 0; z = (z ^ (z >>> 16)) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0; y = (y + Math.imul(z, x)) >>> 0; z = (z + Math.imul(x, y)) >>> 0;
    return [x, y, z];
}
/** Hash of an integer lattice point with a salt -> 3 uints. */
const hashCell = (ix: number, iy: number, iz: number, salt: number) =>
    pcg3d((ix | 0) >>> 0, ((iy | 0) ^ salt) >>> 0, ((iz | 0) + Math.imul(salt, 0x9e37)) >>> 0);
const u01 = (u: number) => (u >>> 8) * (1 / 16777216);
const fract = (x: number) => x - Math.floor(x);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const smooth = (e0: number, e1: number, x: number) => { const t = sat((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };

/** 3D value noise in [0, 1) with smoothstep interpolation (C1). */
export function vnoise3(x: number, y: number, z: number, salt: number): number {
    const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
    const fx = x - ix, fy = y - iy, fz = z - iz;
    const wx = fx * fx * (3 - 2 * fx), wy = fy * fy * (3 - 2 * fy), wz = fz * fz * (3 - 2 * fz);
    const v = (a: number, b: number, c: number) => u01(hashCell(ix + a, iy + b, iz + c, salt)[0]);
    const x00 = mix(v(0, 0, 0), v(1, 0, 0), wx), x10 = mix(v(0, 1, 0), v(1, 1, 0), wx);
    const x01 = mix(v(0, 0, 1), v(1, 0, 1), wx), x11 = mix(v(0, 1, 1), v(1, 1, 1), wx);
    return mix(mix(x00, x10, wy), mix(x01, x11, wy), wz);
}
/** fBm of vnoise3, `oct` octaves, lacunarity 2.03, gain 0.5, normalised to [0, 1). */
export function fbm3(x: number, y: number, z: number, oct: number, salt: number): number {
    let s = 0, a = 0.5, n = 0;
    for (let o = 0; o < oct; o++) {
        s += a * vnoise3(x, y, z, (salt + o) >>> 0); n += a;
        x *= 2.03; y *= 2.03; z *= 2.03; a *= 0.5;
    }
    return s / n;
}

/** Host-side constants shared by the CPU reference and the shader. */
function derived(p: NightSkyParams) {
    const ang = ((p.rotation + p.rotationSpeed * p.time) * Math.PI) / 180;
    const az = (p.auroraAzimuth * Math.PI) / 180;
    return { rotCos: Math.cos(ang), rotSin: Math.sin(ang), azCos: Math.cos(az), azSin: Math.sin(az), mask: nightSkyLayerMask(p.mode), salt: Math.imul(p.seed, 131) >>> 0 };
}

const kStarGridScale = [1, 2.03, 4.07];
const kStarLayerDensity = [1, 0.3, 0.08];

/**
 * CPU reference of the shader: sky radiance towards `dir` (sky frame, y up; world directions go through toSkyFrame(upAxis)) for a pixel footprint of `pixelAngle` radians.
 * Returns the background and the layer sum separately (dst = replace ? bg + layers : color + layers).
 */
export function evalNightSky(dir: readonly number[], params: Partial<NightSkyParams> = {}, pixelAngle = 0.002): { bg: [number, number, number]; layers: [number, number, number] } {
    const p = resolveNightSkyParams(params);
    const D = derived(p);
    const l = Math.hypot(dir[0]!, dir[1]!, dir[2]!) || 1;
    const d = [dir[0]! / l, dir[1]! / l, dir[2]! / l];
    const S = p.skyScale;
    let bg: number[];
    if (d[1]! < 0) bg = p.groundColour.map((c) => c * S);
    else { const t = Math.sqrt(d[1]!); bg = [0, 1, 2].map((c) => mix(p.horizonColour[c]!, p.zenithColour[c]!, t) * S); }
    const L = [0, 0, 0];
    if (d[1]! <= 0 || D.mask === 0) return { bg: bg as [number, number, number], layers: [0, 0, 0] };
    const fade = smooth(0, p.horizonFade, d[1]!);
    // rotated direction for stars and nebula
    const r = [D.rotCos * d[0]! + D.rotSin * d[2]!, d[1]!, -D.rotSin * d[0]! + D.rotCos * d[2]!];
    let dust = 1;
    if (D.mask & 2) {
        const q = r.map((v) => v * p.nebulaScale);
        const wx = fbm3(q[0]! + 1.7, q[1]! + 9.2, q[2]! + 3.1, 4, (D.salt + 40) >>> 0);
        const wy = fbm3(q[0]! + 8.3, q[1]! + 2.8, q[2]! + 5.4, 4, (D.salt + 50) >>> 0);
        const wz = fbm3(q[0]! + 4.6, q[1]! + 7.7, q[2]! + 0.9, 4, (D.salt + 60) >>> 0);
        const k = p.nebulaWarp;
        const n = fbm3(q[0]! + k * (wx - 0.5), q[1]! + k * (wy - 0.5), q[2]! + k * (wz - 0.5), 5, (D.salt + 70) >>> 0);
        const dens = Math.pow(sat((n - p.nebulaThreshold) / Math.max(1 - p.nebulaThreshold, 1e-3)), 1.5);
        const hue = fbm3(2 * q[0]! + 5.3, 2 * q[1]! + 1.1, 2 * q[2]! + 7.9, 3, (D.salt + 80) >>> 0);
        const lanes = sat(2.5 * fbm3(3 * q[0]! + 11, 3 * q[1]! + 4, 3 * q[2]! + 2, 4, (D.salt + 90) >>> 0) - 1.1);
        dust = 1 - p.nebulaDust * lanes;
        const e = p.nebulaIntensity * dens * dust;
        for (let c = 0; c < 3; c++) L[c]! += e * mix(p.nebulaColourA[c]!, p.nebulaColourB[c]!, smooth(0.3, 0.7, hue));
    }
    if (D.mask & 1) {
        const rad = Math.max(p.starSize, 0.05) * pixelAngle;
        for (let layer = 0; layer < 3; layer++) {
            const N = p.starGrid * kStarGridScale[layer]!;
            const P = r.map((v) => v * N);
            const c0 = P.map(Math.floor);
            const salt = (D.salt + 10 + layer) >>> 0;
            for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
                const cx = c0[0]! + dx, cy = c0[1]! + dy, cz = c0[2]! + dz;
                const h = hashCell(cx, cy, cz, salt);
                const h2 = pcg3d(h[0], h[1] ^ 0x5bd1e995, h[2]);
                if (u01(h2[0]) >= p.starDensity * kStarLayerDensity[layer]!) continue;
                const q = [cx + 0.1 + 0.8 * u01(h[0]), cy + 0.1 + 0.8 * u01(h[1]), cz + 0.1 + 0.8 * u01(h[2])];
                const ql = Math.hypot(q[0]!, q[1]!, q[2]!);
                if (Math.abs(ql - N) >= 0.5) continue;
                const s = q.map((v) => v / ql);
                const ex = r[0]! - s[0]!, ey = r[1]! - s[1]!, ez = r[2]! - s[2]!;
                const a2 = (ex * ex + ey * ey + ez * ez) / (rad * rad);
                if (a2 > 9) continue;
                const ub = u01(h2[1]);
                const bright = 0.04 + 3 * Math.pow(ub, 8);
                const tw = 1 + p.starTwinkle * Math.sin(p.time * (2 + 4 * u01(h2[2])) + 6.2831853 * fract(ub * 7.13));
                const temp = fract(ub * 13.7 + u01(h2[2]) * 3.1);
                const col = [0, 1, 2].map((c) => mix(1, mix([0.72, 0.82, 1][c]!, [1, 0.78, 0.55][c]!, temp), p.starColourVar));
                const e = p.starIntensity * bright * tw * Math.exp(-2 * a2) * dust;
                for (let c = 0; c < 3; c++) L[c]! += e * col[c]!;
            }
        }
    }
    if (D.mask & 4) {
        const K = p.auroraLayers;
        const acc = [0, 0, 0];
        for (let cu = 0; cu < p.auroraCurtains; cu++) {
            for (let i = 0; i < K; i++) {
                const t = i / (K - 1);
                const h = p.auroraHeight * (1 + p.auroraThickness * t);
                const px = (d[0]! * h) / d[1]!, pz = (d[2]! * h) / d[1]!;
                const a = D.azCos * px + D.azSin * pz, b = D.azSin * px - D.azCos * pz;
                const ts = p.time * p.auroraSpeed;
                const line = p.auroraDistance + cu * p.auroraSpacing + p.auroraWave * (2 * vnoise3(a * p.auroraFreq, ts, cu * 7.3 + 0.5, (D.salt + 20) >>> 0) - 1);
                const db = (b - line) / p.auroraWidth;
                const band = Math.exp(-db * db);
                const rays = 0.35 + 0.65 * vnoise3(a * p.auroraRayFreq, 2 * ts, cu * 3.1 + 11.5, (D.salt + 30) >>> 0);
                const w = band * rays * Math.exp(-3.5 * t) * Math.exp(-Math.hypot(px, pz) / p.auroraExtent);
                const ct = smooth(0.1, 0.9, t);
                for (let c = 0; c < 3; c++) acc[c]! += w * mix(p.auroraColourLow[c]!, p.auroraColourHigh[c]!, ct);
            }
        }
        for (let c = 0; c < 3; c++) L[c]! += (acc[c]! * p.auroraIntensity * 8) / K;
    }
    return { bg: bg as [number, number, number], layers: L.map((v) => v * fade * S) as [number, number, number] };
}

/** Lat-long EnvMap of the night sky in world space (upAxis applied; Falcor's world_to_latlong_map convention; height = width / 2). */
export function createNightSkyEnvMap(device: Device, params: Partial<NightSkyParams> = {}, width = 1024): EnvMap {
    const height = Math.max(1, width >> 1);
    const data = new Float32Array(width * height * 4);
    const pixelAngle = (2 * Math.PI) / width;
    const toSky = toSkyFrame(resolveNightSkyParams(params).upAxis);
    for (let j = 0; j < height; j++) {
        const theta = (Math.PI * (j + 0.5)) / height;
        for (let i = 0; i < width; i++) {
            const phi = Math.PI * (2 * ((i + 0.5) / width) - 1);
            const dir = [Math.sin(theta) * Math.sin(phi), Math.cos(theta), -Math.sin(theta) * Math.cos(phi)];
            const s = evalNightSky(toSky(dir), { ...params, composite: "replace" }, pixelAngle);
            const o = (j * width + i) * 4;
            data[o] = s.bg[0] + s.layers[0]; data[o + 1] = s.bg[1] + s.layers[1]; data[o + 2] = s.bg[2] + s.layers[2]; data[o + 3] = 1;
        }
    }
    return new EnvMap(device, { width, height, data } as ConstructorParameters<typeof EnvMap>[1]);
}


/** Rotation taking `up` to +y (minimal rotation, Rodrigues); identity for +y. Sky formulas use +y up. */
export function toSkyFrame(up: readonly number[]): (v: readonly number[]) => [number, number, number] {
    const l = Math.hypot(up[0]!, up[1]!, up[2]!) || 1;
    const u = [up[0]! / l, up[1]! / l, up[2]! / l];
    const k = [-u[2]!, 0, u[0]!]; // u x (0, 1, 0)
    const s = Math.hypot(k[0]!, k[2]!), c = u[1]!;
    if (s < 1e-9) return c > 0 ? (v) => [v[0]!, v[1]!, v[2]!] : (v) => [v[0]!, -v[1]!, -v[2]!];
    const kx = k[0]! / s, kz = k[2]! / s;
    return (v) => {
        const kv = kx * v[0]! + kz * v[2]!;
        const cx = -kz * v[1]!, cy = kz * v[0]! - kx * v[2]!, cz = kx * v[1]!; // k x v (k = (kx, 0, kz))
        return [v[0]! * c + cx * s + kx * kv * (1 - c), v[1]! * c + cy * s, v[2]! * c + cz * s + kz * kv * (1 - c)];
    };
}

const norm3 = (v: number[]): [number, number, number] => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
    return [v[0]! / l, v[1]! / l, v[2]! / l];
};

export class ProceduralNightSky extends RenderPass {
    private p: NightSkyParams = { ...kDefaultNightSky };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.p) as (keyof NightSkyParams)[]) (this.p as unknown as Record<string, unknown>)[k] = props.get(k, this.p[k]);
        resolveNightSkyParams(this.p);
    }

    get params(): NightSkyParams {
        return { ...this.p };
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(this.p) as (keyof NightSkyParams)[]) (this.p as unknown as Record<string, unknown>)[k] = props.get(k, this.p[k]);
        resolveNightSkyParams(this.p);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Time", this.p.time, 0, 600, 0.01, (v) => (this.p.time = v));
        ui.slider("Sky scale", this.p.skyScale, 0, 10, 0.001, (v) => (this.p.skyScale = v));
        ui.slider("Star intensity", this.p.starIntensity, 0, 10, 0.01, (v) => (this.p.starIntensity = v));
        ui.slider("Nebula intensity", this.p.nebulaIntensity, 0, 2, 0.001, (v) => (this.p.nebulaIntensity = v));
        ui.slider("Aurora intensity", this.p.auroraIntensity, 0, 2, 0.001, (v) => (this.p.auroraIntensity = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "color", "Scene / background colour", true);
        addTexInput(r, "posW", "World position, w = 0 marks background", true);
        addColorOutput(r, "dst", "Sky behind the scene", w, h);
        addColorOutput(r, "layer", "Premultiplied sky layer for DepthComposite", w, h, true);
        addColorOutput(r, "layerDepth", "Layer depth (r = 3e38)", w, h, true);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        if (!dst) return;
        const color = rd.getTexture("color"), posW = rd.getTexture("posW");
        const layer = rd.getTexture("layer"), layerDepth = rd.getTexture("layerDepth");
        const p = resolveNightSkyParams(this.p);
        const D = derived(p);
        const W = dst.width, H = dst.height;
        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        let fwd = [0, 0, -1], up = [0, 1, 0], tanY = Math.tan(Math.PI / 8), aspect = W / H;
        if (this.scene) {
            const cam = this.scene.camera;
            const cp = cam.getPosition(), ct = cam.getTarget(), cu = cam.getUpVector();
            fwd = norm3([ct.x - cp.x, ct.y - cp.y, ct.z - cp.z]);
            up = [cu.x, cu.y, cu.z];
            tanY = Math.tan(0.5 * cam.getFovY());
            aspect = cam.getAspectRatio();
        }
        const right = norm3([fwd[1]! * up[2]! - fwd[2]! * up[1]!, fwd[2]! * up[0]! - fwd[0]! * up[2]!, fwd[0]! * up[1]! - fwd[1]! * up[0]!]);
        const trueUp = [right[1] * fwd[2]! - right[2] * fwd[1]!, right[2] * fwd[0]! - right[0] * fwd[2]!, right[0] * fwd[1]! - right[1] * fwd[0]!];
        const toSky = toSkyFrame(p.upAxis);
        const [rs, us, fs] = [toSky(right), toSky(trueUp), toSky(fwd)];
        const S = p.skyScale;
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        cb["gFrameDim"] = [W, H];
        cb["gHasColor"] = color ? 1 : 0;
        cb["gHasPos"] = posW ? 1 : 0;
        cb["gMask"] = D.mask;
        cb["gReplace"] = p.composite === "replace" ? 1 : 0;
        cb["gSalt"] = D.salt;
        cb["gCurtains"] = p.auroraCurtains;
        cb["gAuroraLayers"] = p.auroraLayers;
        cb["gCamRight"] = [...rs.map((v) => v * tanY * aspect), (2 * tanY) / H];
        cb["gCamUp"] = [...us.map((v) => v * tanY), S];
        cb["gCamFwd"] = [...fs, p.horizonFade];
        cb["gZenith"] = [...p.zenithColour, p.time];
        cb["gHorizon"] = [...p.horizonColour, D.rotCos];
        cb["gGround"] = [...p.groundColour, D.rotSin];
        cb["gStar"] = [p.starIntensity, p.starDensity, p.starGrid, Math.max(p.starSize, 0.05)];
        cb["gStar2"] = [p.starTwinkle, p.starColourVar, 0, 0];
        cb["gNeb"] = [p.nebulaIntensity, p.nebulaScale, p.nebulaThreshold, p.nebulaWarp];
        cb["gNebA"] = [...p.nebulaColourA, p.nebulaDust];
        cb["gNebB"] = [...p.nebulaColourB, 0];
        cb["gAur"] = [p.auroraIntensity, p.auroraHeight, p.auroraThickness, p.auroraDistance];
        cb["gAur2"] = [p.auroraSpacing, p.auroraWave, p.auroraFreq, p.auroraRayFreq];
        cb["gAur3"] = [p.auroraWidth, p.auroraExtent, p.time * p.auroraSpeed, 0];
        cb["gAurAz"] = [D.azCos, D.azSin, 0, 0];
        cb["gAurLow"] = [...p.auroraColourLow, 0];
        cb["gAurHigh"] = [...p.auroraColourHigh, 0];
        root["gColor"] = color ?? dummyTexture(this.device);
        root["gPosW"] = posW ?? dummyTexture(this.device);
        root["gDst"] = dst;
        cb["gHasLayer"] = layer ? 1 : 0;
        cb["gHasLayerDepth"] = layerDepth ? 1 : 0;
        root["gLayer"] = layer ?? dummyTexture(this.device, "out:layer");
        root["gLayerDepth"] = layerDepth ?? dummyTexture(this.device, "out:layerDepth");
        this.pass.execute(ctx, W, H);
    }
}

registerRenderPass("ProceduralNightSky", (device, props) => new ProceduralNightSky(device, props), {
    desc: "Procedural night sky: stars, nebula and aurora curtains (background, DepthComposite layer, env map)",
});
