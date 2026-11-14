/**
 * VolumetricClouds: raymarched volumetric cloud layer in an altitude slab (curved planet shell or flat), with a weather
 * map for coverage and cloud type, wind, a light march towards the sun, dual-lobe Henyey-Greenstein phase with a powder
 * term, height-blended sky ambient, optional low-resolution marching and temporal reprojection. One render-graph node,
 * two dispatches per frame:
 *   march   (W/f x H/f, f = 1, 2, 4 by `resolution`): per ray, `steps` jittered samples through the slab interval,
 *           clamped to maxDistance and to the farthest scene surface of the footprint; `lightSteps` samples towards
 *           the light per cloud sample; energy-conserving integration (S, T) and the front distance.
 *   resolve (W x H): bilinear upsample, removes clouds in front of nearer scene surfaces, temporal reprojection of the
 *           cloud front point (camera + wind compensated) with a neighbourhood clamp, outputs.
 *
 * Weather (the "sky weather" knobs): coverage [0, 1], cloudType (0 stratus, 0.5 stratocumulus, 1 cumulus: vertical
 * profile), a 2D weather map (value-noise fBm over the horizontal plane, weatherScale) that varies coverage by
 * weatherAmount and type by typeVariation, density (extinction), wind (world units per time unit) x time.
 * Geometry: cloudBottom / cloudTop are altitudes above groundHeight along upAxis; planetRadius > 0 curves the slab
 * (rays reach the horizon at a finite distance), 0 = flat slab. Units are the scene's (defaults: metres).
 * Light: lightDir ([0, 0, 0] = the scene's first directional light, else up), lightColor x lightIntensity; ambient =
 * lerp(ambientBottom, ambientTop, normalised height) (the sky irradiance; set it from the sky in use).
 * fadeDistance > 0 fades clouds with the distance of their front (aerial perspective towards the background sky);
 * opacity scales S and 1 - T the same way.
 *
 * Outputs: dst = color * T + S where clouds are visible (scene surfaces nearer than the cloud front keep `color`);
 *          layer = premultiplied (S, 1 - T) and transmittance = (T, T, T, 1) for DepthComposite; layerDepth (r) =
 *          linear view depth of the cloud front (2.9e38 where there is none).
 * Inputs : color (optional; black without), posW (optional; GBufferRaster.posW, w = 0 on background: clamps the march to
 *          the scene surface; without it the clouds cover the whole frame).
 *
 * History (temporal = true): resolved (S, T) and (front, valid) in two ping-pong pairs (Utils/History.ts). Reset on the
 * first execute, resetHistory() / resetGraphHistory(), scene or camera change, a change of size, resolution or
 * temporal, Camera.resetHistory() and a primed camera cut; on a reset both pairs are cleared and the per-frame jitter
 * sequence restarts (it is indexed by frames since the reset), so a shot after a reset is bit-identical to the same
 * shot on a fresh context. temporal = false keeps no state that affects the output (static per-pixel jitter).
 * Identity: coverage 0, density 0 or opacity 0 copy `color` to dst bit for bit (layer 0, transmittance 1) and reset the
 * history.
 *
 * CPU reference: marchCloudRay() (march kernel for one ray) with cloudFrameConstants() (the constants both use).
 *
 * Adapted in part from bevy-volumetric-clouds (https://github.com/evroon/bevy-volumetric-clouds, MIT License,
 * Copyright (c) 2025 evroon; notice in VolumetricClouds.cs.slang): slab/planet-shell intersection, dual-lobe phase,
 * growing light step, energy-conserving integration and the default values of those parameters.
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type PropertyValue,
    type RenderContext,
    type Scene,
    type Texture,
    type UIWidgets,
} from "@web-falcor/falcor";
import { HistoryTracker, PingPongTextures } from "../Utils/History.js";
import { addColorOutput, addTexInput, dummyTexture, keyLightDir, parseColour } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/VolumetricClouds/VolumetricClouds.cs.slang";
const kResolutions = ["full", "half", "quarter"] as const;
/** Layer depth where a pixel has no cloud (behind any surface). */
export const kCloudNoDepth = 2.9e38;

export interface VolumetricCloudsParams {
    coverage: number;
    cloudType: number;
    density: number;
    weatherScale: number;
    weatherAmount: number;
    typeVariation: number;
    shapeScale: number;
    octaves: number;
    detailScale: number;
    detailStrength: number;
    detailOctaves: number;
    cloudBottom: number;
    cloudTop: number;
    groundHeight: number;
    planetRadius: number;
    maxDistance: number;
    upAxis: number[];
    wind: number[];
    time: number;
    seed: number;
    lightDir: number[];
    lightColor: number[];
    lightIntensity: number;
    ambientTop: number[];
    ambientBottom: number[];
    albedo: number;
    phaseForward: number;
    phaseBackward: number;
    phaseMix: number;
    powder: number;
    steps: number;
    lightSteps: number;
    lightStep: number;
    lightStepGrowth: number;
    minTransmittance: number;
    jitter: number;
    fadeDistance: number;
    opacity: number;
    resolution: string;
    temporal: boolean;
    temporalBlend: number;
}

export const kDefaultVolumetricClouds: VolumetricCloudsParams = {
    coverage: 0.5,
    cloudType: 0.6,
    density: 0.03,
    weatherScale: 1 / 12000,
    weatherAmount: 0.6,
    typeVariation: 0.3,
    shapeScale: 1 / 1800,
    octaves: 4,
    detailScale: 1 / 300,
    detailStrength: 0.35,
    detailOctaves: 2,
    cloudBottom: 1250,
    cloudTop: 2400,
    groundHeight: 0,
    planetRadius: 6371000,
    maxDistance: 40000,
    upAxis: [0, 1, 0],
    wind: [8, 0, 3],
    time: 0,
    seed: 0,
    lightDir: [0, 0, 0],
    lightColor: [1, 0.9, 0.85],
    lightIntensity: 1.4,
    ambientTop: [0.99, 1.11, 1.33],
    ambientBottom: [0.26, 0.45, 0.58],
    albedo: 1,
    phaseForward: 0.8,
    phaseBackward: -0.2,
    phaseMix: 0.5,
    powder: 0.5,
    steps: 48,
    lightSteps: 6,
    lightStep: 40,
    lightStepGrowth: 1.5,
    minTransmittance: 0.01,
    jitter: 1,
    fadeDistance: 0,
    opacity: 1,
    resolution: "full",
    temporal: true,
    temporalBlend: 0.9,
};

type V3 = [number, number, number];
const norm3 = (v: readonly number[]): V3 => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
    return [v[0]! / l, v[1]! / l, v[2]! / l];
};
const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
const v3 = (a: unknown, d: readonly number[]): V3 => {
    if (typeof a === "number") return [a, a, a];
    const x = Array.isArray(a) ? a.map(Number) : d;
    return [x[0] ?? d[0]!, x[1] ?? d[1]!, x[2] ?? d[2]!];
};

/** Parameters with defaults filled in, colours parsed and ranges clamped. */
export function resolveVolumetricCloudsParams(p: Partial<VolumetricCloudsParams>): VolumetricCloudsParams {
    const r = { ...kDefaultVolumetricClouds, ...p } as VolumetricCloudsParams;
    const cl = (x: unknown, lo: number, hi: number) => Math.min(Math.max(Number(x), lo), hi);
    r.coverage = cl(r.coverage, 0, 1);
    r.cloudType = cl(r.cloudType, 0, 1);
    r.density = Math.max(0, Number(r.density));
    r.opacity = cl(r.opacity, 0, 1);
    r.weatherAmount = Math.max(0, Number(r.weatherAmount));
    r.typeVariation = Math.max(0, Number(r.typeVariation));
    r.detailStrength = Math.max(0, Number(r.detailStrength));
    r.octaves = Math.round(cl(r.octaves, 1, 8));
    r.detailOctaves = Math.round(cl(r.detailOctaves, 1, 4));
    r.cloudBottom = Number(r.cloudBottom);
    r.cloudTop = Math.max(Number(r.cloudTop), r.cloudBottom + 1e-3);
    r.planetRadius = Math.max(0, Number(r.planetRadius));
    r.maxDistance = Math.max(1e-3, Number(r.maxDistance));
    r.upAxis = v3(r.upAxis, [0, 1, 0]);
    if (!r.upAxis.some((x) => x !== 0)) r.upAxis = [0, 1, 0];
    r.wind = v3(r.wind, [0, 0, 0]);
    r.lightColor = parseColour(r.lightColor);
    r.ambientTop = parseColour(r.ambientTop);
    r.ambientBottom = parseColour(r.ambientBottom);
    r.phaseForward = cl(r.phaseForward, -0.99, 0.99);
    r.phaseBackward = cl(r.phaseBackward, -0.99, 0.99);
    r.phaseMix = cl(r.phaseMix, 0, 1);
    r.powder = cl(r.powder, 0, 1);
    r.steps = Math.round(cl(r.steps, 1, 512));
    r.lightSteps = Math.round(cl(r.lightSteps, 0, 32));
    r.lightStep = Math.max(1e-6, Number(r.lightStep));
    r.lightStepGrowth = Math.max(1, Number(r.lightStepGrowth));
    r.minTransmittance = cl(r.minTransmittance, 0, 1);
    r.jitter = cl(r.jitter, 0, 1);
    r.fadeDistance = Math.max(0, Number(r.fadeDistance));
    r.seed = Math.max(0, Math.floor(Number(r.seed))) >>> 0;
    r.temporalBlend = cl(r.temporalBlend, 0, 0.99);
    if (!(kResolutions as readonly string[]).includes(r.resolution)) throw new Error(`VolumetricClouds: resolution must be one of ${kResolutions.join("|")} (got ${r.resolution})`);
    r.temporal = !!r.temporal;
    return r;
}

/** Camera state as used by the kernels (Falcor camera data: pixel ray = ndc.x * U + ndc.y * V + W). */
export interface CloudCamera { pos: V3; U: V3; V: V3; W: V3 }

export function cloudCamera(scene: Scene): CloudCamera {
    const d = scene.camera.getData();
    return {
        pos: [d.posW.x, d.posW.y, d.posW.z], U: [d.cameraU.x, d.cameraU.y, d.cameraU.z],
        V: [d.cameraV.x, d.cameraV.y, d.cameraV.z], W: [d.cameraW.x, d.cameraW.y, d.cameraW.z],
    };
}

/** Per-frame constants shared by the kernels and the CPU reference. */
export interface CloudFrame {
    p: VolumetricCloudsParams;
    cam: CloudCamera;
    up: V3; e1: V3; e2: V3;
    altitude: number;
    light: V3;
    lightRgb: V3;
    offset: V3;
    salt: number;
}

export function cloudFrameConstants(params: Partial<VolumetricCloudsParams>, cam: CloudCamera, light: readonly number[]): CloudFrame {
    const p = resolveVolumetricCloudsParams(params);
    const up = norm3(p.upAxis);
    const ref: V3 = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
    const e2 = norm3(cross(ref, up)), e1 = cross(up, e2);
    return {
        p, cam, up, e1, e2,
        altitude: dot(cam.pos, up) - p.groundHeight,
        light: norm3(light),
        lightRgb: p.lightColor.map((c) => c * p.lightIntensity) as V3,
        offset: p.wind.map((w) => w * p.time) as V3,
        salt: Math.imul(p.seed, 0x2545f491) >>> 0,
    };
}

// ---- CPU reference of the kernels (integer hash bit-identical to the shader) ----

function pcg3d(x: number, y: number, z: number): [number, number, number] {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    y = (Math.imul(y, 1664525) + 1013904223) >>> 0;
    z = (Math.imul(z, 1664525) + 1013904223) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0; y = (y + Math.imul(z, x)) >>> 0; z = (z + Math.imul(x, y)) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0; y = (y ^ (y >>> 16)) >>> 0; z = (z ^ (z >>> 16)) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0; y = (y + Math.imul(z, x)) >>> 0; z = (z + Math.imul(x, y)) >>> 0;
    return [x, y, z];
}
const u01 = (u: number) => (u >>> 8) * (1 / 16777216);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const sstep = (e0: number, e1: number, x: number) => { const t = sat((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };

function vnoise3(x: number, y: number, z: number, salt: number): number {
    const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
    const fx = x - ix, fy = y - iy, fz = z - iz;
    const wx = fx * fx * (3 - 2 * fx), wy = fy * fy * (3 - 2 * fy), wz = fz * fz * (3 - 2 * fz);
    const h = (a: number, b: number, c: number) => u01(pcg3d((ix + a) >>> 0, (iy + b) >>> 0, (((iz + c) | 0) ^ salt) >>> 0)[0]);
    const x00 = mix(h(0, 0, 0), h(1, 0, 0), wx), x10 = mix(h(0, 1, 0), h(1, 1, 0), wx);
    const x01 = mix(h(0, 0, 1), h(1, 0, 1), wx), x11 = mix(h(0, 1, 1), h(1, 1, 1), wx);
    return mix(mix(x00, x10, wy), mix(x01, x11, wy), wz);
}

function fbm3(x: number, y: number, z: number, oct: number, salt: number): number {
    let s = 0, a = 0.5, n = 0;
    for (let o = 0; o < oct; o++) {
        s += a * vnoise3(x, y, z, (salt + Math.imul(o, 7919)) >>> 0);
        n += a;
        x = x * 2.03 + 17.1; y = y * 2.03 + 5.3; z = z * 2.03 + 11.7;
        a *= 0.5;
    }
    return s / n;
}

function heightProfile(h: number, t: number): number {
    const st = sstep(0, 0.07, h) * (1 - sstep(0.12, 0.28, h));
    const sc = sstep(0, 0.1, h) * (1 - sstep(0.3, 0.6, h));
    const cu = sstep(0, 0.12, h) * (1 - sstep(0.55, 0.98, h));
    return t < 0.5 ? mix(st, sc, 2 * t) : mix(sc, cu, 2 * t - 1);
}

function altitudeOf(f: CloudFrame, r: readonly number[]): number {
    const a = f.altitude, R = f.p.planetRadius, ru = dot(r, f.up);
    if (R <= 0) return a + ru;
    const Ra = R + a, k = 2 * Ra * ru + dot(r, r);
    return a + k / (Math.sqrt(Math.max(Ra * Ra + k, 0)) + Ra);
}

/** Cloud density in [0, 1] at camera + r (altitude alt). */
export function cloudDensity(f: CloudFrame, r: readonly number[], alt: number): number {
    const p = f.p;
    const h = (alt - p.cloudBottom) / (p.cloudTop - p.cloudBottom);
    if (h < 0 || h > 1) return 0;
    const q = [0, 1, 2].map((k) => f.cam.pos[k]! + r[k]! - f.offset[k]!);
    const wu = dot(q, f.e1) * p.weatherScale, wv = dot(q, f.e2) * p.weatherScale;
    const wc = fbm3(wu, wv, 0.5, 3, (f.salt + 101) >>> 0);
    const cov = sat(p.coverage * (1 + p.weatherAmount * (2 * wc - 1)));
    if (cov <= 0) return 0;
    const wt = fbm3(wu, wv, 3.5, 2, (f.salt + 202) >>> 0);
    const prof = heightProfile(h, sat(p.cloudType + p.typeVariation * (2 * wt - 1)));
    if (prof <= 0) return 0;
    const base = fbm3(q[0]! * p.shapeScale, q[1]! * p.shapeScale, q[2]! * p.shapeScale, p.octaves, f.salt);
    let d = sat((base * prof - (1 - cov)) / Math.max(cov, 1e-3));
    if (d > 0 && p.detailStrength > 0) {
        const det = fbm3(q[0]! * p.detailScale, q[1]! * p.detailScale, q[2]! * p.detailScale, p.detailOctaves, (f.salt + 303) >>> 0);
        d = sat(d - p.detailStrength * det * (1 - d));
    }
    return d;
}

const hg = (c: number, g: number) => { const den = 1 + g * g - 2 * g * c; return (0.0795774715459 * (1 - g * g)) / (den * Math.sqrt(den)); };

/** Slab interval [t0, t1] along unit dir (t1 <= t0: none). */
export function cloudSlabInterval(f: CloudFrame, dir: readonly number[]): [number, number] {
    const a = f.altitude, R = f.p.planetRadius, bot = f.p.cloudBottom, top = f.p.cloudTop, mu = dot(dir, f.up);
    if (R <= 0) {
        if (a < bot) return mu > 0 ? [(bot - a) / mu, (top - a) / mu] : [0, -1];
        if (a > top) return mu < 0 ? [(top - a) / mu, (bot - a) / mu] : [0, -1];
        return [0, mu > 0 ? (top - a) / mu : mu < 0 ? (bot - a) / mu : 3e38];
    }
    const Ra = R + a, b = Ra * mu;
    const db = b * b + (bot - a) * (2 * R + bot + a), dt = b * b + (top - a) * (2 * R + top + a);
    if (a < bot) return [-b + Math.sqrt(Math.max(db, 0)), -b + Math.sqrt(Math.max(dt, 0))];
    if (a > top) return mu < 0 && dt >= 0 ? [-b - Math.sqrt(dt), db >= 0 ? -b - Math.sqrt(db) : -b + Math.sqrt(dt)] : [0, -1];
    return [0, mu < 0 && db >= 0 ? -b - Math.sqrt(db) : -b + Math.sqrt(Math.max(dt, 0))];
}

/** Per-pixel jitter of the march kernel (pixel of the march grid; frame = frames since the history reset, 0 without temporal). */
export function cloudJitter(f: CloudFrame, px: number, py: number, frame: number): number {
    if (f.p.jitter <= 0) return 0.5;
    const h = u01(pcg3d(px >>> 0, py >>> 0, (Math.imul(frame, 0x9e3779b9) + f.salt) >>> 0)[0]);
    return 0.5 + f.p.jitter * (h - 0.5);
}

/**
 * CPU reference of the march kernel for one ray: unit direction dir, march limited to maxT (scene surface), jitter jit.
 * Returns the in-scattered radiance S, transmittance T and the front distance along the ray (-1 = no cloud).
 */
export function marchCloudRay(f: CloudFrame, dir: readonly number[], maxT = 3e38, jit = 0.5): { S: V3; T: number; front: number } {
    const p = f.p;
    const [t0, t1r] = cloudSlabInterval(f, dir);
    const t1 = Math.min(t1r, Math.min(maxT, p.maxDistance));
    let S: V3 = [0, 0, 0], T = 1, front = -1;
    if (!(t1 > t0)) return { S, T, front };
    const n = Math.max(p.steps, 1), dt = (t1 - t0) / n;
    const c = dot(dir, f.light);
    const phase = mix(hg(c, p.phaseForward), hg(c, p.phaseBackward), p.phaseMix);
    for (let i = 0; i < n; i++) {
        const t = t0 + (i + jit) * dt;
        const r = [dir[0]! * t, dir[1]! * t, dir[2]! * t];
        const alt = altitudeOf(f, r);
        const d = cloudDensity(f, r, alt);
        if (d > 0) {
            if (front < 0) front = t;
            const sigma = d * p.density;
            const h = sat((alt - p.cloudBottom) / (p.cloudTop - p.cloudBottom));
            let s = 0, ds = p.lightStep, od = 0;
            for (let k = 0; k < p.lightSteps; k++) {
                const m = s + 0.5 * ds;
                const rl = [r[0]! + f.light[0] * m, r[1]! + f.light[1] * m, r[2]! + f.light[2] * m];
                const al = altitudeOf(f, rl);
                if (al > p.cloudTop) break;
                od += cloudDensity(f, rl, al) * ds;
                s += ds;
                ds *= p.lightStepGrowth;
            }
            const Tl = Math.exp(-p.density * od);
            const powder = mix(1, 1 - Math.exp(-2 * sigma * p.lightStep), p.powder);
            const dT = Math.exp(-sigma * dt);
            S = [0, 1, 2].map((k) => S[k]! + T * p.albedo * (f.lightRgb[k]! * phase * (Tl * powder) + mix(p.ambientBottom[k]!, p.ambientTop[k]!, h)) * (1 - dT)) as V3;
            T *= dT;
            if (T < p.minTransmittance) break;
        }
    }
    if (front >= 0 && (p.fadeDistance > 0 || p.opacity < 1)) {
        const fd = (p.fadeDistance > 0 ? Math.exp(-front / p.fadeDistance) : 1) * p.opacity;
        S = S.map((x) => x * fd) as V3;
        T = 1 - (1 - T) * fd;
    }
    return { S, T, front };
}

/** Unit ray direction of pixel (x, y) of a w x h grid (pixel centres), as the kernels compute it. */
export function cloudPixelRay(cam: CloudCamera, x: number, y: number, w: number, h: number): V3 {
    const nx = (2 * (x + 0.5)) / w - 1, ny = 1 - (2 * (y + 0.5)) / h;
    return norm3([0, 1, 2].map((k) => nx * cam.U[k]! + ny * cam.V[k]! + cam.W[k]!));
}

export class VolumetricClouds extends RenderPass {
    private p: VolumetricCloudsParams = { ...kDefaultVolumetricClouds };
    private passes: Record<string, ComputePass> = {};
    private readonly history = new HistoryTracker();
    private histA: PingPongTextures;
    private histB: PingPongTextures;
    private march: [Texture, Texture] | null = null;
    private prevCam: CloudCamera | null = null;
    private prevTime = 0;

    constructor(device: Device, props: Properties) {
        super(device);
        this.histA = new PingPongTextures(device, ResourceFormat.RGBA32Float, "VolumetricClouds::historyA");
        this.histB = new PingPongTextures(device, ResourceFormat.RGBA32Float, "VolumetricClouds::historyB");
        this.setProperties(props);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p } as unknown as Record<string, PropertyValue>);
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(this.p) as (keyof VolumetricCloudsParams)[]) (this.p as unknown as Record<string, unknown>)[k] = props.get(k, this.p[k] as PropertyValue);
        resolveVolumetricCloudsParams(this.p);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Coverage", this.p.coverage, 0, 1, 0.01, (v) => (this.p.coverage = v));
        ui.slider("Cloud type", this.p.cloudType, 0, 1, 0.01, (v) => (this.p.cloudType = v));
        ui.slider("Density", this.p.density, 0, 0.2, 0.001, (v) => (this.p.density = v));
        ui.slider("Time", this.p.time, 0, 3600, 0.1, (v) => (this.p.time = v));
        ui.slider("Steps", this.p.steps, 1, 256, 1, (v) => (this.p.steps = Math.round(v)));
        ui.dropdown("Resolution", [...kResolutions], this.p.resolution, (v) => (this.p.resolution = v));
        ui.checkbox("Temporal", this.p.temporal, (v) => (this.p.temporal = v));
    }

    /** Drops the temporal history: the next frame is computed as on a fresh context. */
    resetHistory(): void {
        this.history.reset();
    }

    /** Frames since the last history reset. */
    get frameIndex(): number {
        return this.history.frameIndex;
    }

    /** The light direction used for shading (lightDir, else the scene's key light, else up). */
    get lightDirection(): V3 {
        return keyLightDir(this.scene, this.p.lightDir);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "color", "Scene / sky colour (HDR)", true);
        addTexInput(r, "posW", "World position (GBufferRaster.posW; w = 0 on background)", true);
        addColorOutput(r, "dst", "Colour with clouds", w, h);
        addColorOutput(r, "layer", "Premultiplied cloud layer (S, 1 - T) for DepthComposite", w, h, true);
        addColorOutput(r, "layerDepth", "Cloud front view depth (r; 2.9e38 = none)", w, h, true);
        addColorOutput(r, "transmittance", "Cloud transmittance (T, T, T, 1)", w, h, true);
        return r;
    }

    private pass(entry: string): ComputePass {
        return (this.passes[entry] ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: entry }));
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        if (!dst) return;
        const color = rd.getTexture("color"), posW = rd.getTexture("posW");
        const layer = rd.getTexture("layer"), layerDepth = rd.getTexture("layerDepth"), trans = rd.getTexture("transmittance");
        const p = resolveVolumetricCloudsParams(this.p);
        const W = dst.width, H = dst.height;
        const factor = p.resolution === "quarter" ? 4 : p.resolution === "half" ? 2 : 1;
        const temporal = p.temporal && p.temporalBlend > 0;
        const frame = this.history.begin(this.scene, `${W}x${H}:${factor}:${temporal}`);

        if (p.coverage <= 0 || p.density <= 0 || p.opacity <= 0 || !this.scene) {
            // Identity: no clouds. Copy, clear the optional layers, start the history afresh when clouds come back.
            if (color && color.format === dst.format && color.width === W && color.height === H) ctx.copyTexture(dst, color);
            else if (color) ctx.blit(color, dst, "nearest");
            else ctx.clearTexture(dst, [0, 0, 0, 1]);
            if (layer) ctx.clearTexture(layer, [0, 0, 0, 0]);
            if (layerDepth) ctx.clearTexture(layerDepth, [kCloudNoDepth, 0, 0, 0]);
            if (trans) ctx.clearTexture(trans, [1, 1, 1, 1]);
            this.history.reset();
            this.prevCam = null;
            return;
        }

        const cam = cloudCamera(this.scene);
        const f = cloudFrameConstants(p, cam, this.lightDirection);
        const lw = Math.ceil(W / factor), lh = Math.ceil(H / factor);
        if (!this.march || this.march[0].width !== lw || this.march[0].height !== lh) {
            const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            this.march = [0, 1].map(() => this.device.createTexture2D(lw, lh, ResourceFormat.RGBA32Float, 1, 1, undefined, flags)) as [Texture, Texture];
        }
        const hA = this.histA.get(ctx, W, H, frame), hB = this.histB.get(ctx, W, H, frame);
        const historyValid = temporal && frame.frameIndex > 0 && this.prevCam !== null;
        const prev = historyValid ? this.prevCam! : cam;
        const wd = historyValid ? p.wind.map((w) => w * (p.time - this.prevTime)) : [0, 0, 0];
        const flags = (posW ? 1 : 0) | (historyValid ? 2 : 0) | (temporal ? 4 : 0) | (color ? 8 : 0);
        const sq = (a: number[]) => dot(a, a);
        const setCB = (cp: ComputePass) => {
            const root = cp.getRootVar();
            const cb = root["CB"]!;
            cb["gDims"] = [W, H, lw, lh];
            cb["gFlags"] = [factor, flags, p.steps, p.lightSteps];
            cb["gSeed"] = [f.salt, temporal ? frame.frameIndex : 0, p.octaves, p.detailOctaves];
            cb["gCamPos"] = [...cam.pos, Math.hypot(...cam.W)];
            cb["gCamU"] = [...cam.U, sq(cam.U)];
            cb["gCamV"] = [...cam.V, sq(cam.V)];
            cb["gCamW"] = [...cam.W, sq(cam.W)];
            cb["gPrevPos"] = [...prev.pos, Math.hypot(...prev.W)];
            cb["gPrevU"] = [...prev.U, sq(prev.U)];
            cb["gPrevV"] = [...prev.V, sq(prev.V)];
            cb["gPrevW"] = [...prev.W, sq(prev.W)];
            cb["gUp"] = [...f.up, f.altitude];
            cb["gE1"] = [...f.e1, 0];
            cb["gE2"] = [...f.e2, 0];
            cb["gSlab"] = [p.cloudBottom, p.cloudTop, p.planetRadius, p.maxDistance];
            cb["gShape"] = [p.shapeScale, p.detailScale, p.detailStrength, p.weatherScale];
            cb["gWeather"] = [p.coverage, p.cloudType, p.weatherAmount, p.typeVariation];
            cb["gMedia"] = [p.density, p.albedo, p.powder, p.minTransmittance];
            cb["gPhase"] = [p.phaseForward, p.phaseBackward, p.phaseMix, p.jitter];
            cb["gLightDir"] = [...f.light, p.lightStep];
            cb["gLightCol"] = [...f.lightRgb, p.lightStepGrowth];
            cb["gAmbTop"] = [...p.ambientTop, p.fadeDistance];
            cb["gAmbBot"] = [...p.ambientBottom, p.temporalBlend];
            cb["gOffset"] = [...f.offset, p.opacity];
            cb["gWindDelta"] = [...wd, 0];
            root["gPosW"] = posW ?? dummyTexture(this.device);
            return root;
        };
        {
            const cp = this.pass("march");
            const root = setCB(cp);
            root["gOutA"] = this.march[0];
            root["gOutB"] = this.march[1];
            cp.execute(ctx, lw, lh);
        }
        {
            const cp = this.pass("resolve");
            const root = setCB(cp);
            root["gColor"] = color ?? dummyTexture(this.device);
            root["gMarchA"] = this.march[0];
            root["gMarchB"] = this.march[1];
            root["gHistA"] = hA.read;
            root["gHistB"] = hB.read;
            root["gOutA"] = hA.write;
            root["gOutB"] = hB.write;
            root["gDst"] = dst;
            root["gLayer"] = layer ?? dummyTexture(this.device, "out:layer");
            root["gLayerDepth"] = layerDepth ?? dummyTexture(this.device, "out:layerDepth");
            root["gTrans"] = trans ?? dummyTexture(this.device, "out:transmittance");
            cp.execute(ctx, W, H);
        }
        this.prevCam = temporal ? cam : null;
        this.prevTime = p.time;
    }
}

registerRenderPass("VolumetricClouds", (device, props) => new VolumetricClouds(device, props), {
    desc: "Raymarched volumetric clouds (altitude slab, weather map, light march, temporal reprojection; DepthComposite layer)",
});
