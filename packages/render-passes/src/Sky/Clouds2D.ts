/**
 * Clouds2D: a cheap illustrative cloud layer on a plane above the viewer, drawn over a sky background (PhysicalSky,
 * ProceduralNightSky, an env background) behind the scene geometry. Written from scratch for web-falcor
 * (integer-hash value noise; no third-party shader code).
 *
 * For a view direction d above the horizon the cloud plane is hit at uv = d.xz * height / (d.y + curvature) (curvature
 * keeps the horizon finite), then uv = uv * scale + wind * time. Density: fBm value noise with one domain warp,
 * dens = saturate((n - (1 - coverage)) / softness). Lighting (2D): the density a short step towards the light
 * (lightDir projected on the plane) gives the self-shadow T = exp(-absorption * dens_l); colour = lerp(shadeColour,
 * litColour, T) * brightness, plus a silver lining near the light direction on thin cloud. Coverage fades out towards
 * the horizon (horizonFade). `upAxis` (default +y) sets the sky's up direction (camera and light are rotated into
 * the +y-up sky frame; evalClouds2D takes sky-frame directions). Everything is view-direction only (the clouds sit at infinity).
 *
 * Inputs : color? (sky/scene colour), posW? (w = 0 marks background; geometry pixels pass `color` through)
 * Outputs: dst   (RGBA32Float) = color * (1 - a) + cloud * a over the background;
 *          layer (optional) premultiplied (cloud * a, a) for DepthComposite, zero where posW marks geometry;
 *          layerDepth (optional) r = kCloudDepth (2.9e38: behind surfaces, in front of the sky layers at 3e38).
 * Stateless: `time` is a property.
 * Identity: coverage 0 or opacity 0 returns `color` bit for bit.
 * CPU reference: evalClouds2D() (same formulas).
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture, keyLightDir, parseColour } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Sky/Clouds2D.cs.slang";
/** Layer depth of the cloud layer (view-depth units): behind any surface, in front of a sky layer at 3e38. */
export const kCloudDepth = 2.9e38;

export interface Clouds2DParams {
    coverage: number;
    softness: number;
    scale: number;
    height: number;
    curvature: number;
    wind: number[];
    time: number;
    seed: number;
    octaves: number;
    warp: number;
    /** Towards the light; [0, 0, 0] = the scene's first directional light (else straight up). */
    lightDir: number[];
    shadowStep: number;
    absorption: number;
    litColour: number[];
    shadeColour: number[];
    brightness: number;
    silver: number;
    opacity: number;
    horizonFade: number;
    /** World up axis of the sky (default +y; e.g. [0, 0, 1] for z-up scenes). */
    upAxis: number[];
}

export const kDefaultClouds2D: Clouds2DParams = {
    coverage: 0.5,
    softness: 0.18,
    scale: 1,
    height: 1,
    curvature: 0.06,
    wind: [0.03, 0.01],
    time: 0,
    seed: 0,
    octaves: 5,
    warp: 0.7,
    lightDir: [0, 0, 0],
    shadowStep: 0.06,
    absorption: 2.5,
    litColour: [1, 1, 1],
    shadeColour: [0.42, 0.47, 0.58],
    brightness: 1,
    silver: 0.6,
    opacity: 1,
    upAxis: [0, 1, 0],
    horizonFade: 0.12,
};

/** Parameters with defaults filled in, colours parsed and ranges clamped (lightDir is resolved by the caller). */
export function resolveClouds2DParams(p: Partial<Clouds2DParams>): Clouds2DParams {
    const r = { ...kDefaultClouds2D, ...p } as Clouds2DParams;
    r.litColour = parseColour(r.litColour);
    r.shadeColour = parseColour(r.shadeColour);
    r.coverage = Math.min(Math.max(Number(r.coverage), 0), 1);
    r.opacity = Math.min(Math.max(Number(r.opacity), 0), 1);
    r.softness = Math.max(Number(r.softness), 1e-4);
    r.curvature = Math.max(Number(r.curvature), 1e-4);
    r.horizonFade = Math.max(Number(r.horizonFade), 1e-4);
    r.upAxis = [0, 1, 2].map((i) => Number(r.upAxis?.[i] ?? 0));
    if (!r.upAxis.some((x) => x !== 0)) r.upAxis = [0, 1, 0];
    r.octaves = Math.min(Math.max(Math.round(r.octaves), 1), 8);
    r.seed = Math.max(0, Math.floor(Number(r.seed))) >>> 0;
    r.wind = [Number(r.wind[0] ?? 0), Number(r.wind[1] ?? 0)];
    return r;
}

// ---- integer hash and value noise (bit-identical integer part on CPU and GPU) ----

/** pcg3d (Jarzynski & Olano 2020), 32-bit unsigned arithmetic. */
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
const smooth = (e0: number, e1: number, x: number) => { const t = sat((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };

/** 2D value noise in [0, 1), smoothstep interpolation. */
function vnoise2(x: number, y: number, salt: number): number {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const wx = fx * fx * (3 - 2 * fx), wy = fy * fy * (3 - 2 * fy);
    const v = (a: number, b: number) => u01(pcg3d(((ix + a) | 0) >>> 0, ((iy + b) | 0) >>> 0, salt)[0]);
    return mix(mix(v(0, 0), v(1, 0), wx), mix(v(0, 1), v(1, 1), wx), wy);
}
/** fBm of vnoise2 with a per-octave rotation (about 37 degrees) and lacunarity 2.02, normalised to [0, 1). */
function fbm2(x: number, y: number, oct: number, salt: number): number {
    let s = 0, a = 0.5, n = 0;
    for (let o = 0; o < oct; o++) {
        s += a * vnoise2(x, y, (salt + o) >>> 0); n += a;
        const nx = 2.02 * (0.8 * x - 0.6 * y), ny = 2.02 * (0.6 * x + 0.8 * y);
        x = nx; y = ny; a *= 0.5;
    }
    return s / n;
}

/** Density field on the cloud plane (uv already scaled and advected). */
function density(p: Clouds2DParams, u: number, v: number, salt: number): number {
    const w = fbm2(0.5 * u + 3.1, 0.5 * v + 7.7, 3, (salt + 20) >>> 0) - 0.5;
    const n = fbm2(u + p.warp * w, v - p.warp * w, p.octaves, (salt + 40) >>> 0);
    return sat((n - (1 - p.coverage)) / p.softness);
}

/**
 * CPU reference of the shader for direction `dir` (sky frame, y up; world vectors go through clouds2DSkyFrame(upAxis)) and a unit light direction `light`.
 * Returns the premultiplied cloud colour and its alpha (dst = color * (1 - a) + rgb).
 */
export function evalClouds2D(dir: readonly number[], params: Partial<Clouds2DParams> = {}, light: readonly number[] = [0, 1, 0]): { rgb: [number, number, number]; a: number } {
    const p = resolveClouds2DParams(params);
    const l = Math.hypot(dir[0]!, dir[1]!, dir[2]!) || 1;
    const d = [dir[0]! / l, dir[1]! / l, dir[2]! / l];
    if (d[1]! <= 0 || p.coverage <= 0 || p.opacity <= 0) return { rgb: [0, 0, 0], a: 0 };
    const salt = Math.imul(p.seed, 131) >>> 0;
    const k = p.height / (d[1]! + p.curvature);
    const u = d[0]! * k * p.scale + p.wind[0]! * p.time, v = d[2]! * k * p.scale + p.wind[1]! * p.time;
    const dens = density(p, u, v, salt);
    const a = dens * p.opacity * smooth(0, p.horizonFade, d[1]!);
    if (a <= 0) return { rgb: [0, 0, 0], a: 0 };
    const ll = Math.hypot(light[0]!, light[2]!);
    const lx = ll > 1e-6 ? light[0]! / ll : 0, lz = ll > 1e-6 ? light[2]! / ll : 0;
    const ds = density(p, u + lx * p.shadowStep, v + lz * p.shadowStep, salt);
    const T = Math.exp(-p.absorption * ds);
    const cosL = Math.max(d[0]! * light[0]! + d[1]! * light[1]! + d[2]! * light[2]!, 0);
    const sil = p.silver * Math.pow(cosL, 8) * (1 - dens);
    const rgb = [0, 1, 2].map((c) => (mix(p.shadeColour[c]!, p.litColour[c]!, T) + sil * p.litColour[c]!) * p.brightness * a) as [number, number, number];
    return { rgb, a };
}


/** Rotation taking `up` to +y (minimal rotation, Rodrigues); identity for +y. Sky formulas use +y up. */
export function clouds2DSkyFrame(up: readonly number[]): (v: readonly number[]) => [number, number, number] {
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

export class Clouds2D extends RenderPass {
    private p: Clouds2DParams = { ...kDefaultClouds2D };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(this.p) as (keyof Clouds2DParams)[]) (this.p as unknown as Record<string, unknown>)[k] = props.get(k, this.p[k]);
        resolveClouds2DParams(this.p);
    }

    /** The light direction used for shading (lightDir, else the scene's key light, else up). */
    get lightDirection(): [number, number, number] {
        return keyLightDir(this.scene, this.p.lightDir);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Coverage", this.p.coverage, 0, 1, 0.01, (v) => (this.p.coverage = v));
        ui.slider("Softness", this.p.softness, 0.01, 1, 0.01, (v) => (this.p.softness = v));
        ui.slider("Scale", this.p.scale, 0.05, 10, 0.01, (v) => (this.p.scale = v));
        ui.slider("Time", this.p.time, 0, 600, 0.01, (v) => (this.p.time = v));
        ui.slider("Opacity", this.p.opacity, 0, 1, 0.01, (v) => (this.p.opacity = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "color", "Sky / scene colour", true);
        addTexInput(r, "posW", "World position, w = 0 marks background", true);
        addColorOutput(r, "dst", "Clouds over the background", w, h);
        addColorOutput(r, "layer", "Premultiplied cloud layer for DepthComposite", w, h, true);
        addColorOutput(r, "layerDepth", "Layer depth (r = 2.9e38)", w, h, true);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        if (!dst) return;
        const color = rd.getTexture("color"), posW = rd.getTexture("posW");
        const layer = rd.getTexture("layer"), layerDepth = rd.getTexture("layerDepth");
        const p = resolveClouds2DParams(this.p);
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
        const toSky = clouds2DSkyFrame(p.upAxis);
        const [rs, us, fs] = [toSky(right), toSky(trueUp), toSky(fwd)];
        const L = toSky(this.lightDirection);
        const ll = Math.hypot(L[0], L[2]);
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        cb["gFrameDim"] = [W, H];
        cb["gHasColor"] = color ? 1 : 0;
        cb["gHasPos"] = posW ? 1 : 0;
        cb["gSalt"] = Math.imul(p.seed, 131) >>> 0;
        cb["gOctaves"] = p.octaves;
        cb["gHasLayer"] = layer ? 1 : 0;
        cb["gHasLayerDepth"] = layerDepth ? 1 : 0;
        cb["gCamRight"] = [...rs.map((v) => v * tanY * aspect), p.coverage];
        cb["gCamUp"] = [...us.map((v) => v * tanY), p.softness];
        cb["gCamFwd"] = [...fs, p.horizonFade];
        cb["gPlane"] = [p.height, p.curvature, p.scale, p.warp];
        cb["gWind"] = [p.wind[0]! * p.time, p.wind[1]! * p.time, p.opacity, p.brightness];
        cb["gLight"] = [...L, p.silver];
        cb["gLight2"] = [ll > 1e-6 ? L[0] / ll : 0, ll > 1e-6 ? L[2] / ll : 0, p.shadowStep, p.absorption];
        cb["gLit"] = [...p.litColour, 0];
        cb["gShade"] = [...p.shadeColour, 0];
        root["gColor"] = color ?? dummyTexture(this.device);
        root["gPosW"] = posW ?? dummyTexture(this.device);
        root["gDst"] = dst;
        root["gLayer"] = layer ?? dummyTexture(this.device, "out:layer");
        root["gLayerDepth"] = layerDepth ?? dummyTexture(this.device, "out:layerDepth");
        this.pass.execute(ctx, W, H);
    }
}

registerRenderPass("Clouds2D", (device, props) => new Clouds2D(device, props), {
    desc: "Illustrative 2D cloud layer over the sky background (DepthComposite layer)",
});
