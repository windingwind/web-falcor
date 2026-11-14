/**
 * LightCookiePattern: generates procedural light cookies for DeferredLighting's optional `cookie` input: drifting cloud
 * shadows on the sun, window frames and dappled light through foliage. The output is a size x size RGBA32Float texture,
 * rgb = lerp(1, shadowColor, strength x occlusion), a = 1. Every pattern tiles seamlessly, so it can be repeated over
 * the scene by a directional light (`cookieWorldSize`) or projected once by a spot light.
 *
 * Occlusion by `mode` (uv = texel centre in [0, 1]^2):
 *  - clouds : fBm of periodic value noise (`octaves`, base period `scale` cells, doubling per octave, gain 0.5) at
 *             frac(uv - wind x time); occlusion = smoothstep(1 - coverage - softness, 1 - coverage + softness, fbm);
 *  - window : one window in `rect` (uv x0, y0, x1, y1), `panes` = [columns, rows], outer `frame` and inner `mullion`
 *             widths as fractions of the window; outside the rect the light is blocked (a wall), edges soft by
 *             `softness`;
 *  - leaves : sun flecks through a canopy: `scale` x `scale` cells, one jittered disc of light per cell (radius
 *             spotSize x (0.5 .. 1) cells, soft edge `softness`), swaying by `sway` cells at `swaySpeed` cycles/s;
 *             occlusion = 1 - brightest fleck (3 x 3 neighbour cells).
 * Patterns depend only on the parameters (`seed`, `time`), never on the rendered image.
 *
 * Stateless: animation is a function of the `time` property. Identity: strength 0 gives a white cookie, which leaves
 * DeferredLighting's output bitwise unchanged.
 *
 * Cloud shadows after the scrolling-noise idea of PlayCanvas's cloud-shadows example (MIT,
 * https://github.com/playcanvas/engine); hash: PCG (Jarzynski & Olano 2020). Written for this pass.
 *
 * Outputs: dst (RGBA32Float, size x size)
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
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { enumIndex, parseColour } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/RasterLighting/LightCookiePattern.cs.slang";
export const kLightCookieModes = ["clouds", "window", "leaves"] as const;

const kDefaults = {
    mode: "clouds" as string,
    size: 512,
    seed: 0,
    strength: 0.8,
    shadowColor: [0, 0, 0] as number[] | string,
    time: 0,
    // clouds
    coverage: 0.5,
    softness: 0.15,
    scale: 4,
    octaves: 5,
    wind: [0.01, 0.004] as number[],
    // window
    rect: [0.2, 0.15, 0.8, 0.85] as number[],
    panes: [2, 3] as number[],
    frame: 0.04,
    mullion: 0.03,
    // leaves
    spotSize: 0.35,
    sway: 0.05,
    swaySpeed: 0.3,
};
export type LightCookiePatternParams = typeof kDefaults;

export class LightCookiePattern extends RenderPass {
    private p: LightCookiePatternParams = JSON.parse(JSON.stringify(kDefaults));
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(kDefaults) as (keyof LightCookiePatternParams)[]) if (props.has(k)) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        enumIndex("LightCookiePattern", "mode", this.p.mode, kLightCookieModes);
        const size = Math.round(Number(this.p.size));
        if (!(size >= 1 && size <= 4096)) throw new Error(`LightCookiePattern: size must be in [1, 4096] (${this.p.size})`);
        this.p.size = size;
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Strength", this.p.strength, 0, 1, 0.01, (v) => (this.p.strength = v));
        ui.slider("Time", this.p.time, 0, 600, 0.01, (v) => (this.p.time = v));
        ui.slider("Coverage", this.p.coverage, 0, 1, 0.01, (v) => (this.p.coverage = v));
        ui.slider("Softness", this.p.softness, 0, 0.5, 0.005, (v) => (this.p.softness = v));
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("dst", "Light cookie (rgb multiplier)").texture2D(this.p.size, this.p.size).format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    /** The parameters as the shader uses them (clamped), for CPU references. */
    static resolved(p: LightCookiePatternParams) {
        const mode = enumIndex("LightCookiePattern", "mode", p.mode, kLightCookieModes);
        const soft = Number(p.softness);
        return {
            mode, seed: Math.max(0, Math.round(Number(p.seed))) >>> 0, strength: Math.min(Math.max(Number(p.strength), 0), 1),
            shadowColor: parseColour(p.shadowColor), time: Number(p.time), wind: [Number(p.wind[0] ?? 0), Number(p.wind[1] ?? 0)],
            coverage: Math.min(Math.max(Number(p.coverage), 0), 1),
            softness: mode === 2 ? Math.min(Math.max(soft, 1e-4), 1) : Math.max(soft, 1e-5),
            scale: Math.max(1, Math.round(Number(p.scale))), octaves: Math.min(Math.max(Math.round(Number(p.octaves)), 1), 8),
            spotSize: Math.min(Math.max(Number(p.spotSize), 0), 0.8), sway: Math.min(Math.max(Number(p.sway), 0), 0.3), swaySpeed: Number(p.swaySpeed),
            rect: [0, 1, 2, 3].map((i) => Number(p.rect[i] ?? [0, 0, 1, 1][i])), panes: [0, 1].map((i) => Math.max(1, Math.round(Number(p.panes[i] ?? 1)))),
            frame: Math.max(Number(p.frame), 0), mullion: Math.max(Number(p.mullion), 0),
        };
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        if (!dst) return;
        if (!this.pass) this.pass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "main" });
        const r = LightCookiePattern.resolved(this.p);
        const cb = this.pass.getRootVar()["CB"]!;
        const consts: Record<string, unknown> = {
            gDim: [dst.width, dst.height], gMode: r.mode, gSeed: r.seed, gShadowColor: r.shadowColor, gStrength: r.strength,
            gWind: r.wind, gTime: r.time, gCoverage: r.coverage, gSoftness: r.softness, gScale: r.scale, gOctaves: r.octaves,
            gSpotSize: r.spotSize, gSway: r.sway, gSwaySpeed: r.swaySpeed, gRect: r.rect, gPanes: r.panes, gFrame: r.frame, gMullion: r.mullion,
        };
        for (const [name, v] of Object.entries(consts)) cb[name] = v as never;
        this.pass.getRootVar()["gDst"] = dst;
        this.pass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("LightCookiePattern", (device, props) => new LightCookiePattern(device, props), {
    desc: "Procedural light cookies (cloud shadows, window frame, dappled foliage) for DeferredLighting's cookie input",
});
