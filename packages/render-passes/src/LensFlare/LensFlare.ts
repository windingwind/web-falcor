/**
 * LensFlare: screen-space lens flare driven by the bright pixels of the frame: ghosts, a halo ring and an anamorphic
 * (horizontal) streak, added on top of the HDR colour (place it before tone mapping, next to bloom).
 *
 * 1. Bright pass (half resolution): 2x2 box downsample of `src` (or of the optional `bright` input, e.g. an emissive or
 *    bloom buffer) and a soft-knee threshold: c x max(soft, max(c) - threshold) / max(c), soft = clamp(max(c) -
 *    threshold + knee, 0, 2 knee)^2 / (4 knee).
 * 2. Flare (half resolution), at the image mirrored through the centre (uv' = 1 - uv), ghost vector g = (0.5 - uv') x
 *    ghostSpacing:
 *      ghosts = sum_{i < ghostCount} bright(uv' + i g) x (1 - |0.5 - uv' - i g| / sqrt(0.5))^2;
 *      halo   = bright(uv' + normalize(g) x haloRadius) x (1 - |0.5 - that| / sqrt(0.5))^haloFalloff;
 *      both with a chromatic offset (red / blue sampled at +- chromatic along g);
 *      streak = normalised horizontal kernel exp(-4 |j| / 24), j in [-24, 24], tap spacing streakLength x width / 24.
 *    flare = ghosts x ghostIntensity x ghostTint + halo x haloIntensity x haloTint + streak x streakIntensity x streakTint.
 * 3. Composite (full resolution): dst.rgb = src.rgb + intensity x bilinear(flare), dst.a = src.a. Optional `flare`
 *    output = intensity x bilinear(flare).
 * Sampling is explicit bilinear with clamp-to-edge addressing (ghost/halo taps outside the image are skipped, streak
 * taps outside the row read zero), so the result does not depend on sampler state.
 *
 * Stateless (no history). Identity: intensity 0 copies src bit for bit; a frame with no pixel above the threshold (knee
 * 0) also reproduces src exactly.
 *
 * The method follows John Chapman's "pseudo lens flare" (ghosts and halo from a thresholded, mirrored image, as also used
 * by Filament's flare material, Apache-2.0) and KinoStreak's horizontal streak (MIT, Copyright (c) 2017 Keijiro
 * Takahashi). The code is written for this pass and copies neither.
 *
 * Inputs : src, bright? (replaces src as the source of the bright pass)
 * Outputs: dst (RGBA32Float), flare? (RGBA32Float)
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
    type Texture,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture, parseColour } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/LensFlare/LensFlare.cs.slang";
const kEntries = ["prefilter", "flare", "composite"] as const;
type Entry = (typeof kEntries)[number];
export const kLensFlareMaxGhosts = 8;

const kDefaults = {
    intensity: 1,
    threshold: 1,
    knee: 0.5,
    ghostCount: 4,
    ghostSpacing: 0.3,
    ghostIntensity: 1,
    ghostTint: [1, 0.9, 0.8] as number[] | string,
    haloRadius: 0.45,
    haloFalloff: 5,
    haloIntensity: 0.5,
    haloTint: [0.8, 0.9, 1] as number[] | string,
    chromatic: 0.006,
    streakIntensity: 0,
    streakLength: 0.4,
    streakTint: [0.55, 0.7, 1] as number[] | string,
};
export type LensFlareParams = typeof kDefaults;

export class LensFlare extends RenderPass {
    private p: LensFlareParams = JSON.parse(JSON.stringify(kDefaults));
    private passes = new Map<Entry, ComputePass>();
    private half: [Texture, Texture] | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(kDefaults) as (keyof LensFlareParams)[]) if (props.has(k)) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Intensity", this.p.intensity, 0, 4, 0.01, (v) => (this.p.intensity = v));
        ui.slider("Threshold", this.p.threshold, 0, 20, 0.01, (v) => (this.p.threshold = v));
        ui.slider("Ghosts", this.p.ghostCount, 0, kLensFlareMaxGhosts, 1, (v) => (this.p.ghostCount = Math.round(v)));
        ui.slider("Ghost spacing", this.p.ghostSpacing, 0, 1, 0.01, (v) => (this.p.ghostSpacing = v));
        ui.slider("Halo", this.p.haloIntensity, 0, 4, 0.01, (v) => (this.p.haloIntensity = v));
        ui.slider("Streak", this.p.streakIntensity, 0, 8, 0.01, (v) => (this.p.streakIntensity = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "HDR colour");
        addTexInput(r, "bright", "Optional source of the bright pass (replaces src there)", true);
        addColorOutput(r, "dst", "Colour with the flare added", w, h);
        addColorOutput(r, "flare", "Flare only (rgb)", w, h, true);
        return r;
    }

    private kernel(e: Entry): ComputePass {
        let k = this.passes.get(e);
        if (!k) this.passes.set(e, (k = ComputePass.create(this.device, { path: kShaderFile, csEntry: e })));
        return k;
    }

    /** The half-resolution bright and flare images of the last frame, for tests. */
    get halfTextures(): readonly Texture[] | null {
        return this.half;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const bright = rd.getTexture("bright"), flareOut = rd.getTexture("flare");
        const W = dst.width, H = dst.height, W2 = Math.max(1, (W + 1) >> 1), H2 = Math.max(1, (H + 1) >> 1);
        if (!this.half || this.half[0].width !== W2 || this.half[0].height !== H2) {
            const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            this.half = [0, 1].map(() => this.device.createTexture2D(W2, H2, ResourceFormat.RGBA32Float, 1, 1, undefined, flags)) as [Texture, Texture];
        }
        const [B, F] = this.half;
        const p = this.p;
        const consts: Record<string, unknown> = {
            gFrameDim: [W, H], gHalfDim: [W2, H2], gHasBright: bright ? 1 : 0,
            gThreshold: Number(p.threshold), gKnee: Math.max(Number(p.knee), 0), gIntensity: Number(p.intensity),
            gGhostCount: Math.max(0, Math.min(Math.round(Number(p.ghostCount)), kLensFlareMaxGhosts)), gGhostSpacing: Number(p.ghostSpacing),
            gGhostIntensity: Number(p.ghostIntensity), gHaloRadius: Number(p.haloRadius), gGhostTint: parseColour(p.ghostTint),
            gHaloFalloff: Math.max(Number(p.haloFalloff), 0), gHaloTint: parseColour(p.haloTint), gHaloIntensity: Number(p.haloIntensity),
            gStreakTint: parseColour(p.streakTint), gStreakIntensity: Number(p.streakIntensity), gStreakLength: Math.max(Number(p.streakLength), 0),
            gChromatic: Number(p.chromatic), gHasFlareOut: flareOut ? 1 : 0,
        };
        const d = (slot: string) => dummyTexture(this.device, slot);
        const run = (e: Entry, x: number, y: number, bind: { brightHalf: Texture; flareHalf: Texture; halfOut: Texture; dst: Texture; flareOut: Texture }) => {
            const k = this.kernel(e);
            const root = k.getRootVar();
            const cb = root["CB"]!;
            for (const [name, v] of Object.entries(consts)) cb[name] = v as never;
            root["gSrc"] = src;
            root["gBright"] = bright ?? d("in");
            root["gBrightHalf"] = bind.brightHalf;
            root["gFlareHalf"] = bind.flareHalf;
            root["gHalfOut"] = bind.halfOut;
            root["gDst"] = bind.dst;
            root["gFlareOut"] = bind.flareOut;
            k.execute(ctx, x, y);
        };
        if (Number(p.intensity) !== 0) {
            run("prefilter", W2, H2, { brightHalf: d("in:lf1"), flareHalf: d("in:lf2"), halfOut: B, dst: d("out:lf1"), flareOut: d("out:lf2") });
            run("flare", W2, H2, { brightHalf: B, flareHalf: d("in:lf2"), halfOut: F, dst: d("out:lf1"), flareOut: d("out:lf2") });
        }
        run("composite", W, H, { brightHalf: d("in:lf1"), flareHalf: F, halfOut: d("out:lf3"), dst, flareOut: flareOut ?? d("out:lf2") });
    }
}

registerRenderPass("LensFlare", (device, props) => new LensFlare(device, props), {
    desc: "Screen-space lens flare from bright pixels: ghosts, halo ring and anamorphic streak (chromatic, tinted)",
});
