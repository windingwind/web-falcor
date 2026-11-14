/**
 * RainWetness: wet surfaces and puddles for rain scenes, applied between deferred lighting and screen-space reflections.
 *
 * The pass works on the lit image and the G-buffer, so no material or lighting pass has to change:
 *  - wetness: every surface gets wet by `wetness` x a facing term (up-facing surfaces fully, vertical ones by
 *    `sideWetness`, overhangs less); wet surfaces lose diffuse light (`darkening` x the DeferredLighting `diffuse`
 *    output, or x the colour when it is not connected) and get glossier (roughness x (1 - gloss x wet));
 *  - puddles: on up-facing surfaces (N.up above `puddleUpMin`) where a planar fBm exceeds 1 - `puddleCoverage`; a puddle
 *    is a water film: F0 0.02, roughness `puddleRoughness`, normal = up, the underlying specular damped by
 *    `puddleSpecular` (needs the `specular` input);
 *  - ripples: expanding rings on the puddles (`rippleScale` grid, `rippleRate` rings per second per cell), a pure
 *    function of `time` and `seed` (set `time` per frame for clips; no history).
 * The modified `specRoughOut` and `normWOut` feed SSR (and any other screen-space pass), which then reflects the scene in the
 * puddles. `wetness` is a debug/mask output: wet (r), puddle (g), ripple height (b).
 *
 * Identity: wetness 0 and puddleCoverage 0 copy color, specRough and normW to dst, specRoughOut and normWOut bit for
 * bit (as does every pixel with neither wetness nor puddle).
 *
 * Inputs : color, posW, normW, specRough (GBufferRaster), diffuse? specular? (DeferredLighting optional outputs)
 * Outputs: dst, specRoughOut, normWOut, wetness? (RGBA32Float)
 * Params : wetness, sideWetness, darkening, gloss, puddleCoverage, puddleEdge, puddleScale, puddleUpMin, puddleRoughness,
 *          puddleSpecular, rippleStrength, rippleScale, rippleRate, time, seed, upAxis
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
import { addColorOutput, addTexInput, dummyTexture } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Weather/RainWetness.cs.slang";

const kDefaults = {
    wetness: 0.8,
    sideWetness: 0.3,
    darkening: 0.45,
    gloss: 0.6,
    puddleCoverage: 0.35,
    puddleEdge: 0.08,
    puddleScale: 0.35,
    puddleUpMin: 0.9,
    puddleRoughness: 0.02,
    puddleSpecular: 0.8,
    rippleStrength: 1,
    rippleScale: 0.25,
    rippleRate: 1.5,
    time: 0,
    seed: 0,
    upAxis: [0, 1, 0] as number[],
};
export type RainWetnessParams = typeof kDefaults;

/** Unit up vector and an orthonormal planar basis (t1, t2) for it (shared with the tests). */
export function planarBasis(upAxis: ArrayLike<number>): { up: number[]; t1: number[]; t2: number[] } {
    let up = [Number(upAxis[0]), Number(upAxis[1]), Number(upAxis[2])];
    const l = Math.hypot(up[0]!, up[1]!, up[2]!);
    up = l > 0 ? up.map((x) => x / l) : [0, 1, 0];
    const a = Math.abs(up[0]!) < 0.9 ? [1, 0, 0] : [0, 0, 1];
    const cross = (x: number[], y: number[]) => [x[1]! * y[2]! - x[2]! * y[1]!, x[2]! * y[0]! - x[0]! * y[2]!, x[0]! * y[1]! - x[1]! * y[0]!];
    let t2 = cross(up, a);
    const l2 = Math.hypot(t2[0]!, t2[1]!, t2[2]!);
    t2 = t2.map((x) => x / l2);
    const t1 = cross(t2, up);
    return { up, t1, t2 };
}

export class RainWetness extends RenderPass {
    private p: RainWetnessParams = JSON.parse(JSON.stringify(kDefaults));
    private cp: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof RainWetnessParams)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Wetness", this.p.wetness, 0, 1, 0.01, (v) => (this.p.wetness = v));
        ui.slider("Darkening", this.p.darkening, 0, 1, 0.01, (v) => (this.p.darkening = v));
        ui.slider("Gloss", this.p.gloss, 0, 1, 0.01, (v) => (this.p.gloss = v));
        ui.slider("Puddle coverage", this.p.puddleCoverage, 0, 1, 0.01, (v) => (this.p.puddleCoverage = v));
        ui.slider("Ripple strength", this.p.rippleStrength, 0, 4, 0.01, (v) => (this.p.rippleStrength = v));
        ui.slider("Time", this.p.time, 0, 60, 0.01, (v) => (this.p.time = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "color", "Lit colour (HDR)");
        addTexInput(r, "posW", "World position (w = 0: background)");
        addTexInput(r, "normW", "World shading normal");
        addTexInput(r, "specRough", "Specular reflectance (rgb) and roughness (a)");
        addTexInput(r, "diffuse", "DeferredLighting.diffuse (darkening acts on it; else on the colour)", true);
        addTexInput(r, "specular", "DeferredLighting.specular (damped under puddles)", true);
        addColorOutput(r, "dst", "Colour with wet darkening", w, h);
        addColorOutput(r, "specRoughOut", "Wet specular reflectance and roughness", w, h);
        addColorOutput(r, "normWOut", "Normal with puddle flattening and ripples", w, h);
        addColorOutput(r, "wetness", "wet (r), puddle (g), ripple height (b)", w, h, true);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst"), color = rd.getTexture("color"), posW = rd.getTexture("posW");
        const normW = rd.getTexture("normW"), specRough = rd.getTexture("specRough");
        const srOut = rd.getTexture("specRoughOut"), nOut = rd.getTexture("normWOut");
        if (!dst || !color || !posW || !normW || !specRough || !srOut || !nOut) return;
        const W = dst.width, H = dst.height;
        const diffuse = rd.getTexture("diffuse"), specular = rd.getTexture("specular");
        const { up, t1, t2 } = planarBasis(this.p.upAxis);
        this.cp ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: "main" });
        const root = this.cp.getRootVar();
        const cb = root["CB"]!;
        const n = (k: keyof RainWetnessParams) => Number(this.p[k]);
        const vals: Record<string, unknown> = {
            gFrameDim: [W, H], gHasDiffuse: diffuse ? 1 : 0, gHasSpecular: specular ? 1 : 0,
            gUp: up, gT1: t1, gT2: t2,
            gWetness: Math.min(Math.max(n("wetness"), 0), 1), gSideWetness: Math.min(Math.max(n("sideWetness"), 0), 1),
            gDarkening: Math.min(Math.max(n("darkening"), 0), 1), gGloss: Math.min(Math.max(n("gloss"), 0), 1),
            gPuddleCoverage: Math.min(Math.max(n("puddleCoverage"), 0), 1), gPuddleEdge: Math.max(n("puddleEdge"), 1e-4),
            gPuddleScale: n("puddleScale"), gPuddleUpMin: Math.min(Math.max(n("puddleUpMin"), -1), 0.999),
            gPuddleRoughness: Math.min(Math.max(n("puddleRoughness"), 0), 1), gPuddleSpecular: Math.min(Math.max(n("puddleSpecular"), 0), 1),
            gRippleStrength: Math.max(n("rippleStrength"), 0), gRippleScale: Math.max(n("rippleScale"), 1e-3),
            gRippleRate: n("rippleRate"), gTime: n("time"), gSeed: Math.max(0, Math.round(n("seed"))) >>> 0,
        };
        for (const [k, v] of Object.entries(vals)) cb[k] = v as never;
        root["gColor"] = color;
        root["gDiffuse"] = diffuse ?? dummyTexture(this.device);
        root["gSpecular"] = specular ?? dummyTexture(this.device, "in2");
        root["gPosW"] = posW;
        root["gNormW"] = normW;
        root["gSpecRough"] = specRough;
        root["gDst"] = dst;
        root["gSpecRoughOut"] = srOut;
        root["gNormWOut"] = nOut;
        root["gWetnessOut"] = rd.getTexture("wetness") ?? dummyTexture(this.device, "out:wetness");
        this.cp.execute(ctx, W, H);
    }
}

registerRenderPass("RainWetness", (device, props) => new RainWetness(device, props), {
    desc: "Wet surfaces, puddles and puddle ripples on the lit image + G-buffer (darkening, gloss, flat water normals for SSR)",
});
