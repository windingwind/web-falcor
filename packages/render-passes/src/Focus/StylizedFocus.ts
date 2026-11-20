/**
 * StylizedFocus: region-of-interest abstraction. A focus weight f in [0, 1] (1 = in focus) is computed per pixel and the
 * out-of-focus amount a = strength x (1 - f) drives two abstraction channels: blur and desaturation.
 *
 * Focus weight by `mode`:
 *  - depth  : f = 1 - smoothstep(focusRange, focusRange + falloff, | |posW - camera| - focusDistance |)  (world units);
 *  - point  : f = 1 - smoothstep(focusRange, focusRange + falloff, |posW - focusPoint|)                (world units);
 *  - screen : f = 1 - smoothstep(focusRange, focusRange + falloff, |(uv - screenCenter) x (aspect, 1)|) (frame heights);
 *  - ids    : f = 1 on visible pixels of the listed ids (StyleAovs.id raw id in g, at most 32), else 0;
 *  - mask   : f = saturate(mask.r).
 *  Background pixels (posW.a = 0) get `backgroundFocus` in the depth and point modes.
 *
 * Abstraction (only where a > 0; elsewhere dst = src bit for bit):
 *  - blur: two separable Gaussians of src at sigma2 = blurRadius x frame height and sigma1 = sigma2 / 3 (65 taps per
 *    axis, stride max(3 sigma / 32, 1), linear lookups, clamp to edge); t = saturate(a):
 *    c = t < 0.5 ? lerp(src, blur1, 2t) : lerp(blur1, blur2, 2t - 1);
 *  - desaturation: c = lerp(c, luma(c), saturate(a x desaturate)), luma = Rec. 709 weights.
 *  dst.a = src.a. Optional `focus` output = f (rgb), 1 (a).
 *
 * Stateless (no history). Identity: strength 0, or f = 1 everywhere, copies src bit for bit.
 *
 * After Cole, DeCarlo, Finkelstein, Kin, Morley, Santella 2006, "Directing Gaze in 3D Models with Stylized Focus"
 * (EGSR): emphasis falls off with the distance to a focus and controls how much each region is abstracted. Written for
 * this pass from the paper's description.
 *
 * Inputs : src, posW? (depth/point modes), id? (ids mode), mask? (mask mode)
 * Outputs: dst (RGBA32Float), focus? (RGBA32Float)
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
import { addColorOutput, addTexInput, dummyTexture, enumIndex } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Focus/StylizedFocus.cs.slang";
/** Compute entry points declared by the companion Slang shader. */
const kEntries = ["blurH", "blurV", "composite"] as const;
type Entry = (typeof kEntries)[number];
/** Ordered option names; preserve this order for numeric property compatibility. */
export const kStylizedFocusModes = ["depth", "point", "screen", "ids", "mask"] as const;
export const kStylizedFocusMaxIds = 32;

/** Creation defaults for the pass parameters documented in this file header. */
const kDefaults = {
    mode: "depth" as string,
    strength: 1,
    focusDistance: 10,
    focusPoint: [0, 0, 0] as number[],
    screenCenter: [0.5, 0.5] as number[],
    focusRange: 1,
    falloff: 5,
    backgroundFocus: 0,
    ids: [] as number[],
    blurRadius: 0.01,
    desaturate: 0.8,
};
/** Creation properties for StylizedFocus, including its documented defaults. */
export type StylizedFocusParams = typeof kDefaults;

/** Blurs and desaturates regions outside a selected depth, point, screen or identity focus. */
export class StylizedFocus extends RenderPass {
    private mParameters: StylizedFocusParams = JSON.parse(JSON.stringify(kDefaults));
    private mPasses: Map<Entry, ComputePass> = new Map<Entry, ComputePass>();
    private mTextures: [Texture, Texture, Texture] | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(kDefaults) as (keyof StylizedFocusParams)[])
            if (props.has(k)) (this.mParameters as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        enumIndex("StylizedFocus", "mode", this.mParameters.mode, kStylizedFocusModes);
        const ids = (this.mParameters.ids ?? []).map(Number);
        if (ids.length > kStylizedFocusMaxIds)
            throw new Error(`StylizedFocus: at most ${kStylizedFocusMaxIds} ids (use mode "mask" for larger selections)`);
        if (ids.some((v) => !Number.isInteger(v) || v < 0)) throw new Error("StylizedFocus: ids must be non-negative integers");
        this.mParameters.ids = ids;
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.mParameters)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Strength", this.mParameters.strength, 0, 1, 0.01, (v) => (this.mParameters.strength = v));
        ui.slider("Focus distance", this.mParameters.focusDistance, 0, 1000, 0.1, (v) => (this.mParameters.focusDistance = v));
        ui.slider("Focus range", this.mParameters.focusRange, 0, 100, 0.01, (v) => (this.mParameters.focusRange = v));
        ui.slider("Falloff", this.mParameters.falloff, 0.001, 100, 0.01, (v) => (this.mParameters.falloff = v));
        ui.slider("Blur radius", this.mParameters.blurRadius, 0, 0.1, 0.001, (v) => (this.mParameters.blurRadius = v));
        ui.slider("Desaturate", this.mParameters.desaturate, 0, 1, 0.01, (v) => (this.mParameters.desaturate = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "Colour");
        addTexInput(r, "posW", "World position (alpha 0 = background); depth and point modes", true);
        addTexInput(r, "id", "StyleAovs.id (raw id in g); ids mode", true);
        addTexInput(r, "mask", "Focus weight in r; mask mode", true);
        addColorOutput(r, "dst", "Colour with the out-of-focus region abstracted", w, h);
        addColorOutput(r, "focus", "Focus weight (1 = in focus)", w, h, true);
        return r;
    }

    private kernel(e: Entry): ComputePass {
        let k = this.mPasses.get(e);
        if (!k) this.mPasses.set(e, (k = ComputePass.create(this.device, { path: kShaderFile, csEntry: e })));
        return k;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src");
        const dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const posW = rd.getTexture("posW");
        const id = rd.getTexture("id");
        const mask = rd.getTexture("mask");
        const focusOut = rd.getTexture("focus");
        const p = this.mParameters;
        const mode = enumIndex("StylizedFocus", "mode", p.mode, kStylizedFocusModes);
        if (mode <= 1 && !posW) throw new Error(`StylizedFocus: mode ${p.mode} needs the posW input`);
        if (mode === 3 && !id) throw new Error("StylizedFocus: mode ids needs the id input");
        const W = dst.width;
        const H = dst.height;
        if (!this.mTextures || this.mTextures[0].width !== W || this.mTextures[0].height !== H) {
            const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            this.mTextures = [0, 1, 2].map(() => this.device.createTexture2D(W, H, ResourceFormat.RGBA32Float, 1, 1, undefined, flags)) as [
                Texture,
                Texture,
                Texture,
            ];
        }
        const [T, B1, B2] = this.mTextures;
        const cam = this.scene ? this.scene.camera.getPosition() : null;
        const strength = Number(p.strength);
        const blurOn = strength > 0 && Number(p.blurRadius) > 0;
        const sigma2 = Number(p.blurRadius) * H;
        const ids = p.ids;
        const consts: Record<string, unknown> = {
            gFrameDim: [W, H],
            gMode: mode,
            gHasMask: mask ? 1 : 0,
            gCamPos: cam ? [cam.x, cam.y, cam.z] : [0, 0, 0],
            gFocusDistance: Number(p.focusDistance),
            gFocusPoint: p.focusPoint.map(Number),
            gFocusRange: Math.max(Number(p.focusRange), 0),
            gScreenCenter: p.screenCenter.map(Number),
            gFalloff: Math.max(Number(p.falloff), 1e-6),
            gBackgroundFocus: Number(p.backgroundFocus),
            gIds: Array.from({ length: 8 }, (_, i) => [0, 1, 2, 3].map((k) => ids[i * 4 + k] ?? 0)),
            gIdCount: ids.length,
            gStrength: strength,
            gDesaturate: Number(p.desaturate),
            gHasBlur: blurOn ? 1 : 0,
            gSigma: 1,
            gHasFocusOut: focusOut ? 1 : 0,
        };
        const d = (slot: string) => dummyTexture(this.device, slot);
        const run = (e: Entry, sigma: number, inT: Texture, out: Texture) => {
            const k = this.kernel(e);
            const root = k.getRootVar();
            const cb = root["CB"]!;
            for (const [name, v] of Object.entries(consts)) cb[name] = v as never;
            cb["gSigma"] = sigma;
            root["gSrc"] = src;
            root["gPosW"] = posW ?? d("in:sf1");
            root["gId"] = id ?? d("in:sf2");
            root["gMask"] = mask ?? d("in:sf3");
            root["gIn"] = inT;
            root["gBlur1"] = e === "composite" && blurOn ? B1 : d("in:sf4");
            root["gBlur2"] = e === "composite" && blurOn ? B2 : d("in:sf5");
            root["gOut"] = out;
            root["gDst"] = e === "composite" ? dst : d("out:sf1");
            root["gFocusOut"] = e === "composite" && focusOut ? focusOut : d("out:sf2");
            k.execute(ctx, W, H);
        };
        if (blurOn) {
            for (const [sigma, outT] of [
                [sigma2 / 3, B1],
                [sigma2, B2],
            ] as const) {
                const s = Math.max(sigma, 0.25);
                run("blurH", s, src, T);
                run("blurV", s, T, outT);
            }
        }
        run("composite", 1, d("in:sf6"), d("out:sf3"));
    }
}

registerRenderPass("StylizedFocus", (device, props) => new StylizedFocus(device, props), {
    desc: "Stylized focus: keep a focus region (depth, point, screen, ids or mask) sharp, blur and desaturate the rest",
});
