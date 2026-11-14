/**
 * SelectiveBloom: bloom that only selected objects emit. The selection is a list of ids matched against the raw id of
 * StyleAovs' `id` output (g channel; instance ids with idSource = "instance"), optionally multiplied by a `mask` input
 * (r channel in [0, 1], e.g. a SubsetRaster mask or a per-instance custom AOV); `invert` swaps selected and unselected.
 *
 * 1. Prefilter (half resolution): 2x2 box downsample of src x selection, soft-knee threshold:
 *    c x max(soft, max(c) - threshold) / max(c), soft = clamp(max(c) - threshold + knee, 0, 2 knee)^2 / (4 knee).
 * 2. Separable Gaussian at half resolution, sigma = radius x half height (pixels); 65 taps per axis with stride
 *    max(3 sigma / 32, 1) and linear lookups between texels (clamp to edge).
 * 3. Composite (full resolution): dst.rgb = src.rgb + intensity x tint x bilinear(glow); dst.a = src.a.
 *    Optional `glow` output = intensity x tint x bilinear(glow).
 * Only visible pixels of the selected objects glow (the selection comes from the visible-surface ids); the glow itself
 * spreads over neighbouring pixels, including other objects in front.
 *
 * Stateless (no history). Identity: intensity 0, or an empty selection (no ids, no mask, not inverted), copies src bit
 * for bit.
 *
 * The idea of masking the bloom input by an object selection follows pmndrs/postprocessing's SelectiveBloomEffect
 * (https://github.com/pmndrs/postprocessing, zlib licence, Copyright (c) 2015 Raoul van Rüschen); the code is written for
 * this pass.
 *
 * Inputs : src, id? (StyleAovs.id), mask? (r = selection weight)
 * Outputs: dst (RGBA32Float), glow? (RGBA32Float)
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

const kShaderFile = "RenderPasses/Focus/SelectiveBloom.cs.slang";
const kEntries = ["prefilter", "blurH", "blurV", "composite"] as const;
type Entry = (typeof kEntries)[number];
export const kSelectiveBloomMaxIds = 32;

const kDefaults = {
    ids: [] as number[],
    invert: false,
    intensity: 1,
    threshold: 0,
    knee: 0,
    radius: 0.02,
    tint: [1, 1, 1] as number[] | string,
};
export type SelectiveBloomParams = typeof kDefaults;

export class SelectiveBloom extends RenderPass {
    private p: SelectiveBloomParams = JSON.parse(JSON.stringify(kDefaults));
    private passes = new Map<Entry, ComputePass>();
    private half: [Texture, Texture] | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(kDefaults) as (keyof SelectiveBloomParams)[]) if (props.has(k)) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        const ids = (this.p.ids ?? []).map(Number);
        if (ids.length > kSelectiveBloomMaxIds) throw new Error(`SelectiveBloom: at most ${kSelectiveBloomMaxIds} ids (use the mask input for larger selections)`);
        if (ids.some((v) => !Number.isInteger(v) || v < 0)) throw new Error("SelectiveBloom: ids must be non-negative integers");
        this.p.ids = ids;
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Intensity", this.p.intensity, 0, 8, 0.01, (v) => (this.p.intensity = v));
        ui.slider("Threshold", this.p.threshold, 0, 20, 0.01, (v) => (this.p.threshold = v));
        ui.slider("Radius", this.p.radius, 0.001, 0.2, 0.001, (v) => (this.p.radius = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "HDR colour");
        addTexInput(r, "id", "StyleAovs.id (raw id in g, alpha 0 = background)", true);
        addTexInput(r, "mask", "Selection weight in r (multiplies the id selection)", true);
        addColorOutput(r, "dst", "Colour with the selective glow added", w, h);
        addColorOutput(r, "glow", "Glow only (rgb)", w, h, true);
        return r;
    }

    private kernel(e: Entry): ComputePass {
        let k = this.passes.get(e);
        if (!k) this.passes.set(e, (k = ComputePass.create(this.device, { path: kShaderFile, csEntry: e })));
        return k;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const id = rd.getTexture("id"), mask = rd.getTexture("mask"), glowOut = rd.getTexture("glow");
        const W = dst.width, H = dst.height, W2 = Math.max(1, (W + 1) >> 1), H2 = Math.max(1, (H + 1) >> 1);
        if (!this.half || this.half[0].width !== W2 || this.half[0].height !== H2) {
            const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            this.half = [0, 1].map(() => this.device.createTexture2D(W2, H2, ResourceFormat.RGBA32Float, 1, 1, undefined, flags)) as [Texture, Texture];
        }
        const [A, B] = this.half;
        const p = this.p;
        const ids = p.ids;
        const idsActive = !!id && ids.length > 0;
        const anySelected = p.invert ? !!id || !!mask : idsActive || (!!mask && !id);
        const intensity = anySelected ? Number(p.intensity) : 0;
        const packed = Array.from({ length: 8 }, (_, i) => [0, 1, 2, 3].map((k) => ids[i * 4 + k] ?? 0));
        const consts: Record<string, unknown> = {
            gFrameDim: [W, H], gHalfDim: [W2, H2], gIds: packed, gIdCount: ids.length, gHasId: id ? 1 : 0, gHasMask: mask ? 1 : 0,
            gInvert: p.invert ? 1 : 0, gThreshold: Number(p.threshold), gKnee: Math.max(Number(p.knee), 0), gIntensity: intensity,
            gSigma: Math.max(Number(p.radius) * H2, 0.5), gTint: parseColour(p.tint), gHasGlowOut: glowOut ? 1 : 0,
        };
        const d = (slot: string) => dummyTexture(this.device, slot);
        const run = (e: Entry, x: number, y: number, inT: Texture, out: Texture, dstT: Texture, glowT: Texture) => {
            const k = this.kernel(e);
            const root = k.getRootVar();
            const cb = root["CB"]!;
            for (const [name, v] of Object.entries(consts)) cb[name] = v as never;
            root["gSrc"] = src;
            root["gId"] = id ?? d("in:sb1");
            root["gMask"] = mask ?? d("in:sb2");
            root["gIn"] = inT;
            root["gOut"] = out;
            root["gDst"] = dstT;
            root["gGlowOut"] = glowT;
            k.execute(ctx, x, y);
        };
        if (intensity !== 0) {
            run("prefilter", W2, H2, d("in:sb3"), A, d("out:sb1"), d("out:sb2"));
            run("blurH", W2, H2, A, B, d("out:sb1"), d("out:sb2"));
            run("blurV", W2, H2, B, A, d("out:sb1"), d("out:sb2"));
        }
        run("composite", W, H, intensity !== 0 ? A : d("in:sb3"), d("out:sb3"), dst, glowOut ?? d("out:sb2"));
    }
}

registerRenderPass("SelectiveBloom", (device, props) => new SelectiveBloom(device, props), {
    desc: "Bloom restricted to selected objects (id list from StyleAovs.id and/or a mask input)",
});
