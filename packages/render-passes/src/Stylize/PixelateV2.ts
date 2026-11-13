/**
 * PixelateV2: pixelation on a block grid with selectable downsampling and an optional low-resolution outline.
 *  - mode "avg": mean of the block (box filter); "nearest": the block's centre pixel (no colour mixing, crisp pixel-art
 *    edges); "majority": the mean of the pixels in the block's most frequent colour bin (keeps flat colours, drops
 *    minority detail);
 *  - offset [x, y] (0 <= offset < block) shifts the grid, e.g. to keep it fixed to a snapped camera;
 *  - outline (0-1) darkens grid cells on depth/normal discontinuities of the low-resolution grid (1-cell outlines as in
 *    real-time 3D pixel-art renderers); needs posW (and normW for creases).
 * Nearest-neighbour upsampling to full resolution. block = 1 with mode avg and outline 0 is the identity.
 *
 * Inputs : src, posW (optional), normW (optional).
 * Params : block (1-32), mode {avg|nearest|majority}, offset [x, y], outline, outlineThreshold (relative depth),
 *          normalThreshold (1 - n.n').
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type PropertyValue,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, cameraData, dummyTexture, enumIndex } from "./StyleCommon.js";

const kShaderFile = "RenderPasses/Stylize/PixelateV2.cs.slang";
const kModes = ["avg", "nearest", "majority"] as const;
const kDefaults = { block: 4, mode: "nearest" as string, offset: [0, 0] as number[], outline: 0, outlineThreshold: 0.05, normalThreshold: 0.3 };

export class PixelateV2 extends RenderPass {
    private p = { ...kDefaults };
    private passes: Record<string, ComputePass> = {};
    private tex: Record<string, Texture> = {};

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof typeof kDefaults)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k]);
        enumIndex("PixelateV2", "mode", this.p.mode, kModes);
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)) as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Block (px)", this.p.block, 1, 32, 1, (v) => (this.p.block = v));
        ui.slider("Outline", this.p.outline, 0, 1, 0.01, (v) => (this.p.outline = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "Colour");
        addTexInput(r, "posW", "World-space position (optional; low-res outline)", true);
        addTexInput(r, "normW", "World-space normal (optional; low-res outline creases)", true);
        addColorOutput(r, "dst", "Pixelated colour", w, h);
        return r;
    }

    private texture(name: string, w: number, h: number): Texture {
        let t = this.tex[name];
        if (!t || t.width !== w || t.height !== h) {
            t = this.tex[name] = new Texture(this.device, {
                type: ResourceType.Texture2D, width: w, height: h, format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource, name: `PixelateV2::${name}`,
            });
        }
        return t;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const posW = rd.getTexture("posW"), normW = rd.getTexture("normW");
        const W = dst.width, H = dst.height;
        const B = Math.min(Math.max(Math.round(this.p.block), 1), 32);
        const off = [0, 1].map((i) => ((Math.round(Number(this.p.offset[i] ?? 0)) % B) + B) % B);
        const lw = Math.ceil((W + off[0]!) / B), lh = Math.ceil((H + off[1]!) / B);
        const lowA = this.texture("low", lw, lh), geo = this.texture("geo", lw, lh), lowB = this.texture("lowOut", lw, lh);
        const cam = this.scene ? cameraData(this.scene) : null;
        const dIn = dummyTexture(this.device);
        const stages: [string, number, number, Record<string, Texture>][] = [
            ["lowMain", lw, lh, { gLowIn: dIn, gLowGeoIn: dIn, gLow: lowA, gLowGeo: geo, gDst: dummyTexture(this.device, "out:PixelateV2.dst") }],
            ["edgeMain", lw, lh, { gLowIn: lowA, gLowGeoIn: geo, gLow: lowB, gLowGeo: dummyTexture(this.device, "out:PixelateV2.geo"), gDst: dummyTexture(this.device, "out:PixelateV2.dst") }],
            ["upMain", W, H, { gLowIn: lowB, gLowGeoIn: dIn, gLow: dummyTexture(this.device, "out:PixelateV2.low"), gLowGeo: dummyTexture(this.device, "out:PixelateV2.geo"), gDst: dst }],
        ];
        for (const [entry, dw, dh, binds] of stages) {
            const pass = (this.passes[entry] ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: entry }));
            const root = pass.getRootVar();
            const cb = root["CB"]!;
            cb["gFrameDim"] = [W, H];
            cb["gLowDim"] = [lw, lh];
            cb["gOffset"] = off;
            cb["gBlock"] = B;
            cb["gMode"] = enumIndex("PixelateV2", "mode", this.p.mode, kModes);
            cb["gOutline"] = posW ? this.p.outline : 0;
            cb["gOutlineThreshold"] = this.p.outlineThreshold;
            cb["gNormalThreshold"] = this.p.normalThreshold;
            cb["gCamPos"] = cam?.pos ?? [0, 0, 0];
            cb["gCamFwd"] = cam?.fwd ?? [0, 0, -1];
            cb["gHasPosW"] = posW && cam ? 1 : 0;
            cb["gHasNormW"] = normW ? 1 : 0;
            root["gSrc"] = src;
            root["gPosW"] = posW ?? dIn;
            root["gNormW"] = normW ?? dIn;
            for (const [k, t] of Object.entries(binds)) root[k] = t;
            pass.execute(ctx, dw, dh);
        }
    }
}

registerRenderPass("PixelateV2", (device, props) => new PixelateV2(device, props), {
    desc: "Block pixelation (average / nearest / majority colour) with grid offset and optional low-res outline",
});
