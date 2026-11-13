/**
 * FXAA: fast approximate anti-aliasing, the FXAA 3.11 "quality" algorithm (T. Lottes, NVIDIA 2011): local-contrast
 * edge detection, edge orientation from the 3x3 luma neighbourhood, an end-of-edge search along the edge (up to
 * `iterations` steps with the 3.11 step schedule) and sub-pixel aliasing removal.
 * Native Falcor 8 has no FXAA pass (only TAA/DLSS); this is a web-falcor addition.
 *
 * Inputs : src (display-referred colour, i.e. after tone mapping; luma = Rec.601 weights of the clamped rgb).
 * Outputs: dst (RGBA32Float, same size; alpha passed through).
 * Params : edgeThresholdMin (0.0312), edgeThresholdMax (0.125), subpixelQuality (0.75), iterations (12).
 * Filtering is done with manual bilinear Load()s, so any float input format works (incl. 32-bit float).
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

const kShaderFile = "RenderPasses/FXAA/FXAA.cs.slang";

export class FXAA extends RenderPass {
    private p = { edgeThresholdMin: 0.0312, edgeThresholdMax: 0.125, subpixelQuality: 0.75, iterations: 12, enabled: true };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.p) as (keyof typeof this.p)[]) (this.p as Record<string, unknown>)[k] = props.get(k, this.p[k]);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.checkbox("Enabled", this.p.enabled, (v) => (this.p.enabled = v));
        ui.slider("Edge threshold min", this.p.edgeThresholdMin, 0, 0.1, 0.0001, (v) => (this.p.edgeThresholdMin = v));
        ui.slider("Edge threshold max", this.p.edgeThresholdMax, 0, 0.5, 0.001, (v) => (this.p.edgeThresholdMax = v));
        ui.slider("Subpixel quality", this.p.subpixelQuality, 0, 1, 0.01, (v) => (this.p.subpixelQuality = v));
        ui.slider("Search iterations", this.p.iterations, 1, 12, 1, (v) => (this.p.iterations = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("src", "Tone-mapped colour").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Anti-aliased colour")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src");
        const dst = rd.getTexture("dst");
        if (!src || !dst) return;
        if (!this.pass) this.pass = ComputePass.create(this.device, { path: kShaderFile });
        const cb = this.pass.getRootVar()["CB"]!;
        cb["gFrameDim"] = [dst.width, dst.height];
        cb["gEdgeThresholdMin"] = this.p.edgeThresholdMin;
        cb["gEdgeThresholdMax"] = this.p.edgeThresholdMax;
        cb["gSubpixelQuality"] = this.p.subpixelQuality;
        cb["gIterations"] = Math.min(12, Math.max(1, Math.round(this.p.iterations)));
        cb["gEnabled"] = this.p.enabled ? 1 : 0;
        const root = this.pass.getRootVar();
        root["gSrc"] = src;
        root["gDst"] = dst;
        this.pass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("FXAA", (device, props) => new FXAA(device, props), { desc: "FXAA 3.11 quality anti-aliasing" });
