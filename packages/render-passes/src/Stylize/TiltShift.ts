/**
 * TiltShift: miniature-look lens blur. A focus band (a line through (0.5, bandCenter) of the image at `angle` degrees,
 * width bandWidth, both in units of the image height) stays sharp; the Gaussian sigma grows to maxBlur pixels over
 * `feather` outside the band. Separable blur, each pass with the sigma of the output pixel (as in common real-time
 * tilt-shift filters). maxBlur = 0 is the identity.
 *
 * Inputs : src. Params : bandCenter (0-1, y down), bandWidth, feather, maxBlur (px sigma), angle (deg).
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
import { addColorOutput, addTexInput } from "./StyleCommon.js";

const kShaderFile = "RenderPasses/Stylize/TiltShift.cs.slang";
const kDefaults = { bandCenter: 0.55, bandWidth: 0.15, feather: 0.25, maxBlur: 6, angle: 0 };

export class TiltShift extends RenderPass {
    private p = { ...kDefaults };
    private h: ComputePass | null = null;
    private v: ComputePass | null = null;
    private tmp: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof typeof kDefaults)[]) this.p[k] = props.get(k, kDefaults[k]);
    }

    override getProperties(): Properties { return new Properties({ ...this.p } as Record<string, PropertyValue>); }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Band centre", this.p.bandCenter, 0, 1, 0.01, (x) => (this.p.bandCenter = x));
        ui.slider("Max blur (px)", this.p.maxBlur, 0, 32, 0.1, (x) => (this.p.maxBlur = x));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "Colour");
        addColorOutput(r, "dst", "Tilt-shift blurred colour", w, h);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const W = dst.width, H = dst.height;
        if (!this.tmp || this.tmp.width !== W || this.tmp.height !== H) {
            this.tmp = new Texture(this.device, { type: ResourceType.Texture2D, width: W, height: H, format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource, name: "TiltShift::tmp" });
        }
        this.h ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: "hMain" });
        this.v ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: "vMain" });
        const a = (this.p.angle * Math.PI) / 180;
        for (const [pass, input, output] of [[this.h, src, this.tmp], [this.v, this.tmp, dst]] as const) {
            const root = pass.getRootVar();
            const cb = root["CB"]!;
            cb["gFrameDim"] = [W, H];
            cb["gBandCenter"] = this.p.bandCenter;
            cb["gBandWidth"] = this.p.bandWidth;
            cb["gFeather"] = this.p.feather;
            cb["gMaxBlur"] = Math.max(this.p.maxBlur, 0);
            cb["gNormal"] = [-Math.sin(a), Math.cos(a)];
            root["gIn"] = input;
            root["gOut"] = output;
            pass.execute(ctx, W, H);
        }
    }
}

registerRenderPass("TiltShift", (device, props) => new TiltShift(device, props), { desc: "Tilt-shift (miniature) band blur" });
