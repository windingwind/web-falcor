/**
 * SSAO: screen-space ambient occlusion with the GTAO estimator (Jimenez et al. 2016, "Practical Real-Time
 * Strategies for Accurate Indirect Occlusion"; horizon/slice integration after Intel's XeGTAO).
 * Native Falcor 8 has no SSAO pass (Falcor 4's SSAO was removed upstream), so this is a web-falcor addition.
 *
 * Inputs : posW (GBuffer posW, w = 0 marks background), normW (GBuffer normW, world space).
 * Outputs: ao (RGBA32Float, rgb = cosine-weighted visibility in [0,1], a = 1; background = 1).
 * The estimate is the cosine-weighted visibility (1/pi) * int V(w) cos(w) dw restricted to occluders within `radius`
 * (world units, linear falloff over the last `falloff` fraction), the same quantity a ray-traced AO with that ray
 * length converges to, up to GTAO's height-field assumption (no thickness behind the depth buffer).
 * Two dispatches: the slice integration (per-pixel rotated slices, interleaved-gradient noise) and a 5x5
 * depth/normal-aware spatial filter (`blur`).
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
    RuntimeError,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/SSAO/SSAO.cs.slang";

export class SSAO extends RenderPass {
    private p = { radius: 1.0, falloff: 0.4, sliceCount: 4, stepsPerSlice: 8, maxPixelRadius: 128, power: 1.0, noise: true, blur: true };
    private aoPass: ComputePass | null = null;
    private blurPass: ComputePass | null = null;
    private tmp: Texture | null = null;
    private frame = 0;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.p) as (keyof typeof this.p)[]) (this.p as Record<string, unknown>)[k] = props.get(k, this.p[k]);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Radius (world)", this.p.radius, 0.01, 50, 0.01, (v) => (this.p.radius = v));
        ui.slider("Falloff fraction", this.p.falloff, 0, 1, 0.01, (v) => (this.p.falloff = v));
        ui.slider("Slices", this.p.sliceCount, 1, 16, 1, (v) => (this.p.sliceCount = v));
        ui.slider("Steps per slice side", this.p.stepsPerSlice, 1, 32, 1, (v) => (this.p.stepsPerSlice = v));
        ui.slider("Max radius (px)", this.p.maxPixelRadius, 4, 512, 1, (v) => (this.p.maxPixelRadius = v));
        ui.slider("Power", this.p.power, 0.1, 4, 0.01, (v) => (this.p.power = v));
        ui.checkbox("Noise", this.p.noise, (v) => (this.p.noise = v));
        ui.checkbox("Spatial filter", this.p.blur, (v) => (this.p.blur = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("posW", "World-space position (w = 0: background)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("normW", "World-space normal").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("ao", "Ambient occlusion (cosine-weighted visibility)")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const posW = rd.getTexture("posW");
        const normW = rd.getTexture("normW");
        const ao = rd.getTexture("ao");
        if (!posW || !normW || !ao) return;
        if (!this.scene) throw new RuntimeError("SSAO: requires a scene (camera)");
        const W = ao.width, H = ao.height;
        if (!this.tmp || this.tmp.width !== W || this.tmp.height !== H) {
            this.tmp = new Texture(this.device, {
                type: ResourceType.Texture2D, width: W, height: H, format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource, name: "SSAO::raw",
            });
        }
        if (!this.aoPass) {
            this.aoPass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "aoMain" });
            this.blurPass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "blurMain" });
        }
        const cam = this.scene.camera;
        const v = cam.getViewMatrix();
        const row = (r: number) => [v.get(r, 0), v.get(r, 1), v.get(r, 2), v.get(r, 3)];
        const focalPx = H / (2 * Math.tan(0.5 * cam.getFovY()));
        for (const pass of [this.aoPass, this.blurPass!]) {
            const cb = pass.getRootVar()["CB"]!;
            cb["gFrameDim"] = [W, H];
            cb["gViewRow0"] = row(0);
            cb["gViewRow1"] = row(1);
            cb["gViewRow2"] = row(2);
            cb["gFocalPx"] = focalPx;
            cb["gRadius"] = this.p.radius;
            cb["gFalloff"] = this.p.falloff;
            cb["gSliceCount"] = Math.max(1, Math.round(this.p.sliceCount));
            cb["gSteps"] = Math.max(1, Math.round(this.p.stepsPerSlice));
            cb["gMaxPixelRadius"] = this.p.maxPixelRadius;
            cb["gPower"] = this.p.power;
            cb["gNoise"] = this.p.noise ? 1 : 0;
            cb["gFrame"] = this.frame;
        }
        const out = this.p.blur ? this.tmp : ao;
        const a = this.aoPass.getRootVar();
        a["gPosW"] = posW;
        a["gNormW"] = normW;
        a["gOut"] = out;
        a["gRaw"] = posW; // unused by aoMain; bound so the shared layout is complete
        this.aoPass.execute(ctx, W, H);
        if (this.p.blur) {
            const b = this.blurPass!.getRootVar();
            b["gPosW"] = posW;
            b["gNormW"] = normW;
            b["gRaw"] = this.tmp;
            b["gOut"] = ao;
            this.blurPass!.execute(ctx, W, H);
        }
        this.frame++;
    }
}

registerRenderPass("SSAO", (device, props) => new SSAO(device, props), { desc: "Screen-space ambient occlusion (GTAO)" });
