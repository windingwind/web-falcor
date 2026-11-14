/**
 * SSR: screen-space reflections traced through a Hi-Z pyramid, with stochastic GGX rays and temporal accumulation.
 *
 * One render-graph node, four dispatches per frame (SSR.cs.slang):
 *  trace -> resolve -> temporal -> composite.
 * Each pixel with roughness <= maxRoughness traces one ray: the mirror direction, or (quality "stochastic", roughness
 * above mirrorRoughness) a GGX visible-normal sample. Rays march through the min-depth pyramid of HiZBuild (r channel)
 * and fetch the lit colour at the hit; misses and rays leaving the screen contribute nothing (the input colour keeps its
 * own environment reflection). The resolve pass reuses neighbours for rough pixels (radius = resolveRadius scaled by
 * roughness / maxRoughness), so quality "mirror" + resolveRadius gives a blurred mirror without noise. The temporal
 * pass blends with the reprojected history (motion vectors, neighbourhood clamp).
 *
 * History: the temporal state lives in a PingPongTextures pair driven by a HistoryTracker (Utils/History.ts). It is
 * reset on the first execute, resetHistory() / resetGraphHistory(), a scene or camera change, a resize,
 * Camera.resetHistory() and a primed camera cut; the random sequence restarts with it (its index is the number of frames
 * since the reset), so a shot rendered after a reset is bit-identical to the same shot on a fresh context. With
 * temporal = false the pass keeps no state that affects its output (the random sequence index is then always 0).
 *
 * Inputs : color (lit colour), depth (GBufferRaster depth), normW, specRough (F0 rgb + roughness a), hiz (HiZBuild.hiz
 *          built from the same depth), mvec (optional; needed for temporal reprojection under camera motion).
 * Outputs: dst = color + intensity * reflection; reflection (optional: rgb = reflected radiance x DFG x confidence,
 *          a = confidence); hit (optional debug: hit uv, hit depth, confidence of this frame's ray).
 * Params : intensity (0 = dst is color, bit for bit), quality (mirror | stochastic), maxRoughness, mirrorRoughness,
 *          maxSteps, thickness (relative to view depth), resolveRadius, temporal, temporalAlpha, borderFade, seed.
 */

import {
    ComputePass,
    FieldFlags,
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
import { HistoryTracker, PingPongTextures } from "../Utils/History.js";

const kShaderFile = "RenderPasses/SSR/SSR.cs.slang";
const kQualities = ["mirror", "stochastic"] as const;

const kDefaults = {
    intensity: 1,
    quality: "stochastic" as string,
    maxRoughness: 0.6,
    mirrorRoughness: 0.02,
    maxSteps: 96,
    thickness: 0.03,
    resolveRadius: 2,
    temporal: true,
    temporalAlpha: 0.9,
    borderFade: 0.05,
    seed: 0,
};

export class SSR extends RenderPass {
    private p = { ...kDefaults };
    private passes: Record<string, ComputePass> = {};
    private readonly history = new HistoryTracker();
    private readonly pp: PingPongTextures;
    private traced: Texture | null = null;
    private resolved: Texture | null = null;
    private scratch: Record<string, Texture> = {};
    private dummy: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof typeof kDefaults)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k]);
        if (!(kQualities as readonly string[]).includes(this.p.quality)) throw new Error(`SSR: quality must be one of ${kQualities.join("|")} (got ${this.p.quality})`);
        this.pp = new PingPongTextures(device, ResourceFormat.RGBA32Float, "SSR::history");
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Intensity", this.p.intensity, 0, 2, 0.01, (v) => (this.p.intensity = v));
        ui.dropdown("Quality", [...kQualities], this.p.quality, (v) => (this.p.quality = v));
        ui.slider("Max roughness", this.p.maxRoughness, 0, 1, 0.01, (v) => (this.p.maxRoughness = v));
        ui.slider("Max steps", this.p.maxSteps, 8, 256, 1, (v) => (this.p.maxSteps = Math.round(v)));
        ui.slider("Thickness", this.p.thickness, 0.001, 0.2, 0.001, (v) => (this.p.thickness = v));
        ui.slider("Resolve radius", this.p.resolveRadius, 0, 6, 1, (v) => (this.p.resolveRadius = Math.round(v)));
        ui.checkbox("Temporal", this.p.temporal, (v) => (this.p.temporal = v));
        ui.slider("Temporal weight", this.p.temporalAlpha, 0, 0.98, 0.01, (v) => (this.p.temporalAlpha = v));
    }

    /** Drops the temporal history: the next frame is computed as on a fresh context. */
    resetHistory(): void {
        this.history.reset();
    }

    /** Frames since the last history reset. */
    get frameIndex(): number {
        return this.history.frameIndex;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        const input = (name: string, desc: string, optional = false) => {
            const f = r.addInput(name, desc).bindFlags(ResourceBindFlags.ShaderResource);
            if (optional) f.flags(FieldFlags.Optional);
        };
        input("color", "Lit colour");
        input("depth", "Depth (standard NDC depth, r)");
        input("normW", "World-space shading normal");
        input("specRough", "Specular reflectance (rgb) and roughness (a)");
        // Any mip count (0 = not constrained, so the field merges with HiZBuild's mip chain).
        r.addInput("hiz", "HiZBuild.hiz of the same depth").texture2D(0, 0, 1, 0).bindFlags(ResourceBindFlags.ShaderResource);
        input("mvec", "Motion vectors (UV, previous minus current)", true);
        const out = (name: string, desc: string, optional: boolean) => {
            const f = r.addOutput(name, desc).texture2D(w, h).format(ResourceFormat.RGBA32Float)
                .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
            if (optional) f.flags(FieldFlags.Optional);
        };
        out("dst", "Colour with reflections", false);
        out("reflection", "Reflected radiance x DFG x confidence (rgb), confidence (a)", true);
        out("hit", "Debug: hit uv (rg), hit depth (b), confidence (a)", true);
        return r;
    }

    private tex(name: string, w: number, h: number): Texture {
        let t = this.scratch[name];
        if (!t || t.width !== w || t.height !== h) {
            t = this.device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, undefined,
                ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
            this.scratch[name] = t;
        }
        return t;
    }

    private pass(entry: string): ComputePass {
        return (this.passes[entry] ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: entry }));
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        const color = rd.getTexture("color"), depth = rd.getTexture("depth"), normW = rd.getTexture("normW");
        const specRough = rd.getTexture("specRough"), hiz = rd.getTexture("hiz");
        if (!dst || !color || !depth || !normW || !specRough || !hiz) return;
        if (!this.scene) throw new Error("SSR: requires a scene (camera)");
        const W = dst.width, H = dst.height;
        const temporal = this.p.temporal && this.p.temporalAlpha > 0;
        // The tracker runs every frame; its key (size, temporal on/off) resets the history when the state layout changes.
        const frame = this.history.begin(this.scene, `${W}x${H}:${temporal}`);
        const mvec = rd.getTexture("mvec");
        this.dummy ??= this.device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(4), ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
        this.traced = this.tex("traced", W, H);
        this.resolved = this.tex("resolved", W, H);
        const hitOut = rd.getTexture("hit") ?? this.tex("hit", W, H);
        const reflOut = rd.getTexture("reflection") ?? this.tex("reflection", W, H);

        const cam = this.scene.camera;
        const data = cam.getData();
        const p = cam.getPosition(), t = cam.getTarget();
        const f = [t.x - p.x, t.y - p.y, t.z - p.z];
        const fl = Math.hypot(f[0]!, f[1]!, f[2]!) || 1;
        const seedFrame = (temporal ? frame.frameIndex : 0) + Math.max(0, Math.round(this.p.seed)) * 7919;
        const setCB = (cp: ComputePass) => {
            const root = cp.getRootVar();
            const cb = root["CB"]!;
            cb["gFrameDim"] = [W, H];
            cb["gMipCount"] = hiz.mipCount;
            cb["gMaxSteps"] = Math.max(1, Math.round(this.p.maxSteps));
            cb["gViewProj"] = data.viewProjMatNoJitter;
            cb["gInvViewProj"] = data.invViewProj;
            cb["gCamPos"] = [p.x, p.y, p.z];
            cb["gThickness"] = this.p.thickness;
            cb["gCamFwd"] = f.map((x) => x / fl);
            cb["gNear"] = cam.getNearPlane();
            cb["gMaxRoughness"] = this.p.maxRoughness;
            cb["gMirrorRoughness"] = this.p.mirrorRoughness;
            cb["gStochastic"] = this.p.quality === "stochastic" ? 1 : 0;
            cb["gFrameSeed"] = seedFrame >>> 0;
            cb["gResolveRadius"] = Math.max(0, Math.round(this.p.resolveRadius));
            cb["gTemporalAlpha"] = Math.min(Math.max(this.p.temporalAlpha, 0), 0.999);
            cb["gHistoryValid"] = frame.frameIndex > 0 ? 1 : 0;
            cb["gIntensity"] = this.p.intensity;
            cb["gBorderFade"] = this.p.borderFade;
            return root;
        };

        {
            const cp = this.pass("trace");
            const root = setCB(cp);
            root["gColor"] = color;
            root["gDepth"] = depth;
            root["gNormW"] = normW;
            root["gSpecRough"] = specRough;
            root["gHiZ"] = hiz;
            root["gTraceOut"] = this.traced;
            root["gHitOut"] = hitOut;
            cp.execute(ctx, W, H);
        }
        {
            const cp = this.pass("resolve");
            const root = setCB(cp);
            root["gTraced"] = this.traced;
            root["gDepth"] = depth;
            root["gNormW"] = normW;
            root["gSpecRough"] = specRough;
            root["gResolveOut"] = this.resolved;
            cp.execute(ctx, W, H);
        }
        let refl = this.resolved;
        if (temporal) {
            const { read, write } = this.pp.get(ctx, W, H, frame);
            const cp = this.pass("temporal");
            const root = setCB(cp);
            root["gResolved"] = this.resolved;
            root["gMvec"] = mvec ?? this.dummy;
            root["gHistory"] = read;
            root["gTemporalOut"] = write;
            cp.execute(ctx, W, H);
            refl = write;
        }
        {
            const cp = this.pass("composite");
            const root = setCB(cp);
            root["gColor"] = color;
            root["gRefl"] = refl;
            root["gDepth"] = depth;
            root["gNormW"] = normW;
            root["gSpecRough"] = specRough;
            root["gDst"] = dst;
            root["gReflectionOut"] = reflOut;
            cp.execute(ctx, W, H);
        }
    }
}

registerRenderPass("SSR", (device, props) => new SSR(device, props), {
    desc: "Screen-space reflections: Hi-Z trace, stochastic GGX rays, neighbour resolve, temporal accumulation (history resets on cuts)",
});
