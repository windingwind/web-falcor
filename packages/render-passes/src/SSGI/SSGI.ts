/**
 * SSGI: screen-space one-bounce diffuse indirect lighting with colour bleeding (visibility bitmask), denoise, temporal
 * accumulation, and an optional "painted" bounce (hue shift, saturation, softness, sky/ground hemisphere ambient).
 *
 * One render-graph node, up to five kinds of dispatch per frame (SSGI.cs.slang):
 *  trace (full or half resolution) -> denoise x N (a-trous, edge-aware) -> accumulate (bilateral upsample + temporal)
 *  -> composite.
 * trace marches `sliceCount` screen-space slices per pixel, `stepCount` samples per side, up to `radius` (world units).
 * Every sample occludes a range of 32 angular sectors of the slice (front surface to front + `thickness`); sectors it
 * occludes first receive the sample's radiance (the `radiance` input, default `color`), weighted by the cosine-weighted
 * solid angle of each sector (GTAO slice measure). The result is the cosine-weighted incident radiance / pi (rgb) and
 * the unoccluded share (a, ambient occlusion); composite adds albedo x that (Lambert) to the colour.
 *
 * Painted mode (mode "painted") is a stylised bounce, continuous in every parameter (no levels, no quantisation, no
 * floor/round/step on luminance or N.L anywhere; unknown band/ramp/level/quantise properties are rejected): the
 * bounce colour is rotated in hue (YIQ, `hueShift` degrees), scaled in chroma (`saturation`), tinted (`tint`), and a
 * sky/ground hemisphere ambient (`skyColor`/`groundColor` by the normal's `upVector` component, `hemiIntensity`,
 * occluded by the AO to the degree `hemiOcclusion`) is added. `softness` widens the denoise (more iterations, looser
 * edge stops) in both modes.
 *
 * History: the temporal state lives in a PingPongTextures pair driven by a HistoryTracker (Utils/History.ts). It is
 * reset on the first execute, resetHistory() / resetGraphHistory(), a scene or camera change, a resize or settings-key
 * change, Camera.resetHistory() and a primed camera cut; the random sequence (slice rotation, step jitter) restarts with
 * it, so a shot rendered after a reset is bit-identical to the same shot on a fresh context. With temporal = false the
 * pass keeps no state that affects its output (the random sequence index is then always 0).
 *
 * Inputs : color (lit colour, composited onto), radiance (optional: the light that bounces, e.g. DeferredLighting.diffuse
 *          or .color; default color), depth (standard NDC depth, r), normW, diffuseOpacity (albedo rgb), hiz (optional:
 *          HiZBuild.hiz of the same depth; far samples read its min-depth mips), mvec (optional; temporal reprojection).
 * Outputs: dst = color + intensity * indirect; indirect (optional: rgb = added light before intensity, a = AO);
 *          gi (optional debug: denoised + accumulated incident radiance / pi (rgb), AO (a)).
 * Params : intensity (0 = dst is color, bit for bit), mode (physical | painted), resolution (full | half), sliceCount,
 *          stepCount, radius, thickness, maxRadiusPx (share of the frame height), hizStartPx, radianceClamp (0 = off),
 *          denoiseIterations, softness, temporal, temporalAlpha, seed, hueShift, saturation, tint, skyColor,
 *          groundColor, hemiIntensity, hemiOcclusion, upVector.
 *
 * Adapted from three.js SSGINode (examples/jsm/tsl/display/SSGINode.js), which follows SSRT3:
 *  - three.js, https://github.com/mrdoob/three.js, MIT licence, Copyright (c) 2010-2026 three.js authors;
 *  - SSRT3, https://github.com/cdrinmatane/SSRT3, MIT licence, Copyright (c) 2024 CDRIN.
 * Method: O. Therrien, Y. Levesque, G. Gilet, "Screen Space Indirect Lighting with Visibility Bitmask", 2023; slice
 * measure after J. Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion" (GTAO), 2016.
 * Permission notice of both MIT sources: Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
 * conditions: The above copyright notice and this permission notice shall be included in all copies or substantial
 * portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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

const kShaderFile = "RenderPasses/SSGI/SSGI.cs.slang";
const kModes = ["physical", "painted"] as const;
const kResolutions = ["full", "half"] as const;

type V3 = [number, number, number];

const kDefaults = {
    intensity: 1,
    mode: "physical" as string,
    resolution: "full" as string,
    sliceCount: 4,
    stepCount: 8,
    radius: 2,
    thickness: 0.5,
    maxRadiusPx: 0.25,
    hizStartPx: 8,
    radianceClamp: 0,
    denoiseIterations: 2,
    softness: 0,
    temporal: true,
    temporalAlpha: 0.9,
    seed: 0,
    hueShift: 20,
    saturation: 1.3,
    tint: [1, 1, 1] as V3,
    skyColor: [0.55, 0.65, 0.85] as V3,
    groundColor: [0.35, 0.28, 0.2] as V3,
    hemiIntensity: 0.25,
    hemiOcclusion: 1,
    upVector: [0, 1, 0] as V3,
};

function vec3(v: unknown, name: string): V3 {
    const o = v as { x?: unknown; y?: unknown; z?: unknown } | null;
    const a: unknown[] | null = Array.isArray(v) ? v : o && typeof o === "object" && "x" in o ? [o.x, o.y, o.z] : null;
    if (!a || a.length !== 3 || a.some((x) => typeof x !== "number" || !Number.isFinite(x))) throw new Error(`SSGI: ${name} must be three numbers`);
    return a as V3;
}

export class SSGI extends RenderPass {
    private p = { ...kDefaults };
    private passes: Record<string, ComputePass> = {};
    private readonly history = new HistoryTracker();
    private readonly pp: PingPongTextures;
    private scratch: Record<string, Texture> = {};
    private dummy: Texture | null = null;
    private dummyRG: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        // Every parameter is continuous: there is no band, ramp, level or quantisation control, and unknown properties
        // (e.g. bands/levels/rampTexture/quantize) are rejected instead of being ignored.
        for (const [k] of props.entries()) {
            if (!(k in kDefaults)) {
                const q = /^(band|ramp|level|quantiz|quantis|posteri|toon|cel)/i.test(k) ? " (SSGI has no banded, ramped or quantised mode)" : "";
                throw new Error(`SSGI: unknown property ${k}${q}`);
            }
        }
        for (const k of Object.keys(kDefaults) as (keyof typeof kDefaults)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k]);
        if (!(kModes as readonly string[]).includes(this.p.mode)) throw new Error(`SSGI: mode must be one of ${kModes.join("|")} (got ${this.p.mode})`);
        if (!(kResolutions as readonly string[]).includes(this.p.resolution)) throw new Error(`SSGI: resolution must be one of ${kResolutions.join("|")} (got ${this.p.resolution})`);
        for (const k of ["tint", "skyColor", "groundColor", "upVector"] as const) this.p[k] = vec3(this.p[k], k);
        const up = this.p.upVector, l = Math.hypot(...up);
        if (!(l > 0)) throw new Error("SSGI: upVector must not be zero");
        this.p.upVector = up.map((x) => x / l) as V3;
        this.pp = new PingPongTextures(device, ResourceFormat.RGBA32Float, "SSGI::history");
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Intensity", this.p.intensity, 0, 4, 0.01, (v) => (this.p.intensity = v));
        ui.dropdown("Mode", [...kModes], this.p.mode, (v) => (this.p.mode = v));
        ui.dropdown("Resolution", [...kResolutions], this.p.resolution, (v) => (this.p.resolution = v));
        ui.slider("Slices", this.p.sliceCount, 1, 16, 1, (v) => (this.p.sliceCount = Math.round(v)));
        ui.slider("Steps", this.p.stepCount, 1, 32, 1, (v) => (this.p.stepCount = Math.round(v)));
        ui.slider("Radius", this.p.radius, 0.01, 50, 0.01, (v) => (this.p.radius = v));
        ui.slider("Thickness", this.p.thickness, 0.001, 10, 0.001, (v) => (this.p.thickness = v));
        ui.slider("Denoise iterations", this.p.denoiseIterations, 0, 5, 1, (v) => (this.p.denoiseIterations = Math.round(v)));
        ui.slider("Softness", this.p.softness, 0, 1, 0.01, (v) => (this.p.softness = v));
        ui.checkbox("Temporal", this.p.temporal, (v) => (this.p.temporal = v));
        ui.slider("Temporal weight", this.p.temporalAlpha, 0, 0.98, 0.01, (v) => (this.p.temporalAlpha = v));
        ui.slider("Hue shift", this.p.hueShift, -180, 180, 1, (v) => (this.p.hueShift = v));
        ui.slider("Saturation", this.p.saturation, 0, 3, 0.01, (v) => (this.p.saturation = v));
        ui.slider("Hemisphere ambient", this.p.hemiIntensity, 0, 2, 0.01, (v) => (this.p.hemiIntensity = v));
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
        input("radiance", "Radiance that bounces (default: color)", true);
        input("depth", "Depth (standard NDC depth, r)");
        input("normW", "World-space shading normal");
        input("diffuseOpacity", "Diffuse albedo (rgb)");
        r.addInput("hiz", "HiZBuild.hiz of the same depth (optional)").texture2D(0, 0, 1, 0).bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        input("mvec", "Motion vectors (UV, previous minus current)", true);
        const out = (name: string, desc: string, optional: boolean) => {
            const f = r.addOutput(name, desc).texture2D(w, h).format(ResourceFormat.RGBA32Float)
                .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
            if (optional) f.flags(FieldFlags.Optional);
        };
        out("dst", "Colour with indirect light", false);
        out("indirect", "Added indirect light before intensity (rgb), ambient occlusion (a)", true);
        out("gi", "Debug: incident radiance / pi after denoise + temporal (rgb), ambient occlusion (a)", true);
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
        const albedo = rd.getTexture("diffuseOpacity");
        if (!dst || !color || !depth || !normW || !albedo) return;
        if (!this.scene) throw new Error("SSGI: requires a scene (camera)");
        const W = dst.width, H = dst.height;
        const scale = this.p.resolution === "half" ? 2 : 1;
        const TW = Math.ceil(W / scale), TH = Math.ceil(H / scale);
        const temporal = this.p.temporal && this.p.temporalAlpha > 0;
        // The tracker runs every frame; its key (size, resolution, temporal on/off) resets the history when the layout changes.
        const frame = this.history.begin(this.scene, `${W}x${H}:${scale}:${temporal}`);
        const radiance = rd.getTexture("radiance") ?? color;
        const hiz = rd.getTexture("hiz");
        const mvec = rd.getTexture("mvec");
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        this.dummy ??= this.device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(4), flags);
        this.dummyRG ??= this.device.createTexture2D(1, 1, ResourceFormat.RG32Float, 1, 1, new Float32Array(2), flags);
        const traced = this.tex("traced", TW, TH), tmp = this.tex("denoise", TW, TH);
        const giOut = rd.getTexture("gi");
        const indOut = rd.getTexture("indirect") ?? this.tex("indirect", W, H);

        const cam = this.scene.camera;
        const data = cam.getData();
        const pos = cam.getPosition(), tgt = cam.getTarget();
        const f = [tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z];
        const fl = Math.hypot(f[0]!, f[1]!, f[2]!) || 1;
        // Pixels per world unit at view depth 1 (vertical).
        const projScale = H / (2 * Math.tan(0.5 * cam.getFovY()));
        const painted = this.p.mode === "painted";
        const soft = Math.min(Math.max(painted || this.p.softness > 0 ? this.p.softness : 0, 0), 1);
        const iters = Math.max(0, Math.round(this.p.denoiseIterations)) + Math.round(3 * soft);
        const seedFrame = (temporal ? frame.frameIndex : 0) + Math.max(0, Math.round(this.p.seed)) * 7919;
        const setCB = (cp: ComputePass) => {
            const root = cp.getRootVar();
            const cb = root["CB"]!;
            cb["gFrameDim"] = [W, H];
            cb["gTraceDim"] = [TW, TH];
            cb["gScale"] = scale;
            cb["gSliceCount"] = Math.max(1, Math.round(this.p.sliceCount));
            cb["gStepCount"] = Math.max(1, Math.round(this.p.stepCount));
            cb["gFrameSeed"] = seedFrame >>> 0;
            cb["gInvViewProj"] = data.invViewProj;
            cb["gCamPos"] = [pos.x, pos.y, pos.z];
            cb["gRadius"] = Math.max(this.p.radius, 0);
            cb["gCamFwd"] = f.map((x) => x / fl);
            cb["gThickness"] = Math.max(this.p.thickness, 0);
            cb["gProjScale"] = projScale;
            cb["gMaxRadiusPx"] = Math.max(this.p.maxRadiusPx, 0) * H;
            cb["gHizMips"] = hiz ? hiz.mipCount : 0;
            cb["gHizStartPx"] = Math.max(this.p.hizStartPx, 1);
            cb["gRadianceClamp"] = Math.max(this.p.radianceClamp, 0);
            cb["gSigmaN"] = 32 * (1 - 0.9 * soft);
            cb["gSigmaZ"] = 0.02 * (1 + 9 * soft);
            cb["gTemporalAlpha"] = Math.min(Math.max(this.p.temporalAlpha, 0), 0.999);
            cb["gHistoryValid"] = temporal && frame.frameIndex > 0 ? 1 : 0;
            cb["gIntensity"] = this.p.intensity;
            cb["gPainted"] = painted ? 1 : 0;
            cb["gHueShift"] = (this.p.hueShift * Math.PI) / 180;
            cb["gSaturation"] = Math.max(this.p.saturation, 0);
            cb["gTint"] = this.p.tint;
            cb["gSkyColor"] = this.p.skyColor;
            cb["gGroundColor"] = this.p.groundColor;
            cb["gHemiIntensity"] = this.p.hemiIntensity;
            cb["gHemiOcclusion"] = Math.min(Math.max(this.p.hemiOcclusion, 0), 1);
            cb["gUp"] = this.p.upVector;
            return root;
        };

        {
            const cp = this.pass("trace");
            const root = setCB(cp);
            root["gDepth"] = depth;
            root["gNormW"] = normW;
            root["gRadiance"] = radiance;
            root["gHiZ"] = hiz ?? this.dummyRG;
            root["gTraceOut"] = traced;
            cp.execute(ctx, TW, TH);
        }
        let src = traced, other = tmp;
        for (let i = 0; i < iters; i++) {
            const cp = this.pass("denoise");
            const root = setCB(cp);
            root["CB"]!["gAtrousStep"] = 1 << i;
            root["gDepth"] = depth;
            root["gNormW"] = normW;
            root["gDenoiseIn"] = src;
            root["gDenoiseOut"] = other;
            cp.execute(ctx, TW, TH);
            [src, other] = [other, src];
        }
        const pair = temporal ? this.pp.get(ctx, W, H, frame) : null;
        const acc = pair ? pair.write : this.tex("accum", W, H);
        {
            const cp = this.pass("accumulate");
            const root = setCB(cp);
            const read = pair ? pair.read : this.dummy;
            root["gDepth"] = depth;
            root["gNormW"] = normW;
            root["gDenoised"] = src;
            root["gMvec"] = mvec ?? this.dummy;
            root["gHistory"] = read;
            root["gAccumOut"] = acc;
            cp.execute(ctx, W, H);
        }
        {
            const cp = this.pass("composite");
            const root = setCB(cp);
            root["gColor"] = color;
            root["gAccum"] = acc;
            root["gDepth"] = depth;
            root["gNormW"] = normW;
            root["gAlbedo"] = albedo;
            root["gDst"] = dst;
            root["gIndirectOut"] = indOut;
            root["gGiOut"] = giOut ?? this.tex("giDebug", W, H);
            cp.execute(ctx, W, H);
        }
    }
}

registerRenderPass("SSGI", (device, props) => new SSGI(device, props), {
    desc: "Screen-space one-bounce diffuse GI (visibility bitmask): colour bleeding, denoise, temporal accumulation (history resets on cuts), optional painted bounce",
});
