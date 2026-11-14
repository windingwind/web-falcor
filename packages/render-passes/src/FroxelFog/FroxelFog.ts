/**
 * FroxelFog: froxel volumetric fog (view-frustum voxel grid) with exponential height fog, local box/sphere fog volumes,
 * shadowed in-scattering of one analytic light through the shared visibility service, optional froxel-space temporal
 * reprojection, front-to-back integration and apply. One render-graph node, three dispatches per frame:
 *   scatter   (grid X x Y x Z, FroxelFogScatter.cs.slang): media + lit in-scatter per froxel, temporal blend;
 *   integrate (grid X x Y, FroxelFog.cs.slang): energy-conserving front-to-back accumulation per froxel column;
 *   apply     (frame W x H, FroxelFog.cs.slang): dst = color * T + S at each pixel's view depth.
 *
 * Grid: gridX x gridY froxels over the screen, gridZ slices; slice edge k sits at view depth far * (k / gridZ)^spread
 * (spread > 1 puts more slices near the camera). Points beyond `far` get the fog accumulated up to `far`; background
 * pixels (posW.w == 0) get the full column when skyFog is on. The volumes are structured buffers with one element per
 * froxel (precision "half" = RGBA16F packed, 8 bytes; "float" = RGBA32F).
 *
 * Media: global density * exp(-heightFalloff * (dot(p, upAxis) - heightBase)) with scattering albedo `albedo`, plus up
 * to 8 local volumes (`volumes`: [{shape: "box" | "sphere", center, halfExtent (or radius), density, albedo, softness,
 * noise, noiseScale}]); each term can be modulated by animated value-noise fBm (noise amount, noiseScale, wind, time).
 * Light: the visibility light (visMode ShadowMap: the CascadedShadowMap light; else lightIndex, -1 = first directional)
 * with Henyey-Greenstein phase (anisotropy), times lightColor x lightIntensity; otherLights > 0 adds the scene's other analytic
 * lights x otherLights (point/spot attenuation as in the scene; shadowed only with visMode RayTraced, since the shadow map
 * belongs to the primary light); plus the isotropic `ambient` radiance.
 *
 * History (temporal = true): the scatter volume lives in a pair of buffers driven by a HistoryTracker (Utils/History.ts):
 * reset on the first execute, resetHistory() / resetGraphHistory(), a scene or camera change, a change of grid or
 * precision, Camera.resetHistory() and a primed camera cut. On a reset both buffers are cleared and the Halton jitter
 * sequence restarts (its index is the number of frames since the reset), so a shot rendered after a reset is
 * bit-identical to the same shot on a fresh context. Without temporal the pass keeps no state that affects its output.
 *
 * Identity: with density 0 and no local volume of positive density the pass copies color to dst bit for bit (fog output
 * (0, 0, 0, 1)) and resets its history.
 *
 * Inputs : color (lit HDR colour), posW (GBufferRaster.posW; w = 0 on background), shadowMap (optional:
 *          CascadedShadowMap.shadowMap for visMode ShadowMap).
 * Outputs: dst (fogged colour), fog (optional: rgb = in-scattered radiance S, a = transmittance T).
 *
 * The froxel layout (exponential slicing with a spread exponent, Halton-jittered froxel-space reprojection, per-slice
 * energy-conserving integration) follows Godot Engine's volumetric fog (MIT, https://github.com/godotengine/godot,
 * Copyright (c) 2014-present Godot Engine contributors, (c) 2007-2014 Juan Linietsky, Ariel Manzur); the notice is in
 * FroxelFogCommon.slangh.
 */

import {
    type Buffer,
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
    type PropertyValue,
    type Device,
    type RenderContext,
    type Texture,
    type UIWidgets,
} from "@web-falcor/falcor";
import { HistoryTracker } from "../Utils/History.js";
import { VisibilityService, kDefaultVisibilityParams, kVisibilityModes, type VisibilityMode } from "../RasterLighting/VisibilityService.js";

const kScatterShader = "RenderPasses/FroxelFog/FroxelFogScatter.cs.slang";
const kShader = "RenderPasses/FroxelFog/FroxelFog.cs.slang";
export const kFroxelFogMaxVolumes = 8;
const kPrecisions = ["half", "float"] as const;

export interface FogVolume {
    shape?: "box" | "sphere";
    center: number[];
    /** Half extent per axis (box) or radii (sphere / ellipsoid); a single number = all axes. */
    halfExtent: number[] | number;
    density: number;
    albedo?: number[];
    /** Edge softness as a fraction of the extent (0 = hard edge). */
    softness?: number;
    noise?: number;
    noiseScale?: number;
}

const kDefaults = {
    density: 0.02,
    heightFalloff: 0,
    heightBase: 0,
    upAxis: [0, 1, 0] as number[],
    albedo: [1, 1, 1] as number[],
    anisotropy: 0.3,
    lightIntensity: 1,
    lightColor: [1, 1, 1] as number[],
    otherLights: 0,
    ambient: [0, 0, 0] as number[],
    noise: 0,
    noiseScale: 0.2,
    wind: [0, 0, 0] as number[],
    time: 0,
    volumes: [] as FogVolume[],
    visMode: "ShadowMap" as string,
    lightIndex: -1,
    gridX: 160,
    gridY: 90,
    gridZ: 64,
    far: 100,
    spread: 2,
    skyFog: true,
    temporal: true,
    temporalBlend: 0.9,
    precision: "half" as string,
};

/** Halton (2, 3, 5) froxel jitter of frame `i` since the last reset (frame 0 = centre). */
export function froxelJitter(i: number): [number, number, number] {
    if (i <= 0) return [0.5, 0.5, 0.5];
    const h = (n: number, b: number) => {
        let f = 1, r = 0;
        while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); }
        return r;
    };
    const k = ((i - 1) % 16) + 1;
    return [h(k, 2), h(k, 3), h(k, 5)];
}

const v3 = (a: unknown, d: number[]): number[] => {
    if (typeof a === "number") return [a, a, a];
    const x = Array.isArray(a) ? a.map(Number) : d;
    return [x[0] ?? d[0]!, x[1] ?? d[1]!, x[2] ?? d[2]!];
};

interface CamState { pos: number[]; U: number[]; V: number[]; W: number[] }

export class FroxelFog extends RenderPass {
    private p = { ...kDefaults };
    private scatterPass: ComputePass | null = null;
    private scatterKey = "";
    private passes: Record<string, ComputePass> = {};
    private readonly history = new HistoryTracker();
    private vol: [Buffer, Buffer] | null = null;
    private integrated: Buffer | null = null;
    private volKey = "";
    private prevCam: CamState | null = null;
    private fogScratch: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof typeof kDefaults)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as PropertyValue);
        this.p.visMode = VisibilityService.parseMode("FroxelFog", this.p.visMode);
        if (!(kPrecisions as readonly string[]).includes(this.p.precision)) throw new Error(`FroxelFog: precision must be one of ${kPrecisions.join("|")} (got ${this.p.precision})`);
        if (!Array.isArray(this.p.volumes)) throw new Error("FroxelFog: volumes must be a list");
        if (this.p.volumes.length > kFroxelFogMaxVolumes) throw new Error(`FroxelFog: at most ${kFroxelFogMaxVolumes} volumes (got ${this.p.volumes.length})`);
        for (const v of this.p.volumes) {
            if (v.shape !== undefined && v.shape !== "box" && v.shape !== "sphere") throw new Error(`FroxelFog: volume shape must be box|sphere (got ${String(v.shape)})`);
        }
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p } as unknown as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Density", this.p.density, 0, 0.5, 0.001, (v) => (this.p.density = v));
        ui.slider("Height falloff", this.p.heightFalloff, 0, 2, 0.01, (v) => (this.p.heightFalloff = v));
        ui.slider("Height base", this.p.heightBase, -100, 100, 0.1, (v) => (this.p.heightBase = v));
        ui.slider("Anisotropy", this.p.anisotropy, -0.95, 0.95, 0.01, (v) => (this.p.anisotropy = v));
        ui.slider("Light intensity", this.p.lightIntensity, 0, 10, 0.01, (v) => (this.p.lightIntensity = v));
        ui.slider("Other lights", this.p.otherLights, 0, 10, 0.01, (v) => (this.p.otherLights = v));
        ui.slider("Noise", this.p.noise, 0, 1, 0.01, (v) => (this.p.noise = v));
        ui.dropdown("Visibility", [...kVisibilityModes], this.p.visMode, (v) => (this.p.visMode = v));
        ui.slider("Far", this.p.far, 1, 2000, 1, (v) => (this.p.far = v));
        ui.checkbox("Temporal", this.p.temporal, (v) => (this.p.temporal = v));
        ui.slider("Temporal blend", this.p.temporalBlend, 0, 0.98, 0.01, (v) => (this.p.temporalBlend = v));
    }

    /** Drops the temporal history: the next frame is computed as on a fresh context. */
    resetHistory(): void {
        this.history.reset();
    }

    /** Frames since the last history reset. */
    get frameIndex(): number {
        return this.history.frameIndex;
    }

    /** The integrated froxel volume of the last frame ((S, T) per froxel, layout as the shaders; null before the first frame). */
    get integratedVolume(): Buffer | null {
        return this.integrated;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("color", "Lit colour (HDR)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("posW", "World position (GBufferRaster.posW; w = 0 on background)").bindFlags(ResourceBindFlags.ShaderResource);
        VisibilityService.addInputs(r);
        r.addOutput("dst", "Colour with fog").texture2D(w, h).format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        r.addOutput("fog", "In-scattered radiance (rgb) and transmittance (a)").texture2D(w, h).format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.scatterPass = null;
        this.scatterKey = "";
    }

    private grid(): [number, number, number] {
        return [this.p.gridX, this.p.gridY, this.p.gridZ].map((g) => Math.max(1, Math.round(Number(g)))) as [number, number, number];
    }

    private hasFog(): boolean {
        return this.p.density > 0 || this.p.volumes.some((v) => Number(v.density) > 0);
    }

    private pass(entry: string, half: boolean): ComputePass {
        const k = `${entry}:${half}`;
        return (this.passes[k] ??= ComputePass.create(this.device, { path: kShader, csEntry: entry, defines: half ? { FROXEL_HALF: 1 } : {} }));
    }

    private buffers(n: number, half: boolean): void {
        const key = `${n}:${half}`;
        if (this.vol && this.volKey === key) return;
        const stride = half ? 8 : 16;
        const mk = () => this.device.createStructuredBuffer(stride, n, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
        this.vol = [mk(), mk()];
        this.integrated = mk();
        this.volKey = key;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst"), color = rd.getTexture("color"), posW = rd.getTexture("posW");
        if (!dst || !color || !posW) return;
        if (!this.scene) throw new Error("FroxelFog: requires a scene (camera)");
        const fogOut = rd.getTexture("fog");
        const [gx, gy, gz] = this.grid();
        const half = this.p.precision === "half";
        const temporal = !!this.p.temporal && this.p.temporalBlend > 0;
        const frame = this.history.begin(this.scene, `${gx}x${gy}x${gz}:${half}:${temporal}`);
        const cam = this.scene.camera;
        const data = cam.getData();
        const cur: CamState = {
            pos: [data.posW.x, data.posW.y, data.posW.z],
            U: [data.cameraU.x, data.cameraU.y, data.cameraU.z],
            V: [data.cameraV.x, data.cameraV.y, data.cameraV.z],
            W: [data.cameraW.x, data.cameraW.y, data.cameraW.z],
        };

        if (!this.hasFog()) {
            // Identity: no media anywhere. Copy, and start the history afresh when fog comes back.
            if (color.format === dst.format && color.width === dst.width && color.height === dst.height) ctx.copyTexture(dst, color);
            else ctx.blit(color, dst, "nearest");
            if (fogOut) ctx.clearTexture(fogOut, [0, 0, 0, 1]);
            this.history.reset();
            this.prevCam = null;
            return;
        }

        const n = gx * gy * gz;
        this.buffers(n, half);
        if (frame.reset || frame.frameIndex === 0) for (const b of this.vol!) ctx.clearBuffer(b);
        const w = temporal ? frame.frameIndex & 1 : 0;
        const write = this.vol![w]!, read = this.vol![1 - w]!;
        const historyValid = temporal && frame.frameIndex > 0 && this.prevCam !== null;
        const prev = historyValid ? this.prevCam! : cur;

        const len = (a: number[]) => Math.hypot(a[0]!, a[1]!, a[2]!);
        const sq = (a: number[]) => a[0]! * a[0]! + a[1]! * a[1]! + a[2]! * a[2]!;
        const spread = Math.max(1e-3, Number(this.p.spread));
        const far = Math.max(1e-4, Number(this.p.far));
        const jit = historyValid ? froxelJitter(frame.frameIndex) : [0.5, 0.5, 0.5];
        const up = v3(this.p.upAxis, [0, 1, 0]);
        const ul = len(up) || 1;
        const lc = v3(this.p.lightColor, [1, 1, 1]).map((x) => x * this.p.lightIntensity);
        const vis = VisibilityService.resolve(this.p.visMode as VisibilityMode, rd, this.scene, this.p.lightIndex);
        const vols: number[][] = [];
        for (const v of this.p.volumes) {
            const he = v3(v.halfExtent, [1, 1, 1]).map((x) => Math.max(Math.abs(x), 1e-6));
            vols.push([...v3(v.center, [0, 0, 0]), v.shape === "sphere" ? 1 : 0], [...he, Number(v.density) || 0],
                [...v3(v.albedo ?? [1, 1, 1], [1, 1, 1]), Math.max(0, Number(v.softness ?? 0.2))], [Number(v.noise ?? 0), Number(v.noiseScale ?? 0.5), 0, 0]);
        }
        while (vols.length < kFroxelFogMaxVolumes * 4) vols.push([0, 0, 0, 0]);
        const W = dst.width, H = dst.height;

        const setCB = (cp: ComputePass) => {
            const root = cp.getRootVar();
            const cb = root["CB"]!;
            cb["gGrid"] = [gx, gy, gz, (historyValid ? 1 : 0) | (this.p.skyFog ? 2 : 0)];
            cb["gMisc"] = [W, H, this.p.volumes.length, vis.lightIndex >= 0 ? vis.lightIndex + 1 : 0];
            cb["gCamPos"] = [...cur.pos, len(cur.W)];
            cb["gCamU"] = [...cur.U, sq(cur.U)];
            cb["gCamV"] = [...cur.V, sq(cur.V)];
            cb["gCamW"] = [...cur.W, 0];
            cb["gPrevPos"] = [...prev.pos, len(prev.W)];
            cb["gPrevU"] = [...prev.U, sq(prev.U)];
            cb["gPrevV"] = [...prev.V, sq(prev.V)];
            cb["gPrevW"] = [...prev.W, 0];
            cb["gSlices"] = [far, spread, 1 / spread, Math.min(Math.max(this.p.temporalBlend, 0), 0.999)];
            cb["gJitter"] = [...jit, this.p.time];
            cb["gGlobal"] = [Math.max(0, this.p.density), this.p.heightFalloff, this.p.heightBase, Math.min(Math.max(this.p.anisotropy, -0.99), 0.99)];
            cb["gUp"] = [...up.map((x) => x / ul), Math.max(0, this.p.noise)];
            cb["gAlbedo"] = [...v3(this.p.albedo, [1, 1, 1]), this.p.noiseScale];
            cb["gWind"] = [...v3(this.p.wind, [0, 0, 0]), Math.max(0, Number(this.p.otherLights))];
            cb["gLightScale"] = [...lc, 0];
            cb["gAmbient"] = [...v3(this.p.ambient, [0, 0, 0]), 0];
            cb["gVol"] = vols;
            return root;
        };

        {
            const defines = this.scene.getSceneDefines().addAll(vis.defines);
            if (half) defines.add("FROXEL_HALF", "1");
            const key = defines.key();
            if (!this.scatterPass || key !== this.scatterKey) {
                this.scatterPass = ComputePass.create(this.device, { path: kScatterShader, defines });
                this.scatterKey = key;
            }
            const root = setCB(this.scatterPass);
            this.scene.bindShaderData(root);
            vis.bind(root, kDefaultVisibilityParams);
            root["gFroxelSrc"] = read;
            root["gFroxelDst"] = write;
            this.scatterPass.execute(ctx, gx, gy, gz);
        }
        {
            const cp = this.pass("integrate", half);
            const root = setCB(cp);
            root["gFroxelSrc"] = write;
            root["gFroxelDst"] = this.integrated!;
            cp.execute(ctx, gx, gy);
        }
        {
            if (!fogOut && (!this.fogScratch || this.fogScratch.width !== W || this.fogScratch.height !== H)) {
                this.fogScratch = this.device.createTexture2D(W, H, ResourceFormat.RGBA32Float, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
            }
            const cp = this.pass("apply", half);
            const root = setCB(cp);
            root["gFroxelSrc"] = this.integrated!;
            root["gColor"] = color;
            root["gPosW"] = posW;
            root["gDst"] = dst;
            root["gFog"] = fogOut ?? this.fogScratch!;
            cp.execute(ctx, W, H);
        }
        this.prevCam = temporal ? cur : null;
    }
}

registerRenderPass("FroxelFog", (device, props) => new FroxelFog(device, props), {
    desc: "Froxel volumetric fog: height fog + local box/sphere volumes, shadowed in-scattering (visibility service), temporal reprojection, integration and apply",
});
