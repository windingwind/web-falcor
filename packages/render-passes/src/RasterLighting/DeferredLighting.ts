/**
 * Raster deferred direct lighting (web-only; Falcor 4's ForwardLighting/CSM passes were
 * dropped upstream). Consumes a V-buffer (GBufferRaster.vbuffer or VBufferRaster.vbuffer),
 * rebuilds the scene's ShadingData per pixel and evaluates, without stochastic sampling:
 *  - point / spot / directional analytic lights (Falcor's evalLightApproximate),
 *    shadowed by CascadedShadowMap (PCF), by ray-traced visibility, or not at all;
 *  - the env map / constant sky as ambient light (unshadowed SH9 diffuse irradiance +
 *    env lookup in the reflection direction; envSpecularOcclusion traces one visibility ray
 *    along it so glossy surfaces don't mirror the env map through nearby geometry);
 *  - emission, and the env map as background.
 * Optional input `clipBuffer` (GBufferRaster's per-instance clip entries): with RayTraced shadows (and the env specular
 * occlusion ray) the visibility rays skip the clipped parts of clipped instances, so clipped / dissolving objects lose
 * their shadow with them. Without it (or without rays) nothing changes; for ShadowMap shadows connect the same buffer
 * to CascadedShadowMap.clipBuffer.
 * brdf = "Material" evaluates the scene material (IMaterialInstance::eval, i.e. the same BSDF
 * the path tracers use); "LambertGGX" is the classic real-time model from the material's
 * BSDFProperties (Lambert + GGX / height-correlated Smith / Schlick).
 */

import {
    Buffer,
    ComputePass,
    FieldFlags,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    SAMPLE_GENERATOR_UNIFORM,
    SampleGenerator,
    float4x4,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { kCascadedShadowMapData, kMaxCascades, type CascadedShadowMapData } from "./CascadedShadowMap.js";

/** Optional per-instance clip input (GBufferRaster's clip buffer layout: one float4 per geometry instance). */
const kClipBuffer = "clipBuffer";
const kClipStride = 16;

const kShaderFile = "RenderPasses/RasterLighting/DeferredLighting.cs.slang";
const kSHShaderFile = "RenderPasses/RasterLighting/EnvSH.cs.slang";

const kBrdfModels = ["Material", "LambertGGX"] as const;
const kShadowModes = ["None", "ShadowMap", "RayTraced"] as const;
type BrdfModel = (typeof kBrdfModels)[number];
type ShadowMode = (typeof kShadowModes)[number];
/** Auto = on when shadowMode is RayTraced (ray queries are in use anyway), off otherwise. */
const kEnvOcclusionModes = ["Auto", "On", "Off"] as const;
type EnvOcclusionMode = (typeof kEnvOcclusionModes)[number];

const kOptionalOutputs = [
    { name: "visibility", texname: "gVisibility", format: ResourceFormat.R32Float, desc: "Visibility of the shadowed light (1 = lit)" },
    { name: "diffuse", texname: "gDiffuse", format: ResourceFormat.RGBA32Float, desc: "Direct diffuse radiance (Material model: whole analytic-light BSDF term)" },
    { name: "specular", texname: "gSpecular", format: ResourceFormat.RGBA32Float, desc: "Direct specular radiance" },
];

export class DeferredLighting extends RenderPass {
    private pass: ComputePass | null = null;
    private passKey = "";
    private shPass: ComputePass | null = null;
    private shBuffer: Buffer | null = null;
    private shSource: unknown = null;
    private sampleGenerator: SampleGenerator;

    private brdf: BrdfModel = "Material";
    private shadowMode: ShadowMode = "ShadowMap";
    private envLighting = true;
    private analyticLights = true;
    private emissive = true;
    private envBackground = true;
    private envSpecularOcclusion: EnvOcclusionMode = "Auto";
    /** Constant shadow depth bias in world units. */
    private depthBias = 0.002;
    /** Normal offset in shadow texels. */
    private normalBias = 1.5;
    private pcfRadius = 1;

    constructor(device: Device, props: Properties) {
        super(device);
        const brdf = props.get<string>("brdf", "Material");
        if (!(kBrdfModels as readonly string[]).includes(brdf)) throw new Error(`DeferredLighting: unknown brdf '${brdf}' (${kBrdfModels.join(", ")})`);
        this.brdf = brdf as BrdfModel;
        const sm = props.get<string>("shadowMode", "ShadowMap");
        if (!(kShadowModes as readonly string[]).includes(sm)) throw new Error(`DeferredLighting: unknown shadowMode '${sm}' (${kShadowModes.join(", ")})`);
        this.shadowMode = sm as ShadowMode;
        this.envLighting = props.get("envLighting", true);
        this.analyticLights = props.get("analyticLights", true);
        this.emissive = props.get("emissive", true);
        this.envBackground = props.get("envBackground", true);
        const eo = props.get<string>("envSpecularOcclusion", "Auto");
        if (!(kEnvOcclusionModes as readonly string[]).includes(eo)) throw new Error(`DeferredLighting: unknown envSpecularOcclusion '${eo}' (${kEnvOcclusionModes.join(", ")})`);
        this.envSpecularOcclusion = eo as EnvOcclusionMode;
        this.depthBias = props.get("depthBias", 0.002);
        this.normalBias = props.get("normalBias", 1.5);
        this.pcfRadius = props.get("pcfRadius", 1);
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_UNIFORM);
    }

    override getProperties(): Properties {
        return new Properties({
            brdf: this.brdf,
            shadowMode: this.shadowMode,
            envLighting: this.envLighting,
            analyticLights: this.analyticLights,
            emissive: this.emissive,
            envBackground: this.envBackground,
            envSpecularOcclusion: this.envSpecularOcclusion,
            depthBias: this.depthBias,
            normalBias: this.normalBias,
            pcfRadius: this.pcfRadius,
        });
    }

    override renderUI(ui: UIWidgets): void {
        ui.dropdown("BRDF", [...kBrdfModels], this.brdf, (v) => (this.brdf = v as BrdfModel));
        ui.dropdown("Shadows", [...kShadowModes], this.shadowMode, (v) => (this.shadowMode = v as ShadowMode));
        ui.checkbox("Env / sky light", this.envLighting, (v) => (this.envLighting = v));
        ui.checkbox("Analytic lights", this.analyticLights, (v) => (this.analyticLights = v));
        ui.checkbox("Emission", this.emissive, (v) => (this.emissive = v));
        ui.checkbox("Env background", this.envBackground, (v) => (this.envBackground = v));
        ui.dropdown("Env specular occlusion", [...kEnvOcclusionModes], this.envSpecularOcclusion, (v) => (this.envSpecularOcclusion = v as EnvOcclusionMode));
        ui.slider("Depth bias (world)", this.depthBias, 0, 0.1, 0.0005, (v) => (this.depthBias = v));
        ui.slider("Normal bias (texels)", this.normalBias, 0, 8, 0.1, (v) => (this.normalBias = v));
        ui.slider("PCF radius", this.pcfRadius, 0, 4, 1, (v) => (this.pcfRadius = Math.round(v)));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Packed hit information (V-buffer)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("viewW", "View direction (world)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("shadowMap", "CascadedShadowMap.shadowMap").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput(kClipBuffer, "Per-instance clip entries (GBufferRaster clipBuffer): visibility rays ignore clipped geometry")
            .rawBuffer(0)
            .bindFlags(ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        r.addOutput("color", "Direct radiance")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        for (const o of kOptionalOutputs) {
            r.addOutput(o.name, o.desc).texture2D(w, h).format(o.format)
                .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
                .flags(FieldFlags.Optional);
        }
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.pass = null;
        this.passKey = "";
        this.shSource = null;
    }

    /** Projects the env map onto SH9 once per env map (GPU, one workgroup). */
    private updateEnvSH(ctx: RenderContext): void {
        const env = this.scene!.getEnvMap();
        if (this.shBuffer && this.shSource === env) return;
        if (!this.shBuffer) {
            this.shBuffer = new Buffer(this.device, {
                size: 9 * 16,
                structSize: 16,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                memoryType: MemoryType.DeviceLocal,
                name: "DeferredLighting::envSH",
            });
        }
        if (!this.shPass) {
            this.shPass = ComputePass.create(this.device, { path: kSHShaderFile, defines: this.scene!.getSceneDefines() });
        }
        const root = this.shPass.getRootVar();
        this.scene!.bindShaderData(root);
        root["gSH"] = this.shBuffer;
        this.shPass.execute(ctx, 64, 1);
        this.shSource = env;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const color = renderData.getTexture("color")!;
        if (!this.scene) {
            ctx.clearTexture(color, [0, 0, 0, 0]);
            return;
        }
        const scene = this.scene;
        const csm = renderData.dictionary.get(kCascadedShadowMapData) as CascadedShadowMapData | undefined;
        const shadowMap = renderData.getTexture("shadowMap");
        const useShadowMap = this.shadowMode === "ShadowMap" && !!shadowMap && !!csm;
        const shadowMode = this.shadowMode === "RayTraced" ? 2 : useShadowMap ? 1 : 0;
        const useEnv = this.envLighting && scene.useEnvLight;
        const outputs = kOptionalOutputs.map((o) => ({ ...o, tex: renderData.getTexture(o.name) }));

        const envOcclusion = useEnv && (this.envSpecularOcclusion === "On" || (this.envSpecularOcclusion === "Auto" && shadowMode === 2));
        const defines = scene.getSceneDefines().addAll({
            BRDF_MODEL: this.brdf === "Material" ? 0 : 1,
            SHADOW_MODE: shadowMode,
            USE_ANALYTIC_LIGHTS: this.analyticLights && scene.useAnalyticLights ? 1 : 0,
            USE_ENV_LIGHT: useEnv ? 1 : 0,
            USE_ENV_BACKGROUND: this.envBackground && scene.useEnvBackground ? 1 : 0,
            USE_EMISSIVE: this.emissive ? 1 : 0,
            ENV_SPEC_OCCLUSION: envOcclusion ? 1 : 0,
            is_valid_gViewW: renderData.getTexture("viewW") ? 1 : 0,
        });
        for (const o of outputs) defines.add(`is_valid_${o.texname}`, o.tex ? 1 : 0);
        // Per-instance clip of the visibility rays: only with a clip buffer bound and rays traced (the default define set
        // stays untouched; the shadow-map path clips in CascadedShadowMap).
        const clipBuffer = renderData.getBuffer(kClipBuffer);
        const clipCount = clipBuffer ? Math.floor(clipBuffer.size / kClipStride) : 0;
        const clip = clipBuffer !== undefined && clipCount > 0 && (shadowMode === 2 || envOcclusion);
        if (clip) defines.add("INSTANCE_CLIP", 1);
        defines.addAll(this.sampleGenerator.getDefines());
        const key = defines.key();
        if (!this.pass || key !== this.passKey) {
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.passKey = key;
        }
        if (useEnv) this.updateEnvSH(ctx);

        const root = this.pass.getRootVar();
        scene.bindShaderData(root);
        const cb = root["CB"];
        cb["gFrameDim"] = [color.width, color.height];
        const cam = scene.camera;
        const p = cam.getPosition(), t = cam.getTarget();
        const fwd = [t.x - p.x, t.y - p.y, t.z - p.z];
        const len = Math.hypot(fwd[0]!, fwd[1]!, fwd[2]!);
        cb["gCamForward"] = fwd.map((x) => x / len);
        const env = scene.getEnvMap();
        cb["gEnvSpecLodScale"] = env ? Math.max(0, env.texture.mipCount - 1) : 0;
        if (useShadowMap) {
            const mats: float4x4[] = [];
            for (let c = 0; c < kMaxCascades; c++) mats.push(csm!.viewProj[Math.min(c, csm!.cascadeCount - 1)]!);
            const pad = (a: number[]) => [0, 1, 2, 3].map((i) => a[Math.min(i, a.length - 1)]!);
            cb["gShadowLightIndex"] = csm!.lightIndex;
            cb["gCascadeCount"] = csm!.cascadeCount;
            cb["gCascadeViewProj"] = mats;
            cb["gCascadeSplits"] = pad(csm!.splits);
            cb["gCascadeTexelWorld"] = pad(csm!.texelWorld);
            cb["gCascadeDepthScale"] = pad(csm!.depthRange.map((r) => 1 / r));
            cb["gCascadeSize"] = csm!.size;
            cb["gDepthBias"] = this.depthBias;
            cb["gNormalBias"] = this.normalBias;
            cb["gPcfRadius"] = this.pcfRadius;
            root["gShadowMap"] = shadowMap!;
        } else {
            // Ray-traced mode shadows every analytic light; the visibility output reports the first directional one.
            const lights = scene.activeLights;
            cb["gShadowLightIndex"] = shadowMode === 2 ? lights.findIndex((l) => l.type === 1) : -1;
            cb["gCascadeCount"] = 0;
        }
        if (useEnv) root["gEnvSH"] = this.shBuffer!;
        if (clip) {
            root["InstanceClipCB"]["gClipCount"] = clipCount;
            root["gInstanceClip"] = clipBuffer!;
        }
        root["gVBuffer"] = renderData.getTexture("vbuffer")!;
        const viewW = renderData.getTexture("viewW");
        if (viewW) root["gViewW"] = viewW;
        root["gOutputColor"] = color;
        for (const o of outputs) if (o.tex) root[o.texname] = o.tex;
        this.pass.execute(ctx, color.width, color.height);
    }
}

registerRenderPass("DeferredLighting", (device, props) => new DeferredLighting(device, props), {
    desc: "Raster deferred direct lighting (analytic lights + shadow map, SH sky, emission) from a V-buffer",
});
