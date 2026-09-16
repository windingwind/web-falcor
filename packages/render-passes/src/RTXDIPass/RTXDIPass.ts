/**
 * Standalone ReSTIR direct-illumination pass mirroring
 * Source/RenderPasses/RTXDIPass (PrepareSurfaceData + FinalShading around the
 * Rendering/RTXDI module, which carries the actual RTXDI SDK resampling).
 *
 * Web divergences (documented, docs §RenderPasses):
 * - GBUFFER_ADJUST_SHADING_NORMALS is fixed to 0: the web RenderData carries
 *   no inter-pass dictionary and the native flag defaults to false (GBufferBase
 *   default; VBufferRT never enables it in the shipped graphs).
 * - Optional outputs bind format-matched 1x1 dummies when their is_valid_*
 *   define is 0 but the storage binding survives DCE (native binds null UAVs;
 *   WebGPU requires every layout entry bound and forbids writable aliasing).
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
    ResourceType,
    RTXDI,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type RTXDIOptions,
    type ShaderVar,
    type UIWidgets,
    kDefaultRTXDIOptions,
    type RTXDIMode,
} from "@web-falcor/falcor";

const kPrepareSurfaceDataFile = "RenderPasses/RTXDIPass/PrepareSurfaceData.cs.slang";
const kFinalShadingFile = "RenderPasses/RTXDIPass/FinalShading.cs.slang";

/** Mirrors kOutputChannels (all optional, RGBA32Float). */
const kOutputChannels: { name: string; texname: string; desc: string }[] = [
    { name: "color", texname: "gColor", desc: "Final color" },
    { name: "emission", texname: "gEmission", desc: "Emissive color" },
    { name: "diffuseIllumination", texname: "gDiffuseIllumination", desc: "Diffuse illumination" },
    { name: "diffuseReflectance", texname: "gDiffuseReflectance", desc: "Diffuse reflectance" },
    { name: "specularIllumination", texname: "gSpecularIllumination", desc: "Specular illumination" },
    { name: "specularReflectance", texname: "gSpecularReflectance", desc: "Specular reflectance" },
];

export class RTXDIPass extends RenderPass {
    private rtxdi: RTXDI | null = null;
    private options: Partial<RTXDIOptions> = {};
    private prepareSurfaceDataPass: ComputePass | null = null;
    private finalShadingPass: ComputePass | null = null;
    private frameDim: [number, number] = [0, 0];
    private rg32Dummy: Texture | null = null;
    private outputDummies = new Map<string, Texture>();

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [key, value] of props.entries()) {
            if (key === "options") this.options = value as Partial<RTXDIOptions>;
            // Native logs a warning for unknown keys (the upstream RTXDI.py
            // graph passes a stale 'useVBuffer' property that native ignores).
            else console.warn(`Unknown property '${key}' in RTXDIPass properties.`);
        }
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Visibility buffer in packed format").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("texGrads", "Texture gradients").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("mvec", "Motion vector buffer (float format)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        for (const { name, desc } of kOutputChannels) {
            r.addOutput(name, desc)
                .texture2D(w, h)
                .format(ResourceFormat.RGBA32Float)
                .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
                .flags(FieldFlags.Optional);
        }
        return r;
    }

    override compile(_ctx: RenderContext, compileData: CompileData): void {
        this.frameDim = [compileData.defaultTexDims[0], compileData.defaultTexDims[1]];
    }

    override getProperties(): Properties {
        return new Properties({ options: this.options as Record<string, never> });
    }

    /**
     * Mirrors RTXDIPass::renderUI -> RTXDI::renderUI: edits the RTXDI options; a change
     * recreates the RTXDI context (native setOptions re-creates its resources/programs).
     */
    override renderUI(ui: UIWidgets): void {
        const o: RTXDIOptions = { ...kDefaultRTXDIOptions, ...this.options };
        const set = <K extends keyof RTXDIOptions>(key: K) => (v: RTXDIOptions[K]) => {
            this.options = { ...o, [key]: v };
            if (this.scene) {
                this.rtxdi = new RTXDI(this.device, this.scene, this.options);
                this.prepareSurfaceDataPass = null;
                this.finalShadingPass = null;
            }
        };
        const int = <K extends keyof RTXDIOptions>(key: K) => (v: number) => set(key)(Math.round(v) as RTXDIOptions[K]);
        const modes: RTXDIMode[] = ["NoResampling", "SpatialResampling", "TemporalResampling", "SpatiotemporalResampling"];
        ui.dropdown("Mode", modes, o.mode, (v) => set("mode")(v as RTXDIMode));
        const presample = ui.group("Light presampling");
        presample.slider("Tile count", o.presampledTileCount, 1, 1024, 1, int("presampledTileCount"));
        presample.slider("Tile size", o.presampledTileSize, 256, 8192, 128, int("presampledTileSize"));
        presample.checkbox("Store compact light info", o.storeCompactLightInfo, set("storeCompactLightInfo"));
        const initial = ui.group("Initial candidate sampling");
        initial.slider("Local light samples", o.localLightCandidateCount, 0, 256, 1, int("localLightCandidateCount"));
        initial.slider("Infinite light samples", o.infiniteLightCandidateCount, 0, 256, 1, int("infiniteLightCandidateCount"));
        initial.slider("Environment light samples", o.envLightCandidateCount, 0, 256, 1, int("envLightCandidateCount"));
        initial.slider("BRDF samples", o.brdfCandidateCount, 0, 256, 1, int("brdfCandidateCount"));
        initial.slider("BRDF Cutoff", o.brdfCutoff, 0, 1, 0.001, set("brdfCutoff"));
        initial.checkbox("Test selected candidate visibility", o.testCandidateVisibility, set("testCandidateVisibility"));
        const resampling = ui.group("Resampling");
        const bias = ["Off", "Basic", "Pairwise", "RayTraced"];
        resampling.dropdown("Bias correction", bias, bias[o.biasCorrection] ?? "Basic", (v) => set("biasCorrection")(Math.max(0, bias.indexOf(v))));
        resampling.slider("Depth threshold", o.depthThreshold, 0, 1, 0.001, set("depthThreshold"));
        resampling.slider("Normal threshold", o.normalThreshold, 0, 1, 0.001, set("normalThreshold"));
        const spatial = ui.group("Spatial resampling");
        spatial.slider("Sampling radius", o.samplingRadius, 0, 100, 0.1, set("samplingRadius"));
        spatial.slider("Sample count", o.spatialSampleCount, 0, 25, 1, int("spatialSampleCount"));
        spatial.slider("Iterations", o.spatialIterations, 0, 10, 1, int("spatialIterations"));
        const temporal = ui.group("Temporal resampling");
        temporal.slider("Max history length", o.maxHistoryLength, 0, 100, 1, int("maxHistoryLength"));
        temporal.slider("Boiling filter strength", o.boilingFilterStrength, 0, 1, 0.001, set("boilingFilterStrength"));
        const misc = ui.group("Misc");
        misc.checkbox("Use emissive textures", o.useEmissiveTextures, set("useEmissiveTextures"));
        misc.checkbox("Enable permutation sampling", o.enablePermutationSampling, set("enablePermutationSampling"));
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.rtxdi = scene ? new RTXDI(this.device, scene, this.options) : null;
        this.prepareSurfaceDataPass = null;
        this.finalShadingPass = null;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene || !this.rtxdi) {
            for (const { name } of kOutputChannels) {
                const tex = renderData.getTexture(name);
                if (tex) ctx.clearTexture(tex);
            }
            return;
        }

        const vbuffer = renderData.getTexture("vbuffer")!;
        // Native binds null SRVs for missing optional inputs (reads return
        // zero); WebGPU needs a texture for every surviving binding.
        if (!this.rg32Dummy) {
            this.rg32Dummy = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: 1,
                height: 1,
                format: ResourceFormat.RG32Float,
                bindFlags: ResourceBindFlags.ShaderResource,
                name: "RTXDIPass::rg32Dummy",
            });
        }
        const mvec = renderData.getTexture("mvec") ?? this.rg32Dummy;
        const texGrads = renderData.getTexture("texGrads") ?? this.rg32Dummy;

        this.rtxdi.beginFrame(ctx, this.frameDim);

        this.prepareSurfaceData(ctx, vbuffer, texGrads, mvec);

        this.rtxdi.update(ctx, mvec);

        this.finalShading(ctx, vbuffer, renderData, mvec);

        this.rtxdi.endFrame(ctx);
    }

    /** Mirrors RTXDIPass::prepareSurfaceData. */
    private prepareSurfaceData(ctx: RenderContext, vbuffer: Texture, texGrads: Texture, mvec: Texture): void {
        if (!this.prepareSurfaceDataPass) {
            const defines = this.scene!.getSceneDefines()
                .addAll(this.rtxdi!.getDefines())
                .add("GBUFFER_ADJUST_SHADING_NORMALS", 0);
            this.prepareSurfaceDataPass = ComputePass.create(this.device, { path: kPrepareSurfaceDataFile, defines });
        }

        const root = this.prepareSurfaceDataPass.getRootVar();
        this.rtxdi!.setShaderData(root, mvec); // binds gScene + gRTXDI

        const v = root["gPrepareSurfaceData"] as ShaderVar;
        v["vbuffer"] = vbuffer;
        v["texGrads"] = texGrads;
        v["frameDim"] = this.frameDim;

        this.prepareSurfaceDataPass.execute(ctx, this.frameDim[0], this.frameDim[1]);
    }

    /** Mirrors RTXDIPass::finalShading. */
    private finalShading(ctx: RenderContext, vbuffer: Texture, renderData: RenderData, mvec: Texture): void {
        if (!this.finalShadingPass) {
            const defines = this.scene!.getSceneDefines()
                .addAll(this.rtxdi!.getDefines())
                .add("GBUFFER_ADJUST_SHADING_NORMALS", 0)
                .add("USE_ENV_BACKGROUND", this.scene!.useEnvBackground ? 1 : 0);
            // is_valid_<name> defines for the optional outputs (connectivity is
            // fixed per graph compile; native re-adds them per frame).
            for (const { texname, name } of kOutputChannels) {
                defines.add(`is_valid_${texname}`, renderData.getTexture(name) !== undefined ? 1 : 0);
            }
            this.finalShadingPass = ComputePass.create(this.device, { path: kFinalShadingFile, defines });
        }

        const root = this.finalShadingPass.getRootVar();
        this.rtxdi!.setShaderData(root, mvec);

        const v = root["gFinalShading"] as ShaderVar;
        v["vbuffer"] = vbuffer;
        v["frameDim"] = this.frameDim;

        // Web divergence: the outputs are members of the FinalShading block
        // (bind-group budget; see the FinalShading.cs.slang override).
        for (const { name, texname } of kOutputChannels) {
            try {
                v[texname] = renderData.getTexture(name) ?? this.outputDummy(name);
            } catch {
                /* binding DCE'd in this variant */
            }
        }

        this.finalShadingPass.execute(ctx, this.frameDim[0], this.frameDim[1]);
    }

    private outputDummy(name: string): Texture {
        let t = this.outputDummies.get(name);
        if (!t) {
            t = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: 1,
                height: 1,
                format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                name: `RTXDIPass::dummyOut_${name}`,
            });
            this.outputDummies.set(name, t);
        }
        return t;
    }
}

registerRenderPass("RTXDIPass", (device, props) => new RTXDIPass(device, props));
