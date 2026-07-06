/**
 * Reference path tracer mirroring Source/RenderPasses/PathTracer (host side of
 * GeneratePaths + TracePass; the resolve pass is skipped for fixed spp == 1,
 * as native does). TracePass runs as the WebFalcor compute megakernel over
 * software ray queries (see the TracePass.rt.slang override).
 *
 * Limits (documented in DESIGN.md parity matrix): fixed spp == 1 only (no
 * adaptive sampleCount input), guide outputs supported via ResolvePass, no
 * NRD outputs, no RTXDI/SER; rayCount/pathLength await the PixelStats port.
 */

import {
    Buffer,
    ComputePass,
    FieldFlags,
    EmissivePowerSampler,
    EnvMapSampler,
    LightBVHSampler,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    SAMPLE_GENERATOR_TINY_UNIFORM,
    SampleGenerator,
    Sampler,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type ShaderVar,
} from "@web-falcor/falcor";

const kGeneratePathsFile = "RenderPasses/PathTracer/GeneratePaths.cs.slang";
const kTracePassFile = "RenderPasses/PathTracer/TracePass.rt.slang";
const kResolvePassFile = "RenderPasses/PathTracer/ResolvePass.cs.slang";

/** Mirrors EmissiveLightSamplerType (EmissiveLightSamplerType.slangh). */
const kEmissiveSamplerTypes: Record<string, number> = { Uniform: 0, LightBVH: 1, Power: 2 };

const kScreenTileDim = 16;

export class PathTracer extends RenderPass {
    private generatePass: ComputePass | null = null;
    private resolvePass: ComputePass | null = null;
    private outputGuideData = false;
    private sampleGuideData: Buffer | null = null;
    private tracePass: ComputePass | null = null;
    private frameCount = 0;
    private sampleGenerator: SampleGenerator;
    // Dummies for optional members that survive dead-code elimination
    // (viewDir/sampleCount/... are only live for other configurations).
    private dummyTexFloat: Texture | null = null;
    private dummyTexUint: Texture | null = null;
    private dummyBufferA: Buffer | null = null;
    private dummyBufferB: Buffer | null = null;
    private dummySampler: Sampler | null = null;
    private envMapSampler: EnvMapSampler | null = null;

    // StaticParams (defaults mirror PathTracer.h).
    private samplesPerPixel = 1;
    private maxSurfaceBounces = 3;
    private maxDiffuseBounces = 3;
    private maxSpecularBounces = 3;
    private maxTransmissionBounces = 10;
    private useBSDFSampling = true;
    private useRussianRoulette = false;
    private useNEE = true;
    private useMIS = true;
    private misHeuristic = 0; // Balance
    private useAlphaTest = true;
    private adjustShadingNormals = false;
    private emissiveSampler = "LightBVH"; // native default (PathTracer.h)
    private powerSampler: EmissivePowerSampler | null = null;
    private lightBVHSampler: LightBVHSampler | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.samplesPerPixel = props.get("samplesPerPixel", 1);
        this.maxSurfaceBounces = props.get("maxSurfaceBounces", 3);
        this.maxDiffuseBounces = props.get("maxDiffuseBounces", 3);
        this.maxSpecularBounces = props.get("maxSpecularBounces", 3);
        this.maxTransmissionBounces = props.get("maxTransmissionBounces", 10);
        this.useBSDFSampling = props.get("useBSDFSampling", true);
        this.useRussianRoulette = props.get("useRussianRoulette", false);
        this.useNEE = props.get("useNEE", true);
        this.useMIS = props.get("useMIS", true);
        this.useAlphaTest = props.get("useAlphaTest", true);
        this.emissiveSampler = props.get("emissiveSampler", "LightBVH");
        if (!(this.emissiveSampler in kEmissiveSamplerTypes)) {
            throw new Error(`PathTracer: unknown emissiveSampler '${this.emissiveSampler}'`);
        }
        // PathTracer defaults to TinyUniform (unlike MinimalPathTracer).
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_TINY_UNIFORM);
        if (this.samplesPerPixel !== 1) throw new Error("PathTracer v1 supports fixed spp == 1 only");
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Fullscreen V-buffer for the primary hits").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("color", "Output color (linear)")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        // Guide outputs (kOutputChannels formats), resolved from per-sample
        // guide data when connected. rayCount/pathLength need PixelStats
        // (pending); they allocate but stay zero.
        const guide: [string, string, ResourceFormat][] = [
            ["albedo", "Output albedo (linear)", ResourceFormat.RGBA8Unorm],
            ["specularAlbedo", "Output specular albedo (linear)", ResourceFormat.RGBA8Unorm],
            ["indirectAlbedo", "Output indirect albedo (linear)", ResourceFormat.RGBA8Unorm],
            ["guideNormal", "Output guide normal (linear)", ResourceFormat.RGBA16Float],
            ["reflectionPosW", "Output reflection pos (world space)", ResourceFormat.RGBA32Float],
            ["rayCount", "Per-pixel ray count", ResourceFormat.R32Uint],
            ["pathLength", "Per-pixel path length", ResourceFormat.R32Uint],
        ];
        for (const [name, desc, format] of guide) {
            r.addOutput(name, desc)
                .texture2D(w, h)
                .format(format)
                .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
                .flags(FieldFlags.Optional);
        }
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.generatePass = null;
        this.tracePass = null;
        this.frameCount = 0;
    }

    /** Mirrors PathTracer::StaticParams::getDefines. */
    private getStaticDefines() {
        const scene = this.scene!;
        return scene.getSceneDefines().addAll({
            SAMPLES_PER_PIXEL: this.samplesPerPixel,
            MAX_SURFACE_BOUNCES: this.maxSurfaceBounces,
            MAX_DIFFUSE_BOUNCES: this.maxDiffuseBounces,
            MAX_SPECULAR_BOUNCES: this.maxSpecularBounces,
            MAX_TRANSMISSON_BOUNCES: this.maxTransmissionBounces,
            ADJUST_SHADING_NORMALS: this.adjustShadingNormals ? 1 : 0,
            USE_BSDF_SAMPLING: this.useBSDFSampling ? 1 : 0,
            USE_NEE: this.useNEE ? 1 : 0,
            USE_MIS: this.useMIS ? 1 : 0,
            USE_RUSSIAN_ROULETTE: this.useRussianRoulette ? 1 : 0,
            USE_RTXDI: 0,
            USE_ALPHA_TEST: this.useAlphaTest ? 1 : 0,
            USE_LIGHTS_IN_DIELECTRIC_VOLUMES: 0,
            DISABLE_CAUSTICS: 0,
            PRIMARY_LOD_MODE: 0, // TexLODMode::Mip0
            USE_NRD_DEMODULATION: 1,
            USE_SER: 0,
            COLOR_FORMAT: 1, // ColorFormat::LogLuvHDR (native default; unused at spp==1)
            MIS_HEURISTIC: this.misHeuristic,
            MIS_POWER_EXPONENT: "2.0",
            _EMISSIVE_LIGHT_SAMPLER_TYPE: kEmissiveSamplerTypes[this.emissiveSampler]!,
            ...(this.lightBVHSampler ? Object.fromEntries(this.lightBVHSampler.getDefines().entries()) : {}),
            INTERIOR_LIST_SLOT_COUNT: 2,
            GBUFFER_ADJUST_SHADING_NORMALS: 0,
            USE_ENV_LIGHT: scene.useEnvLight ? 1 : 0,
            USE_ANALYTIC_LIGHTS: scene.useAnalyticLights ? 1 : 0,
            USE_EMISSIVE_LIGHTS: scene.useEmissiveLights ? 1 : 0,
            USE_CURVES: 0,
            USE_SDF_GRIDS: 0,
            USE_HAIR_MATERIAL: 0,
            USE_VIEW_DIR: 0,
            OUTPUT_GUIDE_DATA: this.outputGuideData ? 1 : 0,
            OUTPUT_NRD_DATA: 0,
            OUTPUT_NRD_ADDITIONAL_DATA: 0,
        }).addAll(this.sampleGenerator.getDefines());
    }

    /** Mirrors PathTracer::bindShaderData for the members live at spp == 1. */
    private bindPathTracerData(var_: ShaderVar, vbuffer: unknown, outputColor: unknown, frameDim: [number, number]): void {
        const tiles: [number, number] = [Math.ceil(frameDim[0] / kScreenTileDim), Math.ceil(frameDim[1] / kScreenTileDim)];
        const p = var_["params"] as ShaderVar;
        p["useFixedSeed"] = 0;
        p["fixedSeed"] = 1;
        p["lodBias"] = 0;
        p["specularRoughnessThreshold"] = 0.25;
        p["frameDim"] = frameDim;
        p["screenTiles"] = tiles;
        p["frameCount"] = this.frameCount;
        p["seed"] = this.frameCount; // seed = useFixedSeed ? fixedSeed : frameCount
        var_["vbuffer"] = vbuffer;
        var_["outputColor"] = outputColor;

        if (!this.dummyTexFloat) {
            const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            const tex = (format: ResourceFormat) =>
                new Texture(this.device, { type: ResourceType.Texture2D, width: 1, height: 1, arraySize: 1, mipLevels: 1, format, bindFlags: storage, name: "PathTracer::dummy" });
            this.dummyTexFloat = tex(ResourceFormat.RGBA32Float);
            this.dummyTexUint = tex(ResourceFormat.R32Uint);
            const buf = () => new Buffer(this.device, { size: 64, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "PathTracer::dummyBuf" });
            this.dummyBufferA = buf();
            this.dummyBufferB = buf();
            this.dummySampler = new Sampler(this.device, {});
        }
        var_["viewDir"] = this.dummyTexFloat;
        var_["sampleCount"] = this.dummyTexUint;
        var_["sampleOffset"] = this.dummyTexUint;
        var_["sampleColor"] = this.dummyBufferA;
        var_["sampleGuideData"] = this.outputGuideData ? this.sampleGuideData! : this.dummyBufferB;
    }

    /** Sets a member only if it survived DCE in this kernel variant. */
    private trySet(var_: ShaderVar, name: string, value: unknown): void {
        try {
            (var_ as Record<string, unknown>)[name] = value;
        } catch {
            /* binding absent in this variant */
        }
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const color = renderData.getTexture("color")!;
        const vbuffer = renderData.getTexture("vbuffer")!;
        const frameDim: [number, number] = [color.width, color.height];
        const tiles = [Math.ceil(frameDim[0] / kScreenTileDim), Math.ceil(frameDim[1] / kScreenTileDim)];

        // Mirrors PathTracer::beginFrame: guide data is produced when any of
        // the guide outputs is connected (drives OUTPUT_GUIDE_DATA).
        const kGuideOutputs = ["albedo", "specularAlbedo", "indirectAlbedo", "guideNormal", "reflectionPosW"] as const;
        const outputGuideData = kGuideOutputs.some((name) => renderData.getTexture(name) !== undefined);
        if (outputGuideData !== this.outputGuideData) {
            this.outputGuideData = outputGuideData;
            this.generatePass = null;
        }

        if (!this.generatePass) {
            if (this.emissiveSampler === "LightBVH" && this.scene.useEmissiveLights && !this.lightBVHSampler) {
                this.lightBVHSampler = new LightBVHSampler(this.device, this.scene.getEmissiveTriangles());
            }
            const defines = this.getStaticDefines();
            this.generatePass = ComputePass.create(this.device, { path: kGeneratePathsFile, defines });
            this.tracePass = ComputePass.create(this.device, { path: kTracePassFile, defines });
            this.resolvePass = this.outputGuideData ? ComputePass.create(this.device, { path: kResolvePassFile, defines }) : null;
        }

        // Per-sample guide data buffer (GuideData = uint4 + float3 + float = 32B),
        // one sample per pixel padded to whole screen tiles (spp == 1).
        if (this.outputGuideData) {
            const sampleCount = tiles[0]! * tiles[1]! * kScreenTileDim * kScreenTileDim;
            if (!this.sampleGuideData || this.sampleGuideData.size < sampleCount * 32) {
                this.sampleGuideData = new Buffer(this.device, {
                    size: sampleCount * 32,
                    structSize: 32,
                    bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                    memoryType: MemoryType.DeviceLocal,
                    name: "PathTracer::sampleGuideData",
                });
            }
        }

        // Generate paths: primary hits from the V-buffer; misses write background.
        {
            const root = this.generatePass.getRootVar();
            this.scene.bindShaderData(root);
            this.bindPathTracerData(root["CB"]["gPathGenerator"] as ShaderVar, vbuffer, color, frameDim);
            // One thread per pixel, padded to whole tiles (numthreads(256,1,1)).
            this.generatePass.execute(ctx, tiles[0]! * kScreenTileDim * kScreenTileDim, tiles[1]!);
        }

        // Trace paths (compute megakernel).
        {
            const root = this.tracePass!.getRootVar();
            this.scene.bindShaderData(root);
            const block = root["gPathTracer"] as ShaderVar;
            this.bindPathTracerData(block, vbuffer, color, frameDim);
            if (this.emissiveSampler === "Power" && this.scene.useEmissiveLights) {
                if (!this.powerSampler) this.powerSampler = new EmissivePowerSampler(this.device, this.scene.getEmissiveFluxes());
                this.powerSampler.bindShaderData(block["emissiveSampler"] as ShaderVar);
            }
            if (this.lightBVHSampler) this.lightBVHSampler.bindShaderData(block["emissiveSampler"] as ShaderVar);
            const envSampler = block["envMapSampler"] as ShaderVar;
            if (this.scene.useEnvLight) {
                if (!this.envMapSampler) this.envMapSampler = new EnvMapSampler(this.device, ctx, this.scene.getEnvMap()!);
                this.envMapSampler.bindShaderData(envSampler);
            } else {
                // Env map sampler members can survive DCE with env light off.
                envSampler["importanceSampler"] = this.dummySampler!;
                envSampler["importanceMap"] = this.dummyTexFloat!;
            }
            this.tracePass!.execute(ctx, frameDim[0], frameDim[1]);
        }

        // Resolve guide data into the connected outputs (mirrors resolvePass;
        // with fixed spp == 1 the color loop is compiled out).
        if (this.outputGuideData && this.resolvePass) {
            const root = this.resolvePass.getRootVar();
            const cb = root["CB"]!["gResolvePass"] as ShaderVar;
            const p = cb["params"] as ShaderVar;
            p["useFixedSeed"] = 0;
            p["fixedSeed"] = 1;
            p["lodBias"] = 0;
            p["specularRoughnessThreshold"] = 0.25;
            p["frameDim"] = frameDim;
            p["screenTiles"] = tiles;
            p["frameCount"] = this.frameCount;
            p["seed"] = this.frameCount;
            this.trySet(cb, "sampleGuideData", this.sampleGuideData!);
            this.trySet(cb, "sampleColor", this.dummyBufferA!);
            this.trySet(cb, "sampleCount", this.dummyTexUint!);
            this.trySet(cb, "sampleOffset", this.dummyTexUint!);
            // NRD-only members that survive DCE with OUTPUT_NRD_DATA=0.
            this.trySet(cb, "primaryHitDiffuseReflectance", this.dummyTexFloat!);
            this.trySet(cb, "sampleNRDRadiance", this.dummyBufferA!);
            this.trySet(cb, "sampleNRDHitDist", this.dummyBufferA!);
            this.trySet(cb, "sampleNRDEmission", this.dummyBufferA!);
            this.trySet(cb, "sampleNRDReflectance", this.dummyBufferA!);
            this.trySet(cb, "sampleNRDPrimaryHitNeeOnDelta", this.dummyBufferA!);
            for (const name of ["albedo", "specularAlbedo", "indirectAlbedo", "guideNormal", "reflectionPosW"]) {
                const key = `output${name[0]!.toUpperCase()}${name.slice(1)}`;
                this.trySet(cb, key, renderData.getTexture(name) ?? this.dummyTexFloat!);
            }
            this.trySet(cb, "outputColor", this.dummyTexFloat!);
            this.resolvePass.execute(ctx, frameDim[0], frameDim[1]);
        }

        this.frameCount++;
    }
}

registerRenderPass("PathTracer", (device, props) => new PathTracer(device, props));
