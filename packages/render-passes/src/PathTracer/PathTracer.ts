/**
 * Reference path tracer mirroring Source/RenderPasses/PathTracer (host side of
 * GeneratePaths + TracePass; the resolve pass is skipped for fixed spp == 1,
 * as native does). TracePass runs as the WebFalcor compute megakernel over
 * software ray queries (see the TracePass.rt.slang override).
 *
 * Limits (documented in docs parity matrix): fixed spp == 1 only (no
 * adaptive sampleCount input), guide outputs supported via ResolvePass, no
 * NRD outputs, no RTXDI/SER; rayCount/pathLength await the PixelStats port.
 */

import {
    PixelDebug,
    PixelStats,
    Buffer,
    ComputePass,
    FieldFlags,
    EmissivePowerSampler,
    EnvMapSampler,
    LightBVHSampler,
    RTXDI,
    kDefaultLightBVHSamplerOptions,
    kDefaultLightBVHOptions,
    kSolidAngleBoundMethods,
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
    type UIWidgets,
    GeometryType,
    Logger,
} from "@web-falcor/falcor";

const kGeneratePathsFile = "RenderPasses/PathTracer/GeneratePaths.cs.slang";
const kTracePassFile = "RenderPasses/PathTracer/TracePass.rt.slang";
const kResolvePassFile = "RenderPasses/PathTracer/ResolvePass.cs.slang";

/** Mirrors EmissiveLightSamplerType (EmissiveLightSamplerType.slangh). */
const kEmissiveSamplerTypes: Record<string, number> = { Uniform: 0, LightBVH: 1, Power: 2 };

const kScreenTileDim = 16;

/**
 * NRD outputs (native kOutputChannels). §9: RGB10A2Unorm and R16Float aren't storage formats in
 * WebGPU; those outputs use RGBA16Float / R32Float like GBufferRT's normWRoughnessMaterialID.
 * Resolve-side outputs are written by the resolve pass, the others by the path tracer.
 */
const kNRDOutputs: [name: string, member: string, desc: string, format: ResourceFormat, additional: boolean, resolved: boolean][] = [
    ["nrdDiffuseRadianceHitDist", "outputNRDDiffuseRadianceHitDist", "Output demodulated diffuse color (linear) and hit distance", ResourceFormat.RGBA32Float, false, true],
    ["nrdSpecularRadianceHitDist", "outputNRDSpecularRadianceHitDist", "Output demodulated specular color (linear) and hit distance", ResourceFormat.RGBA32Float, false, true],
    ["nrdEmission", "primaryHitEmission", "Output primary surface emission", ResourceFormat.RGBA32Float, false, false],
    ["nrdDiffuseReflectance", "primaryHitDiffuseReflectance", "Output primary surface diffuse reflectance", ResourceFormat.RGBA16Float, false, false],
    ["nrdSpecularReflectance", "primaryHitSpecularReflectance", "Output primary surface specular reflectance", ResourceFormat.RGBA16Float, false, false],
    ["nrdDeltaReflectionRadianceHitDist", "outputNRDDeltaReflectionRadianceHitDist", "Output demodulated delta reflection color (linear)", ResourceFormat.RGBA32Float, true, true],
    ["nrdDeltaReflectionReflectance", "deltaReflectionReflectance", "Output delta reflection reflectance color (linear)", ResourceFormat.RGBA16Float, true, false],
    ["nrdDeltaReflectionEmission", "deltaReflectionEmission", "Output delta reflection emission color (linear)", ResourceFormat.RGBA32Float, true, false],
    ["nrdDeltaReflectionNormWRoughMaterialID", "deltaReflectionNormWRoughMaterialID", "Output delta reflection world normal, roughness, and material ID", ResourceFormat.RGBA16Float, true, false],
    ["nrdDeltaReflectionPathLength", "deltaReflectionPathLength", "Output delta reflection path length", ResourceFormat.R32Float, true, false],
    ["nrdDeltaReflectionHitDist", "deltaReflectionHitDist", "Output delta reflection hit distance", ResourceFormat.R32Float, true, false],
    ["nrdDeltaTransmissionRadianceHitDist", "outputNRDDeltaTransmissionRadianceHitDist", "Output demodulated delta transmission color (linear)", ResourceFormat.RGBA32Float, true, true],
    ["nrdDeltaTransmissionReflectance", "deltaTransmissionReflectance", "Output delta transmission reflectance color (linear)", ResourceFormat.RGBA16Float, true, false],
    ["nrdDeltaTransmissionEmission", "deltaTransmissionEmission", "Output delta transmission emission color (linear)", ResourceFormat.RGBA32Float, true, false],
    ["nrdDeltaTransmissionNormWRoughMaterialID", "deltaTransmissionNormWRoughMaterialID", "Output delta transmission world normal, roughness, and material ID", ResourceFormat.RGBA16Float, true, false],
    ["nrdDeltaTransmissionPathLength", "deltaTransmissionPathLength", "Output delta transmission path length", ResourceFormat.R32Float, true, false],
    ["nrdDeltaTransmissionPosW", "deltaTransmissionPosW", "Output delta transmission position", ResourceFormat.RGBA32Float, true, false],
    ["nrdResidualRadianceHitDist", "outputNRDResidualRadianceHitDist", "Output residual color (linear) and hit distance", ResourceFormat.RGBA32Float, false, true],
];
/** The primary-hit outputs that drive OUTPUT_NRD_DATA (native beginFrame); the rest drive OUTPUT_NRD_ADDITIONAL_DATA. */
const kNRDDataOutputs = ["nrdDiffuseRadianceHitDist", "nrdSpecularRadianceHitDist", "nrdResidualRadianceHitDist", "nrdEmission", "nrdDiffuseReflectance", "nrdSpecularReflectance"];
/** NRDSampleData stride in WGSL (NRDBuffers override): NRDRadiance is 32 bytes (float3 aligns to 16). */
const kNRDSampleDataSize = 96;
const kNoRegion = 0xffffffff;
/** UI range for the bounce sliders (Params.slang kMaxBounces). */
const kMaxBounces = 254;

export class PathTracer extends RenderPass {
    private generatePass: ComputePass | null = null;
    private resolvePass: ComputePass | null = null;
    private outputGuideData = false;
    private outputNRDData = false;
    private outputNRDAdditionalData = false;
    private useNRDDemodulation = true;
    /** Pixel-buffer region per connected NRD output (NRDBuffers override). */
    private nrdRegions = new Map<string, number>();
    private nrdSampleData: Buffer | null = null;
    private nrdPixelData: Buffer | null = null;
    private nrdCopyPasses = new Map<string, ComputePass>();
    private fixedSampleCount = true;
    private sampleGuideData: Buffer | null = null;
    private sampleColor: Buffer | null = null;
    private sampleOffset: Texture | null = null;
    private sampleCountInput: Texture | null = null;
    private statsEnabled = false;
    private viewDirInput: Texture | null = null;
    private useViewDir = false;
    private statsBuffer: Buffer | null = null;
    private statsResolvePass: ComputePass | null = null;
    private pixelStats: PixelStats | null = null;
    /** Mirrors mpPixelDebug (shader print()/assert() for one selected pixel; UI in "Debugging"). */
    readonly pixelDebug = new PixelDebug(this.device);
    private tracePass: ComputePass | null = null;
    /** Mirrors mpTraceDeltaReflectionPass / mpTraceDeltaTransmissionPass (NRD additional data). */
    private traceDeltaReflectionPass: ComputePass | null = null;
    private traceDeltaTransmissionPass: ComputePass | null = null;
    private frameCount = 0;
    private sampleGenerator: SampleGenerator;
    // Dummies for optional members that survive dead-code elimination
    // (viewDir/sampleCount/... are only live for other configurations).
    private dummyTexFloat: Texture | null = null;
    private dummyTexUint: Texture | null = null;
    private outputDummies = new Map<string, Texture>();
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
    /** Scene.emissiveVersion the LightBVH was built/refit for (native LightCollection update flags). */
    private lightBVHVersion = -1;
    private lightBVHOptions = kDefaultLightBVHSamplerOptions;
    private primaryLodMode = 0; // TexLODMode::Mip0
    private useRTXDI = false;
    private rtxdiOptions: Record<string, unknown> = {};
    private rtxdi: RTXDI | null = null;
    private dummyMvec: import("@web-falcor/falcor").Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.samplesPerPixel = props.get("samplesPerPixel", 1);
        this.useNRDDemodulation = props.get("useNRDDemodulation", true);
        this.maxSurfaceBounces = props.get("maxSurfaceBounces", 3);
        this.maxDiffuseBounces = props.get("maxDiffuseBounces", 3);
        this.maxSpecularBounces = props.get("maxSpecularBounces", 3);
        this.maxTransmissionBounces = props.get("maxTransmissionBounces", 10);
        // Mirrors native: maxSurfaceBounces initializes the per-lobe limits
        // when they aren't explicitly set; otherwise it is raised to their max.
        if (props.has("maxSurfaceBounces")) {
            if (!props.has("maxDiffuseBounces")) this.maxDiffuseBounces = this.maxSurfaceBounces;
            if (!props.has("maxSpecularBounces")) this.maxSpecularBounces = this.maxSurfaceBounces;
            if (!props.has("maxTransmissionBounces")) this.maxTransmissionBounces = this.maxSurfaceBounces;
        } else {
            this.maxSurfaceBounces = Math.max(this.maxDiffuseBounces, this.maxSpecularBounces, this.maxTransmissionBounces);
        }
        this.useBSDFSampling = props.get("useBSDFSampling", true);
        this.useRussianRoulette = props.get("useRussianRoulette", false);
        this.useNEE = props.get("useNEE", true);
        this.useMIS = props.get("useMIS", true);
        this.useAlphaTest = props.get("useAlphaTest", true);
        this.emissiveSampler = props.get("emissiveSampler", "LightBVH");
        if (!(this.emissiveSampler in kEmissiveSamplerTypes)) {
            throw new Error(`PathTracer: unknown emissiveSampler '${this.emissiveSampler}'`);
        }
        // Mirrors kUseRTXDI / kRTXDIOptions.
        this.useRTXDI = props.get("useRTXDI", false);
        this.rtxdiOptions = (props.getOpt("RTXDIOptions") as Record<string, unknown> | undefined) ?? {};
        // Mirrors kPrimaryLodMode (TexLODMode): Mip0 or RayDiffs; native also
        // rejects RayCones here ("Unsupported tex lod mode. Defaulting to Mip0.").
        const lodModes: Record<string, number> = { Mip0: 0, RayCones: 1, RayDiffs: 2 };
        const lod = props.getOpt<string | number>("primaryLodMode");
        if (lod !== undefined) {
            this.primaryLodMode = (typeof lod === "string" ? lodModes[lod] : lod) ?? 0;
            if (this.primaryLodMode === 1) {
                console.warn("PathTracer: unsupported tex lod mode. Defaulting to Mip0.");
                this.primaryLodMode = 0;
            }
        }
        // Mirrors kLightBVHOptions: nested dict following the native serialization keys.
        const bvhOpts = props.getOpt("lightBVHOptions") as Record<string, unknown> | undefined;
        if (bvhOpts) {
            const build = (bvhOpts["buildOptions"] ?? {}) as Record<string, unknown>;
            const split = build["splitHeuristicSelection"];
            const bound = bvhOpts["solidAngleBoundMethod"];
            this.lightBVHOptions = {
                ...kDefaultLightBVHSamplerOptions,
                ...Object.fromEntries(Object.entries(bvhOpts).filter(([k]) => k in kDefaultLightBVHSamplerOptions && k !== "buildOptions" && k !== "solidAngleBoundMethod")),
                solidAngleBoundMethod:
                    typeof bound === "string" ? (kSolidAngleBoundMethods[bound] ?? kDefaultLightBVHSamplerOptions.solidAngleBoundMethod) : typeof bound === "number" ? bound : kDefaultLightBVHSamplerOptions.solidAngleBoundMethod,
                buildOptions: {
                    ...kDefaultLightBVHOptions,
                    ...Object.fromEntries(Object.entries(build).filter(([k]) => k in kDefaultLightBVHOptions && k !== "splitHeuristicSelection")),
                    splitHeuristicSelection:
                        split === "Equal" || split === "BinnedSAH" || split === "BinnedSAOH" ? split : kDefaultLightBVHOptions.splitHeuristicSelection,
                },
            };
        }
        // PathTracer defaults to TinyUniform (unlike MinimalPathTracer).
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_TINY_UNIFORM);
        // Native asserts spp fits the 16-bit tile sample offsets (tile 16x16 x 16 spp).
        if (this.samplesPerPixel < 1 || this.samplesPerPixel > 16) throw new Error("PathTracer: samplesPerPixel must be in [1,16]");
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Fullscreen V-buffer for the primary hits").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("viewW", "World-space view direction (xyz float format)")
            .bindFlags(ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        r.addInput("mvec", "Motion vector buffer (float format; RTXDI temporal resampling)")
            .bindFlags(ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        r.addInput("sampleCount", "Sample count buffer (integer format)")
            .format(ResourceFormat.R8Uint)
            .bindFlags(ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
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
        for (const [name, desc, format] of [...guide, ...kNRDOutputs.map(([n, , d, f]) => [n, d, f] as const)]) {
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
        // Mirrors native: custom primitives need an intersection shader this pass lacks.
        if (scene?.hasGeometryType(GeometryType.Custom)) Logger.warning("PathTracer: This render pass does not support custom primitives.");
        this.generatePass = null;
        this.tracePass = null;
        this.rtxdi = null;
        this.frameCount = 0;
        // Mirrors resetLighting(): light samplers were built from the previous scene.
        this.lightBVHSampler = null;
        this.powerSampler = null;
        this.envMapSampler = null;
    }

    /** Mirrors PathTracer::reset (script binding): restarts the frame count. */
    reset(): void {
        this.frameCount = 0;
    }

    /** Mirrors PathTracer::getProperties (StaticParams + sampler options). */
    override getProperties(): Properties {
        const lodNames = ["Mip0", "RayCones", "RayDiffs"];
        return new Properties({
            samplesPerPixel: this.samplesPerPixel,
            useNRDDemodulation: this.useNRDDemodulation,
            maxSurfaceBounces: this.maxSurfaceBounces,
            maxDiffuseBounces: this.maxDiffuseBounces,
            maxSpecularBounces: this.maxSpecularBounces,
            maxTransmissionBounces: this.maxTransmissionBounces,
            useBSDFSampling: this.useBSDFSampling,
            useRussianRoulette: this.useRussianRoulette,
            useNEE: this.useNEE,
            useMIS: this.useMIS,
            misHeuristic: this.misHeuristic,
            emissiveSampler: this.emissiveSampler,
            ...(this.emissiveSampler === "LightBVH" ? { lightBVHOptions: this.lightBVHOptions as unknown as Record<string, never> } : {}),
            useRTXDI: this.useRTXDI,
            RTXDIOptions: this.rtxdiOptions as Record<string, never>,
            useAlphaTest: this.useAlphaTest,
            adjustShadingNormals: this.adjustShadingNormals,
            primaryLodMode: lodNames[this.primaryLodMode] ?? this.primaryLodMode,
        });
    }

    /** Static params are shader defines: drop the kernels so execute() rebuilds them (native mRecompile). */
    private recreatePrograms(): void {
        this.generatePass = null;
        this.tracePass = null;
        this.resolvePass = null;
    }

    /** Mirrors PathTracer::renderUI (rendering options; stats live in getPixelStats()). */
    override renderUI(ui: UIWidgets): void {
        const rebuild = <T>(set: (v: T) => void) => (v: T) => {
            set(v);
            this.recreatePrograms();
        };
        if (this.fixedSampleCount) ui.slider("Samples/pixel", this.samplesPerPixel, 1, 16, 1, rebuild((v) => (this.samplesPerPixel = Math.round(v))));
        else ui.text("Samples/pixel: Variable");
        ui.slider("Max surface bounces", this.maxSurfaceBounces, 0, kMaxBounces, 1, rebuild((v) => {
            // Clamps the per-lobe limits like native.
            this.maxSurfaceBounces = Math.round(v);
            this.maxDiffuseBounces = Math.min(this.maxDiffuseBounces, this.maxSurfaceBounces);
            this.maxSpecularBounces = Math.min(this.maxSpecularBounces, this.maxSurfaceBounces);
            this.maxTransmissionBounces = Math.min(this.maxTransmissionBounces, this.maxSurfaceBounces);
        }));
        ui.slider("Max diffuse bounces", this.maxDiffuseBounces, 0, kMaxBounces, 1, rebuild((v) => (this.maxDiffuseBounces = Math.round(v))));
        ui.slider("Max specular bounces", this.maxSpecularBounces, 0, kMaxBounces, 1, rebuild((v) => (this.maxSpecularBounces = Math.round(v))));
        ui.slider("Max transmission bounces", this.maxTransmissionBounces, 0, kMaxBounces, 1, rebuild((v) => (this.maxTransmissionBounces = Math.round(v))));
        ui.checkbox("BSDF importance sampling", this.useBSDFSampling, rebuild((v) => (this.useBSDFSampling = v)));
        ui.checkbox("Russian roulette", this.useRussianRoulette, rebuild((v) => (this.useRussianRoulette = v)));
        ui.checkbox("Next-event estimation (NEE)", this.useNEE, rebuild((v) => (this.useNEE = v)));
        ui.checkbox("Multiple importance sampling (MIS)", this.useMIS, rebuild((v) => (this.useMIS = v)));
        ui.dropdown("MIS heuristic", ["Balance", "PowerTwo", "PowerExp"], ["Balance", "PowerTwo", "PowerExp"][this.misHeuristic] ?? "Balance", rebuild((v: string) => (this.misHeuristic = ["Balance", "PowerTwo", "PowerExp"].indexOf(v))));
        ui.dropdown("Emissive sampler", Object.keys(kEmissiveSamplerTypes), this.emissiveSampler, rebuild((v: string) => {
            this.emissiveSampler = v;
            this.lightBVHSampler = null;
            this.powerSampler = null;
        }));
        ui.checkbox("Use RTXDI", this.useRTXDI, rebuild((v) => {
            this.useRTXDI = v;
            if (!v) this.rtxdi = null;
        }));
        ui.checkbox("Alpha test", this.useAlphaTest, rebuild((v) => (this.useAlphaTest = v)));
        ui.checkbox("Adjust shading normals on secondary hits", this.adjustShadingNormals, rebuild((v) => (this.adjustShadingNormals = v)));
        ui.dropdown("Primary LOD Mode", ["Mip0", "RayDiffs"], this.primaryLodMode === 2 ? "RayDiffs" : "Mip0", rebuild((v: string) => (this.primaryLodMode = v === "RayDiffs" ? 2 : 0)));
        this.pixelDebug.renderUI(ui.group("Debugging"), () => this.recreatePrograms());
    }

    /** Mirrors PathTracer::onMouseEvent (pixel debug selection). */
    onMouseEvent(ev: { type: "buttonDown" | "buttonUp" | "move"; button?: "left" | "right" | "middle"; pos: [number, number] }): boolean {
        return this.pixelDebug.onMouseEvent(ev);
    }

    /** Mirrors PathTracer::StaticParams::getDefines. */
    private getStaticDefines() {
        const scene = this.scene!;
        return scene.getSceneDefines().addAll({
            SAMPLES_PER_PIXEL: this.fixedSampleCount ? this.samplesPerPixel : 0,
            MAX_SURFACE_BOUNCES: this.maxSurfaceBounces,
            MAX_DIFFUSE_BOUNCES: this.maxDiffuseBounces,
            MAX_SPECULAR_BOUNCES: this.maxSpecularBounces,
            MAX_TRANSMISSON_BOUNCES: this.maxTransmissionBounces,
            ADJUST_SHADING_NORMALS: this.adjustShadingNormals ? 1 : 0,
            USE_BSDF_SAMPLING: this.useBSDFSampling ? 1 : 0,
            USE_NEE: this.useNEE ? 1 : 0,
            USE_MIS: this.useMIS ? 1 : 0,
            USE_RUSSIAN_ROULETTE: this.useRussianRoulette ? 1 : 0,
            USE_RTXDI: this.rtxdi ? 1 : 0,
            USE_ALPHA_TEST: this.useAlphaTest ? 1 : 0,
            USE_LIGHTS_IN_DIELECTRIC_VOLUMES: 0,
            DISABLE_CAUSTICS: 0,
            PRIMARY_LOD_MODE: this.primaryLodMode,
            USE_NRD_DEMODULATION: this.useNRDDemodulation ? 1 : 0,
            USE_SER: 0,
            COLOR_FORMAT: 1, // ColorFormat::LogLuvHDR (native default; unused at spp==1)
            MIS_HEURISTIC: this.misHeuristic,
            MIS_POWER_EXPONENT: "2.0",
            // Mirrors native: the sampler-type define comes from the emissive
            // sampler's own getDefines(); without emissive lights there is no
            // sampler and the shader falls back to the NULL sampler.
            ...(scene.useEmissiveLights
                ? this.lightBVHSampler
                    ? Object.fromEntries(this.lightBVHSampler.getDefines().entries())
                    : { _EMISSIVE_LIGHT_SAMPLER_TYPE: kEmissiveSamplerTypes[this.emissiveSampler]! }
                : {}),
            INTERIOR_LIST_SLOT_COUNT: 2,
            GBUFFER_ADJUST_SHADING_NORMALS: 0,
            USE_ENV_LIGHT: scene.useEnvLight ? 1 : 0,
            USE_ANALYTIC_LIGHTS: scene.useAnalyticLights ? 1 : 0,
            USE_EMISSIVE_LIGHTS: scene.useEmissiveLights ? 1 : 0,
            USE_CURVES: scene.hasCurves ? 1 : 0,
            USE_SDF_GRIDS: 0,
            USE_HAIR_MATERIAL: 0,
            USE_VIEW_DIR: this.useViewDir ? 1 : 0,
            OUTPUT_GUIDE_DATA: this.outputGuideData ? 1 : 0,
            OUTPUT_NRD_DATA: this.outputNRDData ? 1 : 0,
            OUTPUT_NRD_ADDITIONAL_DATA: this.outputNRDAdditionalData ? 1 : 0,
            ...(this.statsEnabled ? { _PIXEL_STATS_ENABLED: 1 } : {}),
            ...(this.pixelDebug.enabled ? PixelDebug.getDefines() : {}),
        })
            .addAll(this.sampleGenerator.getDefines())
            .addAll(this.rtxdi ? this.rtxdi.getDefines() : {});
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
        var_["viewDir"] = this.viewDirInput ?? this.dummyTexFloat;
        var_["sampleCount"] = this.sampleCountInput ?? this.dummyTexUint;
        var_["sampleOffset"] = this.sampleOffset ?? this.dummyTexUint;
        var_["sampleColor"] = this.sampleColor ?? this.dummyBufferA;
        var_["sampleGuideData"] = this.outputGuideData ? this.sampleGuideData! : this.dummyBufferB;
    }

    /**
     * Binds the NRDBuffers views (region, width, pixel count per output) and the two shared buffers.
     * `member` is the NRDBuffers field of `var_`, or null for the resolve pass's flat members.
     */
    private bindNRD(root: ShaderVar, var_: ShaderVar, member: string | null, frameDim: [number, number]): void {
        this.trySet(root, "gNRDSampleData", this.nrdSampleData ?? this.dummyBufferA!);
        this.trySet(root, "gNRDPixelData", this.nrdPixelData ?? this.dummyBufferB!);
        let target: ShaderVar;
        try {
            target = member ? (var_[member] as ShaderVar) : var_;
        } catch {
            return; // NRD members stripped in this variant
        }
        for (const [name, field] of kNRDOutputs) {
            const view = { region: this.nrdRegions.get(name) ?? kNoRegion, width: frameDim[0], pixelCount: frameDim[0] * frameDim[1] };
            try {
                const v = target[field] as ShaderVar;
                v["region"] = view.region;
                v["width"] = view.width;
                v["pixelCount"] = view.pixelCount;
            } catch {
                /* not a member of this kernel (tracer vs resolve outputs) or stripped */
            }
        }
    }

    /** Copies the connected NRD outputs out of the pixel buffer, 8 storage textures per dispatch. */
    private copyNRDOutputs(ctx: RenderContext, renderData: RenderData, frameDim: [number, number]): void {
        const outputs = kNRDOutputs.filter(([name]) => renderData.getTexture(name) !== undefined);
        const kFormat: Partial<Record<ResourceFormat, [string, string]>> = {
            [ResourceFormat.RGBA32Float]: ["rgba32f", "float4"],
            [ResourceFormat.RGBA16Float]: ["rgba16f", "float4"],
            [ResourceFormat.R32Float]: ["r32f", "float"],
        };
        for (let first = 0; first < outputs.length; first += 8) {
            const chunk = outputs.slice(first, first + 8);
            const key = chunk.map(([, , , f]) => f).join(",");
            let pass = this.nrdCopyPasses.get(key);
            if (!pass) {
                const decls = chunk.map(([, , , f], i) => `[format("${kFormat[f]![0]}")] RWTexture2D<${kFormat[f]![1]}> gOut${i};`).join("\n");
                const writes = chunk
                    .map(([, , , f], i) => `    if (${i} < gCount) { float4 v = gNRDPixelData[gRegion[${i >> 2}][${i & 3}] * pixels + p.y * gFrameDim.x + p.x]; gOut${i}[p] = ${kFormat[f]![1] === "float" ? "v.x" : "v"}; }`)
                    .join("\n");
                const source = `StructuredBuffer<float4> gNRDPixelData;\n${decls}\ncbuffer CB { uint2 gFrameDim; uint gCount; uint4 gRegion[2]; };\n[numthreads(16, 16, 1)]\nvoid main(uint3 id: SV_DispatchThreadID)\n{\n    const uint2 p = id.xy;\n    if (any(p >= gFrameDim)) return;\n    const uint pixels = gFrameDim.x * gFrameDim.y;\n${writes}\n}\n`;
                pass = ComputePass.create(this.device, { modules: [{ sources: [{ string: source, path: "WebFalcor/NRDOutputCopy.cs.slang" }] }], csEntry: "main" });
                this.nrdCopyPasses.set(key, pass);
            }
            const root = pass.getRootVar();
            root["gNRDPixelData"] = this.nrdPixelData!;
            const cb = root["CB"] as ShaderVar;
            cb["gFrameDim"] = frameDim;
            cb["gCount"] = chunk.length;
            const regions = chunk.map(([name]) => this.nrdRegions.get(name)!);
            while (regions.length < 8) regions.push(0);
            cb["gRegion"] = [regions.slice(0, 4), regions.slice(4, 8)];
            chunk.forEach(([name], i) => (root[`gOut${i}`] = renderData.getTexture(name)!));
            pass.execute(ctx, frameDim[0], frameDim[1]);
        }
    }

    /** Binds the pixel-stats globals (module scope, present when enabled). */
    private bindStats(root: ShaderVar, frameDim: [number, number]): void {
        if (!this.statsEnabled) return;
        try {
            (root["PixelStatsCB"] as ShaderVar)["gPixelStatsDim"] = frameDim;
        } catch {
            /* stats DCE'd in this kernel */
        }
        this.trySet(root, "gStatsBuffer", this.statsBuffer!);
    }

    /** Sets a member only if it survived DCE in this kernel variant. */
    private trySet(var_: ShaderVar, name: string, value: unknown): void {
        try {
            (var_ as Record<string, unknown>)[name] = value;
        } catch {
            /* binding absent in this variant */
        }
    }

    /** Mirrors PixelStats::getStats (valid once stats-collecting frames resolve). */
    getPixelStats(): import("@web-falcor/falcor").PixelStatsAggregate {
        return (this.pixelStats ?? new PixelStats(this.device)).getStats();
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const color = renderData.getTexture("color")!;
        const vbuffer = renderData.getTexture("vbuffer")!;
        const frameDim: [number, number] = [color.width, color.height];
        const tiles = [Math.ceil(frameDim[0] / kScreenTileDim), Math.ceil(frameDim[1] / kScreenTileDim)];
        this.pixelDebug.beginFrame(ctx, frameDim);

        // Mirrors PathTracer::beginFrame: guide data is produced when any of
        // the guide outputs is connected (drives OUTPUT_GUIDE_DATA).
        const kGuideOutputs = ["albedo", "specularAlbedo", "indirectAlbedo", "guideNormal", "reflectionPosW"] as const;
        const outputGuideData = kGuideOutputs.some((name) => renderData.getTexture(name) !== undefined);
        // Mirrors PathTracer::beginFrame: the sample count is variable when the
        // sampleCount input is connected (SAMPLES_PER_PIXEL define becomes 0).
        this.sampleCountInput = renderData.getTexture("sampleCount") ?? null;
        const fixedSampleCount = this.sampleCountInput === null;
        // Mirrors native: pixel stats collect when rayCount/pathLength connect.
        const statsEnabled = renderData.getTexture("rayCount") !== undefined || renderData.getTexture("pathLength") !== undefined;
        // Mirrors native USE_VIEW_DIR: DoF needs the per-pixel thin-lens dirs from the viewW input.
        this.viewDirInput = renderData.getTexture("viewW") ?? null;
        const useViewDir = this.scene.camera.getApertureRadius() > 0 && this.viewDirInput !== null;
        // Mirrors beginFrame's mOutputNRDData / mOutputNRDAdditionalData.
        const outputNRDData = kNRDDataOutputs.some((name) => renderData.getTexture(name) !== undefined);
        const outputNRDAdditionalData = kNRDOutputs.some(([name, , , , additional]) => additional && renderData.getTexture(name) !== undefined);
        this.nrdRegions = new Map(kNRDOutputs.filter(([name]) => renderData.getTexture(name) !== undefined).map(([name], i, arr) => [name, arr.findIndex(([n]) => n === name)]));
        // The resolve reads the primary-hit diffuse reflectance, so it always gets a region with NRD data.
        if (outputNRDData && !this.nrdRegions.has("nrdDiffuseReflectance")) this.nrdRegions.set("nrdDiffuseReflectance", this.nrdRegions.size);
        if (
            outputGuideData !== this.outputGuideData ||
            outputNRDData !== this.outputNRDData ||
            outputNRDAdditionalData !== this.outputNRDAdditionalData ||
            fixedSampleCount !== this.fixedSampleCount ||
            statsEnabled !== this.statsEnabled ||
            useViewDir !== this.useViewDir
        ) {
            this.outputGuideData = outputGuideData;
            this.outputNRDData = outputNRDData;
            this.outputNRDAdditionalData = outputNRDAdditionalData;
            this.fixedSampleCount = fixedSampleCount;
            this.statsEnabled = statsEnabled;
            this.useViewDir = useViewDir;
            this.generatePass = null;
        }

        // Mirrors LightBVHSampler::update on LightCollection MatrixChanged: refit the
        // tree in place (allowRefitting), rebuilding only if the triangle set changed.
        if (this.lightBVHSampler && this.scene.emissiveVersion !== this.lightBVHVersion) {
            const tris = this.scene.getEmissiveTriangles();
            if (!(this.lightBVHOptions.buildOptions.allowRefitting && this.lightBVHSampler.refit(tris))) {
                this.lightBVHSampler = new LightBVHSampler(this.device, tris, this.lightBVHOptions);
                this.generatePass = null; // defines may change with the tree size
            }
            this.lightBVHVersion = this.scene.emissiveVersion;
            this.rtxdi?.notifyLightsChanged();
        } else if (this.rtxdi && this.scene.emissiveVersion !== this.lightBVHVersion) {
            this.lightBVHVersion = this.scene.emissiveVersion;
            this.rtxdi.notifyLightsChanged();
        }

        // Mirrors prepareRTXDI: create before the programs so USE_RTXDI + the
        // RTXDI defines land in the kernels.
        if (this.useRTXDI && !this.rtxdi) {
            this.rtxdi = new RTXDI(this.device, this.scene, this.rtxdiOptions);
            this.generatePass = null;
            if (!this.fixedSampleCount || this.samplesPerPixel !== 1) {
                console.warn("Using RTXDI with samples/pixel != 1 will only generate one RTXDI sample reused for all pixel samples.");
            }
        }

        if (!this.generatePass) {
            if (this.emissiveSampler === "LightBVH" && this.scene.useEmissiveLights && !this.lightBVHSampler) {
                this.lightBVHSampler = new LightBVHSampler(this.device, this.scene.getEmissiveTriangles(), this.lightBVHOptions);
                this.lightBVHVersion = this.scene.emissiveVersion;
            }
            const defines = this.getStaticDefines();
            this.generatePass = ComputePass.create(this.device, { path: kGeneratePathsFile, defines });
            this.tracePass = ComputePass.create(this.device, { path: kTracePassFile, defines });
            this.traceDeltaReflectionPass = this.outputNRDAdditionalData ? ComputePass.create(this.device, { path: kTracePassFile, defines: defines.clone().add("DELTA_REFLECTION_PASS", "") }) : null;
            this.traceDeltaTransmissionPass = this.outputNRDAdditionalData ? ComputePass.create(this.device, { path: kTracePassFile, defines: defines.clone().add("DELTA_TRANSMISSION_PASS", "") }) : null;
            this.resolvePass =
                this.outputGuideData || this.outputNRDData || !this.fixedSampleCount || this.samplesPerPixel > 1
                    ? ComputePass.create(this.device, { path: kResolvePassFile, defines })
                    : null;
        }

        // Per-sample buffers, padded to whole screen tiles; variable mode
        // budgets kMaxSamplesPerPixel = 16 samples per pixel (native assert:
        // tile 16x16 x 16 spp fits 16-bit sample offsets).
        const sppBudget = this.fixedSampleCount ? this.samplesPerPixel : 16;
        const sampleCount = tiles[0]! * tiles[1]! * kScreenTileDim * kScreenTileDim * sppBudget;
        if (this.outputGuideData) {
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
        if (this.outputNRDData || this.outputNRDAdditionalData) {
            const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            if (!this.nrdSampleData || this.nrdSampleData.size < sampleCount * kNRDSampleDataSize) {
                this.nrdSampleData = new Buffer(this.device, { size: sampleCount * kNRDSampleDataSize, structSize: kNRDSampleDataSize, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "PathTracer::nrdSampleData" });
            }
            const pixelBytes = Math.max(1, this.nrdRegions.size) * frameDim[0] * frameDim[1] * 16;
            if (!this.nrdPixelData || this.nrdPixelData.size < pixelBytes) {
                this.nrdPixelData = new Buffer(this.device, { size: pixelBytes, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "PathTracer::nrdPixelData" });
            }
        }
        if (this.statsEnabled) {
            // One packed buffer, 5 per-pixel counter regions (storage-buffer
            // count per stage is limited; see the PixelStats override).
            const statsSize = frameDim[0] * frameDim[1] * 5 * 4;
            if (!this.statsBuffer || this.statsBuffer.size < statsSize) {
                this.statsBuffer = new Buffer(this.device, {
                    size: statsSize,
                    structSize: 4,
                    bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                    memoryType: MemoryType.DeviceLocal,
                    name: "PathTracer::stats",
                });
            }
            // Mirrors PixelStats::beginFrame: stats are per-frame.
            ctx.clearBuffer(this.statsBuffer);
        }

        if (!this.fixedSampleCount || this.samplesPerPixel > 1) {
            // ColorType at COLOR_FORMAT LogLuvHDR = one packed uint per sample.
            if (!this.sampleColor || this.sampleColor.size < sampleCount * 4) {
                this.sampleColor = new Buffer(this.device, {
                    size: sampleCount * 4,
                    structSize: 4,
                    bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                    memoryType: MemoryType.DeviceLocal,
                    name: "PathTracer::sampleColor",
                });
            }
            // Native uses R16Uint; WGSL storage requires r32uint (values < 2^16).
            // Offsets are implicit at fixed spp; the lookup table is variable-only.
            if (!this.fixedSampleCount && (!this.sampleOffset || this.sampleOffset.width !== frameDim[0] || this.sampleOffset.height !== frameDim[1])) {
                this.sampleOffset = new Texture(this.device, {
                    type: ResourceType.Texture2D,
                    width: frameDim[0],
                    height: frameDim[1],
                    format: ResourceFormat.R32Uint,
                    bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                    name: "PathTracer::sampleOffset",
                });
            }
        }

        // RTXDI frame setup (mirrors beginFrame/update around the path generator,
        // which writes the per-pixel surface data the resampling consumes).
        let mvec: import("@web-falcor/falcor").Texture | null = null;
        if (this.rtxdi) {
            this.dummyMvec ??= new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: 1,
                height: 1,
                format: ResourceFormat.RG32Float,
                bindFlags: ResourceBindFlags.ShaderResource,
                name: "PathTracer::dummyMvec",
            });
            mvec = renderData.getTexture("mvec") ?? this.dummyMvec;
            this.rtxdi.beginFrame(ctx, frameDim);
        }

        // Generate paths: primary hits from the V-buffer; misses write background.
        {
            const root = this.generatePass.getRootVar();
            this.scene.bindShaderData(root);
            if (this.rtxdi) this.rtxdi.setShaderData(root, mvec);
            this.bindStats(root, frameDim);
            this.bindPathTracerData(root["CB"]["gPathGenerator"] as ShaderVar, vbuffer, color, frameDim);
            this.bindNRD(root, (root["CB"]["gPathGenerator"] as ShaderVar), "outputNRD", frameDim);
            // One thread per pixel, padded to whole tiles (numthreads(256,1,1)).
            this.generatePass.execute(ctx, tiles[0]! * kScreenTileDim * kScreenTileDim, tiles[1]!);
        }

        // RTXDI resampling over the surface data written by the path generator.
        if (this.rtxdi) this.rtxdi.update(ctx, mvec!);

        // Trace paths (compute megakernel); with NRD additional data native launches two more
        // trace passes for the delta reflection / transmission guide buffers.
        const trace = (pass: ComputePass) => {
            const root = pass.getRootVar();
            this.scene!.bindShaderData(root);
            if (this.rtxdi) this.rtxdi.setShaderData(root, mvec);
            this.bindStats(root, frameDim);
            const block = root["gPathTracer"] as ShaderVar;
            this.bindPathTracerData(block, vbuffer, color, frameDim);
            this.bindNRD(root, block, "outputNRD", frameDim);
            this.pixelDebug.prepareProgram(root, pass);
            if (this.emissiveSampler === "Power" && this.scene!.useEmissiveLights) {
                if (!this.powerSampler) this.powerSampler = new EmissivePowerSampler(this.device, this.scene!.getEmissiveFluxes());
                this.powerSampler.bindShaderData(block["emissiveSampler"] as ShaderVar);
            }
            if (this.lightBVHSampler) this.lightBVHSampler.bindShaderData(block["emissiveSampler"] as ShaderVar);
            const envSampler = block["envMapSampler"] as ShaderVar;
            if (this.scene!.useEnvLight) {
                if (!this.envMapSampler) this.envMapSampler = new EnvMapSampler(this.device, ctx, this.scene!.getEnvMap()!);
                this.envMapSampler.bindShaderData(envSampler);
            } else {
                // Env map sampler members can survive DCE with env light off.
                envSampler["importanceSampler"] = this.dummySampler!;
                envSampler["importanceMap"] = this.dummyTexFloat!;
            }
            pass.execute(ctx, frameDim[0], frameDim[1]);
        };
        trace(this.tracePass!);
        if (this.traceDeltaReflectionPass) trace(this.traceDeltaReflectionPass);
        if (this.traceDeltaTransmissionPass) trace(this.traceDeltaTransmissionPass);

        // Resolve guide data / multi-sample color into the connected outputs
        // (mirrors resolvePass; with fixed spp == 1 the color loop is compiled out).
        if (this.resolvePass) {
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
            this.trySet(cb, "sampleGuideData", this.sampleGuideData ?? this.dummyBufferB!);
            this.trySet(cb, "sampleColor", this.sampleColor ?? this.dummyBufferA!);
            this.trySet(cb, "sampleCount", this.sampleCountInput ?? this.dummyTexUint!);
            this.trySet(cb, "sampleOffset", this.sampleOffset ?? this.dummyTexUint!);
            this.bindNRD(root, cb, null, frameDim);
            // Unconnected resolve outputs bind format-matched 1x1 dummies —
            // one PER OUTPUT: native binds null UAVs (writes dropped); WebGPU
            // storage bindings must match the kernel's static format and must
            // not alias, and out-of-bounds stores are dropped.
            const guideFormats: Record<string, ResourceFormat> = {
                albedo: ResourceFormat.RGBA8Unorm,
                specularAlbedo: ResourceFormat.RGBA8Unorm,
                indirectAlbedo: ResourceFormat.RGBA8Unorm,
                guideNormal: ResourceFormat.RGBA16Float,
                reflectionPosW: ResourceFormat.RGBA32Float,
                color: ResourceFormat.RGBA32Float,
            };
            const outputDummy = (name: string): Texture => {
                let t = this.outputDummies.get(name);
                if (!t) {
                    t = new Texture(this.device, {
                        type: ResourceType.Texture2D,
                        width: 1,
                        height: 1,
                        arraySize: 1,
                        mipLevels: 1,
                        format: guideFormats[name]!,
                        bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                        name: `PathTracer::dummyOut_${name}`,
                    });
                    this.outputDummies.set(name, t);
                }
                return t;
            };
            for (const name of ["albedo", "specularAlbedo", "indirectAlbedo", "guideNormal", "reflectionPosW"]) {
                const key = `output${name[0]!.toUpperCase()}${name.slice(1)}`;
                this.trySet(cb, key, renderData.getTexture(name) ?? outputDummy(name));
            }
            // The tracer writes color directly only at fixed spp == 1; otherwise
            // the resolve averages the per-sample colors.
            this.trySet(cb, "outputColor", this.fixedSampleCount && this.samplesPerPixel === 1 ? outputDummy("color") : color);
            this.resolvePass.execute(ctx, frameDim[0], frameDim[1]);
        }
        if (this.outputNRDData || this.outputNRDAdditionalData) this.copyNRDOutputs(ctx, renderData, frameDim);

        // Resolve pixel stats into the connected outputs (native copies its
        // stats textures in endFrame; the web resolve reads the packed
        // atomic buffer instead).
        if (this.statsEnabled) {
            const rayCount = renderData.getTexture("rayCount");
            const pathLength = renderData.getTexture("pathLength");
            if (!this.statsResolvePass) {
                this.statsResolvePass = ComputePass.create(this.device, {
                    path: "WebFalcor/PixelStatsResolve.cs.slang",
                    defines: { is_valid_gRayCount: rayCount ? 1 : 0, is_valid_gPathLength: pathLength ? 1 : 0 },
                });
            }
            const root = this.statsResolvePass.getRootVar();
            (root["CB"] as ShaderVar)["gFrameDim"] = frameDim;
            root["gStatsBuffer"] = this.statsBuffer!;
            if (rayCount) this.trySet(root, "gRayCount", rayCount);
            if (pathLength) this.trySet(root, "gPathLength", pathLength);
            this.statsResolvePass.execute(ctx, frameDim[0], frameDim[1]);
            // Aggregate totals for getPixelStats (mirrors PixelStats::endFrame).
            this.pixelStats ??= new PixelStats(this.device);
            this.pixelStats.resolve(ctx, this.statsBuffer!, frameDim);
        }

        if (this.rtxdi) this.rtxdi.endFrame(ctx);
        this.pixelDebug.endFrame();
        this.frameCount++;
    }
}

registerRenderPass("PathTracer", (device, props) => new PathTracer(device, props));
