/**
 * Reference path tracer mirroring Source/RenderPasses/PathTracer (host side of
 * GeneratePaths + TracePass; the resolve pass is skipped for fixed spp == 1,
 * as native does). TracePass runs as the WebFalcor compute megakernel over
 * software ray queries (see the TracePass.rt.slang override).
 *
 * v1 limits (documented in DESIGN.md parity matrix): fixed sample count only,
 * no guide/NRD outputs, EmissiveUniformSampler only (LightBVH/Power pending),
 * no RTXDI/SER, spp == 1 (per-sample color buffers pending).
 */

import {
    Buffer,
    ComputePass,
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

/** Mirrors EmissiveLightSamplerType (EmissiveLightSamplerType.slangh). */
const kEmissiveSamplerUniform = 0;

const kScreenTileDim = 16;

export class PathTracer extends RenderPass {
    private generatePass: ComputePass | null = null;
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
            _EMISSIVE_LIGHT_SAMPLER_TYPE: kEmissiveSamplerUniform,
            INTERIOR_LIST_SLOT_COUNT: 2,
            GBUFFER_ADJUST_SHADING_NORMALS: 0,
            USE_ENV_LIGHT: scene.useEnvLight ? 1 : 0,
            USE_ANALYTIC_LIGHTS: scene.useAnalyticLights ? 1 : 0,
            USE_EMISSIVE_LIGHTS: scene.useEmissiveLights ? 1 : 0,
            USE_CURVES: 0,
            USE_SDF_GRIDS: 0,
            USE_HAIR_MATERIAL: 0,
            USE_VIEW_DIR: 0,
            OUTPUT_GUIDE_DATA: 0,
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
        var_["sampleGuideData"] = this.dummyBufferB;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const color = renderData.getTexture("color")!;
        const vbuffer = renderData.getTexture("vbuffer")!;
        const frameDim: [number, number] = [color.width, color.height];
        const tiles = [Math.ceil(frameDim[0] / kScreenTileDim), Math.ceil(frameDim[1] / kScreenTileDim)];

        if (!this.generatePass) {
            const defines = this.getStaticDefines();
            this.generatePass = ComputePass.create(this.device, { path: kGeneratePathsFile, defines });
            this.tracePass = ComputePass.create(this.device, { path: kTracePassFile, defines });
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
            // Env map sampler members can survive DCE with env light off.
            const envSampler = block["envMapSampler"] as ShaderVar;
            envSampler["importanceSampler"] = this.dummySampler!;
            envSampler["importanceMap"] = this.dummyTexFloat!;
            this.tracePass!.execute(ctx, frameDim[0], frameDim[1]);
        }

        this.frameCount++;
    }
}

registerRenderPass("PathTracer", (device, props) => new PathTracer(device, props));
