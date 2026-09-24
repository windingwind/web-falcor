/**
 * Mirrors Source/RenderPasses/WARDiffPathTracer: a differentiable path tracer with warped-area
 * reparameterization (Slang autodiff through the path tracer). The raygen shader runs as a
 * compute kernel (WebFalcor override).
 *
 * §9: Primal and ForwardDiffDebug are ported. BackwardDiff and BackwardDiffDebug crash
 * Slang 2026.18 while it transposes the nested fwd_diff of the reparameterization, so those
 * modes throw with the repro location.
 */

import {
    Buffer,
    ComputePass,
    Logger,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    SAMPLE_GENERATOR_UNIFORM,
    SampleGenerator,
    SceneGradients,
    GeometryType,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type ShaderVar,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/WARDiffPathTracer/WARDiffPathTracer.rt.slang";
/**
 * Slang 2026.7's autodiff refactor (#9808) crashes transposing this kernel's nested fwd_diff;
 * the backward modes compile with the last release before it (fetched by scripts/setup-web.mjs).
 */
const kAutodiffSlang = "/tools/slang-wasm-2026.5.2/slang-wasm.js";

/** Mirrors DiffMode (DiffRendering/SharedTypes.slang). */
export enum DiffMode {
    Primal = 0,
    BackwardDiff = 1,
    ForwardDiffDebug = 2,
    BackwardDiffDebug = 3,
}

/** Mirrors DiffVariableType. */
enum DiffVariableType {
    None = 0,
    Material = 1,
    GeometryTranslation = 2,
}

/** Mirrors GradientType::Count. */
const kGradientTypeCount = 4;

/** std::to_string(float): six fractional digits. */
const floatDefine = (v: number) => v.toFixed(6);

export class WARDiffPathTracer extends RenderPass {
    private pass: ComputePass | null = null;
    private sampleGenerator: SampleGenerator;
    private frameCount = 0;
    private useFixedSeed = false;
    private fixedSeed = 1;
    private dummyBuffer: Buffer | null = null;
    private gradDummies: Buffer[] = [];

    // StaticParams (native defaults).
    private samplesPerPixel = 1;
    private maxBounces = 0;
    private diffMode = DiffMode.ForwardDiffDebug;
    private diffVarName = "";
    private useBSDFSampling = true;
    private useNEE = true;
    private useMIS = true;
    private useWAR = true;
    private auxSampleCount = 16;
    private log10vMFConcentration = 5;
    private log10vMFConcentrationScreen = 5;
    private boundaryTermBeta = 0.01;
    private useAntitheticSampling = true;
    private harmonicGamma = 2;

    // DiffDebugParams.
    private debugVarType = DiffVariableType.None;
    private debugId: [number, number] = [0, 0];
    private debugOffset = 0;
    private debugGrad: [number, number, number, number] = [0, 0, 0, 0];

    constructor(device: Device, props: Properties) {
        super(device);
        this.parseProperties(props);
        // Native creates a uniform generator regardless of the sampleGenerator property's default.
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_UNIFORM);
        if (this.diffVarName === "CBOX_BUNNY_MATERIAL") {
            // Albedo value with materialID = 0.
            this.setDiffDebugParams(DiffVariableType.Material, [0, 0], 0, [1, 1, 1, 0]);
        } else if (this.diffVarName === "CBOX_BUNNY_TRANSLATION") {
            // Vertical translation with meshID = 0.
            this.setDiffDebugParams(DiffVariableType.GeometryTranslation, [0, 0], 0, [0, 1, 0, 0]);
        }
    }

    private parseProperties(props: Properties): void {
        for (const [key, value] of props.entries()) {
            switch (key) {
                case "samplesPerPixel": this.samplesPerPixel = Number(value); break;
                case "maxBounces": this.maxBounces = Number(value); break;
                case "diffMode": this.diffMode = typeof value === "string" ? DiffMode[value as keyof typeof DiffMode] : Number(value); break;
                case "diffVarName": this.diffVarName = String(value); break;
                case "sampleGenerator": break;
                case "fixedSeed": this.fixedSeed = Number(value); this.useFixedSeed = true; break;
                case "useBSDFSampling": this.useBSDFSampling = Boolean(value); break;
                case "useNEE": this.useNEE = Boolean(value); break;
                case "useMIS": this.useMIS = Boolean(value); break;
                case "useWAR": this.useWAR = Boolean(value); break;
                case "auxSampleCount": this.auxSampleCount = Number(value); break;
                case "Log10vMFConcentration": this.log10vMFConcentration = Number(value); break;
                case "Log10vMFConcentrationScreen": this.log10vMFConcentrationScreen = Number(value); break;
                case "boundaryTermBeta": this.boundaryTermBeta = Number(value); break;
                case "useAntitheticSampling": this.useAntitheticSampling = Boolean(value); break;
                default: Logger.warning(`Unknown property '${key}' in WARDiffPathTracer properties.`);
            }
        }
        if (this.diffMode === undefined || Number.isNaN(this.diffMode)) throw new Error("WARDiffPathTracer: unknown diffMode");
    }

    override getProperties(): Properties {
        const p: Record<string, unknown> = {
            samplesPerPixel: this.samplesPerPixel,
            maxBounces: this.maxBounces,
            diffMode: DiffMode[this.diffMode],
            diffVarName: this.diffVarName,
            sampleGenerator: SAMPLE_GENERATOR_UNIFORM,
        };
        if (this.useFixedSeed) p.fixedSeed = this.fixedSeed;
        Object.assign(p, {
            useBSDFSampling: this.useBSDFSampling,
            useNEE: this.useNEE,
            useMIS: this.useMIS,
            useWAR: this.useWAR,
            auxSampleCount: this.auxSampleCount,
            Log10vMFConcentration: this.log10vMFConcentration,
            Log10vMFConcentrationScreen: this.log10vMFConcentrationScreen,
            boundaryTermBeta: this.boundaryTermBeta,
            useAntitheticSampling: this.useAntitheticSampling,
        });
        return new Properties(p as Record<string, never>);
    }

    /** Mirrors WARDiffPathTracer::setDiffDebugParams. */
    setDiffDebugParams(varType: number, id: [number, number], offset: number, grad: [number, number, number, number]): void {
        this.debugVarType = varType;
        this.debugId = id;
        this.debugOffset = offset;
        this.debugGrad = grad;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        // RenderTarget: WebGPU clears through a render pass (native clearUAV).
        const storage = ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget;
        r.addOutput("color", "Output color (sum of direct and indirect)").texture2D(w, h).format(ResourceFormat.RGBA32Float).bindFlags(storage);
        r.addOutput("dColor", "Output derivatives computed via auto-diff").texture2D(w, h).format(ResourceFormat.RGBA32Float).bindFlags(storage);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.frameCount = 0;
        this.pass = null;
        if (scene?.hasGeometryType(GeometryType.Custom)) Logger.error("WARDiffPathTracer: This render pass does not support custom primitives.");
        if (scene?.useEnvLight) Logger.error("WARDiffPathTracer: This render pass does not support environment lights.");
    }

    /** Mirrors WARDiffPathTracer::renderRenderingUI (every option is a shader define). */
    override renderUI(ui: UIWidgets): void {
        const rebuild = <T>(set: (v: T) => void) => (v: T) => {
            set(v);
            this.pass = null;
        };
        ui.slider("Samples/pixel", this.samplesPerPixel, 1, 16, 1, rebuild((v) => (this.samplesPerPixel = Math.round(v))));
        ui.slider("Max bounces", this.maxBounces, 0, 254, 1, rebuild((v) => (this.maxBounces = Math.round(v))));
        ui.dropdown("Diff mode", ["Primal", "BackwardDiff", "ForwardDiffDebug", "BackwardDiffDebug"], DiffMode[this.diffMode], rebuild((v: string) => (this.diffMode = DiffMode[v as keyof typeof DiffMode])));
        ui.text(`Diff variable name: ${this.diffVarName}`);
        ui.checkbox("Antithetic sampling", this.useAntitheticSampling, rebuild((v) => (this.useAntitheticSampling = v)));
        ui.checkbox("BSDF importance sampling", this.useBSDFSampling, rebuild((v) => (this.useBSDFSampling = v)));
        ui.checkbox("Next-event estimation (NEE)", this.useNEE, rebuild((v) => (this.useNEE = v)));
        if (this.useNEE) ui.checkbox("Multiple importance sampling (MIS)", this.useMIS, rebuild((v) => (this.useMIS = v)));
        ui.checkbox("Use fixed seed", this.useFixedSeed, (v) => (this.useFixedSeed = v));
    }

    /** Mirrors StaticParams::getDefines. */
    private getDefines() {
        const scene = this.scene!;
        const defines = scene.getSceneDefines().addAll({
            SAMPLES_PER_PIXEL: this.samplesPerPixel,
            MAX_BOUNCES: this.maxBounces,
            DIFF_MODE: this.diffMode,
            USE_BSDF_SAMPLING: this.useBSDFSampling ? 1 : 0,
            USE_NEE: this.useNEE ? 1 : 0,
            USE_MIS: this.useMIS ? 1 : 0,
            USE_WAR: this.useWAR ? 1 : 0,
            AUX_SAMPLE_COUNT: this.auxSampleCount,
            LOG10_VMF_CONCENTRATION: floatDefine(this.log10vMFConcentration),
            LOG10_VMF_CONCENTRATION_SCREEN: floatDefine(this.log10vMFConcentrationScreen),
            BOUNDARY_TERM_BETA: floatDefine(this.boundaryTermBeta),
            USE_ANTITHETIC_SAMPLING: this.useAntitheticSampling ? 1 : 0,
            HARMONIC_GAMMA: floatDefine(this.harmonicGamma),
            // The uniform emissive sampler (native: LightBVH "seems buggy" here).
            ...(scene.useEmissiveLights ? { _EMISSIVE_LIGHT_SAMPLER_TYPE: 0 } : {}),
            USE_ENV_LIGHT: scene.useEnvLight ? 1 : 0,
            USE_ANALYTIC_LIGHTS: scene.useAnalyticLights ? 1 : 0,
            USE_EMISSIVE_LIGHTS: scene.useEmissiveLights ? 1 : 0,
            // gDiffDebug/gInvOpt as ConstantBuffers (WebGPU's 4 bind groups).
            WEBFALCOR_DIFF_PARAMS_AS_CB: 1,
        });
        if (this.diffVarName) defines.add(this.diffVarName, "");
        return defines.addAll(this.sampleGenerator.getDefines());
    }

    private trySet(var_: ShaderVar, name: string, value: unknown): void {
        try {
            (var_ as Record<string, unknown>)[name] = value;
        } catch {
            /* binding absent in this variant */
        }
    }

    // --- Python surface (native registerBindings) ---
    /** Mirrors scene_gradients: where BackwardDiff accumulates (no gradients without it). */
    scene_gradients: SceneGradients | null = null;
    /** Mirrors dL_dI: per-pixel float3 loss gradient BackwardDiff starts from. */
    dL_dI: Buffer | null = null;
    /** Mirrors run_backward: 0 renders the primal in BackwardDiff mode. */
    run_backward = 1;

    override async initAsync(): Promise<void> {
        await this.device.programManager.loadSlangRuntime(kAutodiffSlang);
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const color = renderData.getTexture("color")!;
        const dColor = renderData.getTexture("dColor")!;
        // Native beginFrame keeps the outputs while BackwardDiff runs backward.
        if (!(this.diffMode === DiffMode.BackwardDiff && this.run_backward === 1)) {
            ctx.clearTexture(color);
            ctx.clearTexture(dColor);
        }
        if (!this.scene) return;
        // The backward modes compile with the pinned pre-refactor Slang (see kAutodiffSlang).
        const backward = this.diffMode === DiffMode.BackwardDiff || this.diffMode === DiffMode.BackwardDiffDebug;
        if (!this.pass) this.pass = ComputePass.create(this.device, { path: kShaderFile, defines: this.getDefines(), slangRuntime: backward ? kAutodiffSlang : undefined });

        const root = this.pass.getRootVar();
        this.scene.bindShaderData(root);
        const [w, h] = [color.width, color.height];
        const params = (root["gDiffPTData"] as ShaderVar)["params"] as ShaderVar;
        params["useFixedSeed"] = this.useFixedSeed ? 1 : 0;
        params["fixedSeed"] = this.fixedSeed;
        params["assertThreshold"] = 1e9;
        params["runBackward"] = this.run_backward;
        params["frameDim"] = [w, h];
        params["screenTiles"] = [0, 0];
        params["frameCount"] = this.frameCount;
        params["seed"] = this.useFixedSeed ? this.fixedSeed : this.frameCount;

        const debug = root["gDiffDebug"] as ShaderVar;
        debug["varType"] = this.debugVarType;
        debug["id"] = this.debugId;
        debug["offset"] = this.debugOffset;
        debug["grad"] = this.debugGrad;
        this.trySet(root["gInvOpt"] as ShaderVar, "meshID", 0);

        // No SceneGradients object in the debug modes (native leaves the block unbound).
        if (!this.dummyBuffer) {
            this.dummyBuffer = new Buffer(this.device, { size: 16, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: "WARDiffPathTracer::dummy" });
        }
        // WebGPU forbids aliasing writable bindings: one placeholder per gradient buffer.
        if (this.gradDummies.length === 0) {
            for (let i = 0; i < kGradientTypeCount; i++) {
                this.gradDummies.push(new Buffer(this.device, { size: 16, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: `WARDiffPathTracer::tmpGrads${i}` }));
            }
        }
        let grads: ShaderVar | null = null;
        try {
            grads = root["gSceneGradients"] as ShaderVar;
        } catch {
            /* gradients unused in this variant */
        }
        if (grads && this.scene_gradients) this.scene_gradients.bindShaderData(grads);
        else if (grads) {
            try {
                grads["gradDim"] = [0, 0, 0, 0];
                grads["hashSize"] = [1, 1, 1, 1];
            } catch {
                /* dims stripped when unused */
            }
            for (let i = 0; i < kGradientTypeCount; i++) this.trySet(grads, `tmpGrads${i}`, this.gradDummies[i]!);
        }
        this.trySet(root, "dLdI", this.dL_dI ?? this.dummyBuffer);
        root["gOutputColor"] = color;
        this.trySet(root, "gOutputDColor", dColor);
        this.pass.execute(ctx, w, h);
        this.frameCount++;
    }
}

registerRenderPass("WARDiffPathTracer", (device, props) => new WARDiffPathTracer(device, props));
