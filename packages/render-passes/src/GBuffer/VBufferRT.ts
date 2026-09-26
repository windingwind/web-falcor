/**
 * Raytraced V-buffer pass mirroring Source/RenderPasses/GBuffer/VBuffer/VBufferRT
 * (compute variant). The upstream VBufferRT.cs.slang compiles unmodified over
 * the software ray-query override; VBufferRT.slang gets write-only textures.
 */

import {
    ComputePass,
    DxSamplePattern,
    HaltonSamplePattern,
    StratifiedSamplePattern,
    float2,
    type CPUSampleGenerator,
    FieldFlags,
    Properties,
    RenderData,
    kRenderPassPRNGDimension,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    SAMPLE_GENERATOR_DEFAULT,
    SampleGenerator,
    CullMode,
    IOSize,
    RayFlags,
    calculateIOSize,
    parseIOSize,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets, kRenderPassGBufferAdjustShadingNormals
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang";

/** Mirrors VBufferRT/GBufferRT::getShaderDefines' ray flags: a forced front/back cull mode culls those faces. */
export function cullRayFlags(forceCullMode: boolean, cullMode: CullMode): number {
    if (forceCullMode && cullMode === CullMode.Front) return RayFlags.CullFrontFacingTriangles;
    if (forceCullMode && cullMode === CullMode.Back) return RayFlags.CullBackFacingTriangles;
    return RayFlags.None;
}

export class VBufferRT extends RenderPass {
    private pass: ComputePass | null = null;
    private frameCount = 0;
    private useAlphaTest = true;
    private adjustShadingNormals = true;
    private forceCullMode = false;
    private cullMode = CullMode.Back;
    /** Stored as natively; inline queries are the only web path. */
    private useTraceRayInline = false;
    private outputSize = IOSize.Default;
    private fixedOutputSize: [number, number] = [512, 512];
    private useDOF = true;
    private computeDOF = false;
    private sampleGenerator: SampleGenerator;
    private cameraJitterGenerator: CPUSampleGenerator | null = null;
    private samplePattern = "Center";
    private sampleCount = 16;

    constructor(device: Device, props: Properties) {
        super(device);
        // GBufferBase::parseProperties.
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        const fixed = props.getOpt<number[] | { x: number; y: number }>("fixedOutputSize");
        if (fixed) this.fixedOutputSize = Array.isArray(fixed) ? [fixed[0]!, fixed[1]!] : [fixed.x, fixed.y];
        this.useAlphaTest = props.get("useAlphaTest", true);
        if (props.has("disableAlphaTest") && !props.has("useAlphaTest")) this.useAlphaTest = !props.get("disableAlphaTest", false);
        this.adjustShadingNormals = props.get("adjustShadingNormals", true);
        this.forceCullMode = props.get("forceCullMode", false);
        const cull = props.getOpt<string | number>("cull");
        if (cull !== undefined) this.cullMode = (typeof cull === "string" ? CullMode[cull as keyof typeof CullMode] : cull) ?? CullMode.Back;
        this.useTraceRayInline = props.get("useTraceRayInline", false);
        this.useDOF = props.get("useDOF", true);
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_DEFAULT);
        this.samplePattern = props.get<string>("samplePattern", "Center");
        this.sampleCount = props.get("sampleCount", 16);
        this.updateSamplePattern();
    }

    /** Mirrors GBufferBase::updateSamplePattern (Center -> no generator). */
    private updateSamplePattern(): void {
        const c = this.sampleCount;
        this.cameraJitterGenerator =
            this.samplePattern === "Stratified" ? new StratifiedSamplePattern(c)
            : this.samplePattern === "Halton" ? new HaltonSamplePattern(c)
            : this.samplePattern === "DirectX" ? new DxSamplePattern(c)
            : null;
    }

    override getProperties(): Properties {
        return new Properties({
            outputSize: IOSize[this.outputSize]!,
            ...(this.outputSize === IOSize.Fixed ? { fixedOutputSize: this.fixedOutputSize } : {}),
            samplePattern: this.samplePattern,
            sampleCount: this.sampleCount,
            useAlphaTest: this.useAlphaTest,
            adjustShadingNormals: this.adjustShadingNormals,
            forceCullMode: this.forceCullMode,
            cull: CullMode[this.cullMode]!,
            useTraceRayInline: this.useTraceRayInline,
            useDOF: this.useDOF,
        });
    }

    /** Mirrors GBufferBase::renderUI + VBufferRT::renderUI (define changes drop the kernel). */
    override renderUI(ui: UIWidgets): void {
        ui.dropdown("Sample pattern", ["Center", "DirectX", "Halton", "Stratified"], this.samplePattern, (v) => {
            this.samplePattern = v;
            this.updateSamplePattern();
        });
        ui.slider("Sample count", this.sampleCount, 1, 1024, 1, (v) => {
            this.sampleCount = Math.max(1, Math.round(v));
            this.updateSamplePattern();
        });
        ui.checkbox("Alpha Test", this.useAlphaTest, (v) => {
            this.useAlphaTest = v;
            this.pass = null;
        });
        ui.checkbox("Force cull mode", this.forceCullMode, (v) => {
            this.forceCullMode = v;
            this.pass = null;
        });
        if (this.forceCullMode) {
            ui.dropdown("Cull mode", ["None", "Front", "Back"], CullMode[this.cullMode]!, (v) => {
                this.cullMode = CullMode[v as keyof typeof CullMode];
                this.pass = null;
            });
        }
        ui.checkbox("Depth-of-field", this.useDOF, (v) => {
            this.useDOF = v;
            this.pass = null;
        });
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, this.fixedOutputSize, compileData.defaultTexDims);
        r.addOutput("vbuffer", "Packed hit information")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Uint)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        r.addOutput("viewW", "View direction (world)")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        r.addOutput("depth", "Depth buffer (NDC)")
            .texture2D(w, h)
            .format(ResourceFormat.R32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        r.addOutput("mvec", "Motion vector")
            .texture2D(w, h)
            .format(ResourceFormat.RG32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        r.addOutput("mask", "Mask")
            .texture2D(w, h)
            .format(ResourceFormat.R32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        // GBufferBase::setScene: a fresh jitter pattern (the stratified one carries RNG state).
        this.updateSamplePattern();
        this.pass = null;
        this.frameCount = 0;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        // GBufferBase::execute: consumers re-adjust shading normals the VBuffer can't carry.
        renderData.dictionary.set(kRenderPassGBufferAdjustShadingNormals, this.adjustShadingNormals);
        if (!this.scene) return;
        const vbuffer = renderData.getTexture("vbuffer")!;
        // Mirrors GBufferBase::updateFrameDim (first jitter sample lands next frame).
        this.scene.camera.setPatternGenerator(
            this.cameraJitterGenerator,
            new float2(Math.fround(1 / vbuffer.width), Math.fround(1 / vbuffer.height)),
        );
        // Mirrors VBufferRT::execute DoF config (two PRNG dims consumed; told downstream).
        const computeDOF = this.useDOF && this.scene.camera.getApertureRadius() > 0;
        if (computeDOF !== this.computeDOF) {
            this.computeDOF = computeDOF;
            this.pass = null;
        }
        if (this.useDOF) renderData.dictionary.set(kRenderPassPRNGDimension, computeDOF ? 2 : 0);
        if (!this.pass) {
            const defines = this.scene.getSceneDefines().addAll({
                USE_ALPHA_TEST: this.useAlphaTest ? 1 : 0,
                RAY_FLAGS: cullRayFlags(this.forceCullMode, this.cullMode),
                COMPUTE_DEPTH_OF_FIELD: this.computeDOF ? 1 : 0,
                is_valid_gDepth: renderData.getTexture("depth") ? 1 : 0,
                is_valid_gMotionVector: renderData.getTexture("mvec") ? 1 : 0,
                is_valid_gViewW: 1,
                is_valid_gTime: 0,
                is_valid_gMask: renderData.getTexture("mask") ? 1 : 0,
            });
            defines.addAll(this.sampleGenerator.getDefines());
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
        }
        const root = this.pass.getRootVar();
        this.scene.bindShaderData(root);
        root["gVBufferRT"]["frameDim"] = [vbuffer.width, vbuffer.height];
        root["gVBufferRT"]["frameCount"] = this.frameCount;
        root["gVBuffer"] = vbuffer;
        root["gViewW"] = renderData.getTexture("viewW")!;
        const depth = renderData.getTexture("depth");
        if (depth) root["gDepth"] = depth;
        const mvec = renderData.getTexture("mvec");
        if (mvec) root["gMotionVector"] = mvec;
        const mask = renderData.getTexture("mask");
        if (mask) root["gMask"] = mask;
        this.pass.execute(ctx, vbuffer.width, vbuffer.height);
        this.frameCount++;
    }
}

registerRenderPass("VBufferRT", (device, props) => new VBufferRT(device, props));
