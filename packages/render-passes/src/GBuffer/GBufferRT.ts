/**
 * Raytraced G-buffer pass mirroring Source/RenderPasses/GBuffer/GBuffer/GBufferRT
 * (compute variant over inline ray queries — the only path on web, so
 * useTraceRayInline is accepted and ignored). Uses the upstream
 * GBufferRT.cs.slang over the WebFalcor GBufferRT.slang override
 * (write-only outputs; normWRoughnessMaterialID is RGBA16Float on web:
 * WGSL has no rgb10a2unorm storage format).
 *
 * Web divergence: WebGPU caps storage textures per stage (8 on the dev GPU),
 * so the connected channels are partitioned into groups of <= 8 and the
 * primary rays are re-traced once per group (deterministic hits -> identical
 * values, ~Nx trace cost for N groups).
 */

import {
    ComputePass,
    DxSamplePattern,
    FieldFlags,
    HaltonSamplePattern,
    IOSize,
    Properties,
    RenderData,
    kRenderPassPRNGDimension,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    SAMPLE_GENERATOR_DEFAULT,
    SampleGenerator,
    StratifiedSamplePattern,
    calculateIOSize,
    float2,
    parseIOSize,
    registerRenderPass,
    type CompileData,
    type CPUSampleGenerator,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/GBuffer/GBuffer/GBufferRT.cs.slang";

/** [output name, shader texture name, format] — kGBufferChannels + kGBufferExtraChannels. */
const kChannels: [string, string, ResourceFormat][] = [
    ["posW", "gPosW", ResourceFormat.RGBA32Float],
    ["normW", "gNormW", ResourceFormat.RGBA32Float],
    ["tangentW", "gTangentW", ResourceFormat.RGBA32Float],
    ["faceNormalW", "gFaceNormalW", ResourceFormat.RGBA32Float],
    ["texC", "gTexC", ResourceFormat.RG32Float],
    ["texGrads", "gTexGrads", ResourceFormat.RGBA16Float],
    ["mvec", "gMotionVector", ResourceFormat.RG32Float],
    ["mtlData", "gMaterialData", ResourceFormat.RGBA32Uint],
    ["vbuffer", "gVBuffer", ResourceFormat.RGBA32Uint],
    ["depth", "gDepth", ResourceFormat.R32Float],
    ["linearZ", "gLinearZ", ResourceFormat.RG32Float],
    ["mvecW", "gMotionVectorW", ResourceFormat.RGBA16Float],
    // Native: RGB10A2Unorm; WGSL has no rgb10a2unorm storage format.
    ["normWRoughnessMaterialID", "gNormalWRoughnessMaterialID", ResourceFormat.RGBA16Float],
    ["guideNormalW", "gGuideNormalW", ResourceFormat.RGBA32Float],
    ["diffuseOpacity", "gDiffOpacity", ResourceFormat.RGBA32Float],
    ["specRough", "gSpecRough", ResourceFormat.RGBA32Float],
    ["emissive", "gEmissive", ResourceFormat.RGBA32Float],
    ["viewW", "gViewW", ResourceFormat.RGBA32Float],
    ["time", "gTime", ResourceFormat.R32Uint],
    ["disocclusion", "gDisocclusion", ResourceFormat.R32Float],
    ["mask", "gMask", ResourceFormat.R32Float],
];

/** Mirrors TexLODMode (TexLODTypes.slang). */
const kLODModes: Record<string, number> = { Mip0: 0, RayCones: 1, RayDiffs: 2 };

/** GBufferBase::SamplePattern spellings. */
const kSamplePatterns = ["Center", "DirectX", "Halton", "Stratified"];

export class GBufferRT extends RenderPass {
    private passes = new Map<string, ComputePass>();
    private frameCount = 0;
    private useAlphaTest = true;
    private adjustShadingNormals = true;
    private useDOF = true;
    private computeDOF = false;
    private lodMode = 0;
    private outputSize = IOSize.Default;
    /** Native kFixedOutputSize default (used when outputSize == Fixed). */
    private fixedOutputSize: [number, number] = [512, 512];
    private sampleGenerator: SampleGenerator;
    private cameraJitterGenerator: CPUSampleGenerator | null = null;
    private samplePattern = "Center";
    private sampleCount = 16;

    constructor(device: Device, props: Properties) {
        super(device);
        this.useAlphaTest = props.get("useAlphaTest", true);
        this.adjustShadingNormals = props.get("adjustShadingNormals", true);
        const lod = props.getOpt<string | number>("texLOD");
        if (lod !== undefined) this.lodMode = (typeof lod === "string" ? kLODModes[lod] : lod) ?? 0;
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        const fixed = props.getOpt<number[] | { x: number; y: number }>("fixedOutputSize");
        if (fixed) this.fixedOutputSize = Array.isArray(fixed) ? [fixed[0]!, fixed[1]!] : [fixed.x, fixed.y];
        // 'useTraceRayInline' accepted: inline queries are the only web path.
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
            fixedOutputSize: this.fixedOutputSize,
            samplePattern: this.samplePattern,
            sampleCount: this.sampleCount,
            useAlphaTest: this.useAlphaTest,
            adjustShadingNormals: this.adjustShadingNormals,
            texLOD: Object.keys(kLODModes).find((k) => kLODModes[k] === this.lodMode) ?? this.lodMode,
            useDOF: this.useDOF,
        });
    }

    /** Mirrors GBufferBase::renderUI + GBufferRT::renderUI (define changes drop the kernels). */
    override renderUI(ui: UIWidgets): void {
        const rebuild = <T>(set: (v: T) => void) => (v: T) => {
            set(v);
            this.passes.clear();
        };
        // Native GBufferBase/ImageLoader/ToneMapper/... "Output size" controls: I/O size changes recompile the graph.
        ui.dropdown("Output size", ["Default", "Fixed", "Full", "Half", "Quarter", "Double"], IOSize[this.outputSize]!, (v) => {
            this.outputSize = IOSize[v as keyof typeof IOSize];
            this.requestRecompile();
        });
        ui.slider("Size in pixels (width)", this.fixedOutputSize[0], 32, 4096, 1, (v) => {
            this.fixedOutputSize = [Math.round(v), this.fixedOutputSize[1]];
            this.requestRecompile();
        });
        ui.slider("Size in pixels (height)", this.fixedOutputSize[1], 32, 4096, 1, (v) => {
            this.fixedOutputSize = [this.fixedOutputSize[0], Math.round(v)];
            this.requestRecompile();
        });
        ui.dropdown("Sample pattern", kSamplePatterns, this.samplePattern, (v) => {
            this.samplePattern = v;
            this.updateSamplePattern();
        });
        ui.slider("Sample count", this.sampleCount, 1, 1024, 1, (v) => {
            this.sampleCount = Math.max(1, Math.round(v));
            this.updateSamplePattern();
        });
        ui.checkbox("Alpha Test", this.useAlphaTest, rebuild((v) => (this.useAlphaTest = v)));
        ui.checkbox("Adjust shading normals", this.adjustShadingNormals, rebuild((v) => (this.adjustShadingNormals = v)));
        ui.dropdown("Texture LOD mode", Object.keys(kLODModes), Object.keys(kLODModes).find((k) => kLODModes[k] === this.lodMode) ?? "Mip0", rebuild((v: string) => (this.lodMode = kLODModes[v]!)));
        ui.checkbox("Depth-of-field", this.useDOF, rebuild((v) => (this.useDOF = v)));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, this.fixedOutputSize, compileData.defaultTexDims);
        for (const [name, , format] of kChannels) {
            r.addOutput(name, `G-buffer ${name}`)
                .texture2D(w, h)
                .format(format)
                .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
                .flags(FieldFlags.Optional);
        }
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.passes.clear();
        this.frameCount = 0;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const anyOutput = kChannels.map(([name]) => renderData.getTexture(name)).find((t) => t !== undefined);
        if (!anyOutput) return;
        const [w, h] = [anyOutput.width, anyOutput.height];

        // Mirrors GBufferBase::updateFrameDim (first jitter sample lands next frame).
        this.scene.camera.setPatternGenerator(this.cameraJitterGenerator, new float2(Math.fround(1 / w), Math.fround(1 / h)));

        // Mirrors GBufferRT::execute DoF config (two PRNG dims consumed; told downstream).
        const computeDOF = this.useDOF && this.scene.camera.getApertureRadius() > 0;
        if (computeDOF !== this.computeDOF) {
            this.computeDOF = computeDOF;
            this.passes.clear();
        }
        if (this.useDOF) renderData.dictionary.set(kRenderPassPRNGDimension, computeDOF ? 2 : 0);

        // Partition connected channels into groups within the storage-texture
        // limit; each group is one kernel variant tracing the same primary rays.
        const present = kChannels.filter(([name]) => renderData.getTexture(name) !== undefined);
        const kGroupSize = 8;
        for (let g = 0; g < present.length; g += kGroupSize) {
            const group = present.slice(g, g + kGroupSize);
            const valid: Record<string, number> = {};
            for (const [, texname] of kChannels) valid[`is_valid_${texname}`] = 0;
            for (const [, texname] of group) valid[`is_valid_${texname}`] = 1;
            const key = JSON.stringify(valid);
            let pass = this.passes.get(key);
            if (!pass) {
                const defines = this.scene.getSceneDefines().addAll({
                    USE_ALPHA_TEST: this.useAlphaTest ? 1 : 0,
                    ADJUST_SHADING_NORMALS: this.adjustShadingNormals ? 1 : 0,
                    LOD_MODE: this.lodMode,
                    RAY_FLAGS: 0,
                    COMPUTE_DEPTH_OF_FIELD: this.computeDOF ? 1 : 0,
                    ...valid,
                });
                defines.addAll(this.sampleGenerator.getDefines());
                pass = ComputePass.create(this.device, { path: kShaderFile, defines });
                this.passes.set(key, pass);
            }

            const root = pass.getRootVar();
            this.scene.bindShaderData(root);
            const cb = root["gGBufferRT"]!;
            cb["frameDim"] = [w, h];
            cb["invFrameDim"] = [Math.fround(1 / w), Math.fround(1 / h)];
            cb["frameCount"] = this.frameCount;
            // Mirrors Camera::computeScreenSpacePixelSpreadAngle (kDefaultFrameHeight = 24).
            const fovY = 2 * Math.atan(0.5 * 24 / this.scene.camera.getFocalLength());
            cb["screenSpacePixelSpreadAngle"] = Math.fround(Math.atan((2 * Math.tan(fovY * 0.5)) / h));
            for (const [name, texname] of group) {
                root[texname] = renderData.getTexture(name)!;
            }
            pass.execute(ctx, w, h);
        }
        this.frameCount++;
    }
}

registerRenderPass("GBufferRT", (device, props) => new GBufferRT(device, props));
