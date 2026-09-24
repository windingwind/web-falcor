/**
 * Mirrors RenderPasses/NRDPass ("NRD"): NVIDIA's Real-time Denoisers 3.1.0 driven like native.
 * The library (wasm/nrd.wasm, the unmodified NRD 3.1.0 C++ built by scripts/build-nrd-wasm.mjs)
 * plans the dispatches; this pass compiles NRD's HLSL through Slang, binds the resources the
 * library names and runs them. Methods: RelaxDiffuseSpecular, RelaxDiffuse,
 * ReblurDiffuseSpecular, SpecularReflectionMv, SpecularDeltaMv.
 *
 * §9 (WebGPU): NRD's registers are dropped at setup (WGSL numbers bindings per group); a
 * pipeline's textures, storage textures and samplers bind in declaration order, which is NRD's
 * register order. Pool formats WebGPU can't store to are widened (R8 unorm -> R32F since some are
 * read-write, RG8 unorm -> RGBA8Unorm, R16F -> R32F), and so are the RG16Float motion-vector outputs (-> RG32Float). Radiance is
 * packed into pass-owned textures rather than in place in the inputs.
 */

import {
    Buffer,
    ComputePass,
    Logger,
    MemoryType,
    NRDDescriptorType,
    NRDLibrary,
    NRDMethod,
    NRDResourceType,
    NRDSampler,
    Properties,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Sampler,
    Texture,
    TextureAddressingMode,
    TextureFilteringMode,
    calculateIOSize,
    parseIOSize,
    perspective,
    registerRenderPass,
    IOSize,
    type CompileData,
    type Device,
    type NRDDenoiserDesc,
    type NRDDispatch,
    type NRDSettings,
    type RenderContext,
    type RenderData,
    type Scene,
    type ShaderVar,
    type UIWidgets,
} from "@web-falcor/falcor";

/** Mirrors NRDPass::DenoisingMethod. */
export enum DenoisingMethod {
    RelaxDiffuseSpecular,
    RelaxDiffuse,
    ReblurDiffuseSpecular,
    SpecularReflectionMv,
    SpecularDeltaMv,
}

const kShaderPackRadiance = "RenderPasses/NRDPass/PackRadiance.cs.slang";
const kShaderFileList = "/tools/nrd-3.1.0/shader-files.json";
/** NRDConstants.slang kNRDDepthRange. */
const kNRDDepthRange = 10000;
/** nrd::RELAX_MAX_HISTORY_FRAME_NUM / REBLUR_MAX_HISTORY_FRAME_NUM. */
const kRelaxMaxHistoryFrameNum = 255;
const kReblurMaxHistoryFrameNum = 63;

const kInput = {
    diffuseRadianceHitDist: "diffuseRadianceHitDist",
    specularRadianceHitDist: "specularRadianceHitDist",
    specularHitDist: "specularHitDist",
    mvec: "mvec",
    normWRoughnessMaterialID: "normWRoughnessMaterialID",
    viewZ: "viewZ",
    deltaPrimaryPosW: "deltaPrimaryPosW",
    deltaSecondaryPosW: "deltaSecondaryPosW",
};
const kOutput = {
    filteredDiffuseRadianceHitDist: "filteredDiffuseRadianceHitDist",
    filteredSpecularRadianceHitDist: "filteredSpecularRadianceHitDist",
    reflectionMvec: "reflectionMvec",
    deltaMvec: "deltaMvec",
};

/** nrd::Format -> ResourceFormat (native getFalcorFormat), widened where WebGPU can't store. */
function poolFormat(format: number): ResourceFormat {
    const table: Record<number, ResourceFormat> = {
        // R8_UNORM (widened): NRD reads-modifies-writes some (history length), and WGSL
        // read_write storage takes only r32 formats.
        0: ResourceFormat.R32Float,
        4: ResourceFormat.RGBA8Unorm, // RG8_UNORM (widened)
        8: ResourceFormat.RGBA8Unorm,
        15: ResourceFormat.R32Uint, // R16_UINT (widened)
        17: ResourceFormat.R32Float, // R16_SFLOAT (widened)
        22: ResourceFormat.RG32Float, // RG16_SFLOAT (widened)
        25: ResourceFormat.RGBA16Uint,
        27: ResourceFormat.RGBA16Float,
        28: ResourceFormat.R32Uint,
        30: ResourceFormat.R32Float,
        31: ResourceFormat.RG32Uint,
        33: ResourceFormat.RG32Float,
        37: ResourceFormat.RGBA32Uint,
        39: ResourceFormat.RGBA32Float,
    };
    const f = table[format];
    if (f === undefined) throw new Error(`NRDPass: unsupported NRD pool format ${format}`);
    return f;
}

/** Native's overrides of the NRD defaults (NRDPass constructor). */
const kRelaxDiffuseSpecularDefaults: Record<string, number | boolean> = {
    diffusePrepassBlurRadius: 16,
    specularPrepassBlurRadius: 16,
    diffuseMaxFastAccumulatedFrameNum: 2,
    specularMaxFastAccumulatedFrameNum: 2,
    diffuseLobeAngleFraction: 0.8,
    disocclusionFixMaxRadius: 32,
    enableSpecularVirtualHistoryClamping: false,
    disocclusionFixNumFramesToFix: 4,
    spatialVarianceEstimationHistoryThreshold: 4,
    atrousIterationNum: 6,
    depthThreshold: 0.02,
    roughnessFraction: 0.5,
    specularLobeAngleFraction: 0.9,
    specularLobeAngleSlack: 10,
};
const kRelaxDiffuseDefaults: Record<string, number | boolean> = {
    prepassBlurRadius: 16,
    diffuseMaxFastAccumulatedFrameNum: 2,
    diffuseLobeAngleFraction: 0.8,
    disocclusionFixMaxRadius: 32,
    disocclusionFixNumFramesToFix: 4,
    spatialVarianceEstimationHistoryThreshold: 4,
    atrousIterationNum: 6,
    depthThreshold: 0.02,
};

/** Serialized ReLAX diffuse/specular keys (native kDiffusePrepassBlurRadius...), in native order. */
const kRelaxDiffuseSpecularKeys = [
    "diffusePrepassBlurRadius", "specularPrepassBlurRadius", "diffuseMaxAccumulatedFrameNum", "specularMaxAccumulatedFrameNum",
    "diffuseMaxFastAccumulatedFrameNum", "specularMaxFastAccumulatedFrameNum", "diffusePhiLuminance", "specularPhiLuminance",
    "diffuseLobeAngleFraction", "specularLobeAngleFraction", "roughnessFraction", "diffuseHistoryRejectionNormalThreshold",
    "specularVarianceBoost", "specularLobeAngleSlack", "disocclusionFixEdgeStoppingNormalPower", "disocclusionFixMaxRadius",
    "disocclusionFixNumFramesToFix", "historyClampingColorBoxSigmaScale", "spatialVarianceEstimationHistoryThreshold",
    "atrousIterationNum", "minLuminanceWeight", "depthThreshold", "roughnessEdgeStoppingRelaxation", "normalEdgeStoppingRelaxation",
    "luminanceEdgeStoppingRelaxation", "enableAntiFirefly", "enableReprojectionTestSkippingWithoutMotion",
    "enableSpecularVirtualHistoryClamping", "enableRoughnessEdgeStopping", "enableMaterialTestForDiffuse", "enableMaterialTestForSpecular",
];
/** Serialized ReLAX diffuse keys -> RelaxDiffuseSettings fields. */
const kRelaxDiffuseKeys: Record<string, string> = {
    diffusePrepassBlurRadius: "prepassBlurRadius",
    diffuseMaxAccumulatedFrameNum: "diffuseMaxAccumulatedFrameNum",
    diffuseMaxFastAccumulatedFrameNum: "diffuseMaxFastAccumulatedFrameNum",
    diffusePhiLuminance: "diffusePhiLuminance",
    diffuseLobeAngleFraction: "diffuseLobeAngleFraction",
    diffuseHistoryRejectionNormalThreshold: "diffuseHistoryRejectionNormalThreshold",
    disocclusionFixEdgeStoppingNormalPower: "disocclusionFixEdgeStoppingNormalPower",
    disocclusionFixMaxRadius: "disocclusionFixMaxRadius",
    disocclusionFixNumFramesToFix: "disocclusionFixNumFramesToFix",
    historyClampingColorBoxSigmaScale: "historyClampingColorBoxSigmaScale",
    spatialVarianceEstimationHistoryThreshold: "spatialVarianceEstimationHistoryThreshold",
    atrousIterationNum: "atrousIterationNum",
    minLuminanceWeight: "minLuminanceWeight",
    depthThreshold: "depthThreshold",
    enableAntiFirefly: "enableAntiFirefly",
    enableReprojectionTestSkippingWithoutMotion: "enableReprojectionTestSkippingWithoutMotion",
    enableMaterialTestForDiffuse: "enableMaterialTest",
};

/** Copies a row-major float4x4 into NRD's column-major float[16] (native copyMatrix). */
function columnMajor(m: { get(r: number, c: number): number }): number[] {
    const out: number[] = [];
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) out.push(m.get(r, c));
    return out;
}

/** One compiled NRD pipeline and its register -> reflected-name tables. */
interface Pipeline {
    pass: ComputePass;
    textures: string[];
    storageTextures: string[];
    samplers: string[];
    /** Byte offsets of the float4x4 constants (transposed on upload, see dispatch). */
    matrixOffsets: number[];
    /** Storage binding -> its read copy (setup patch for NRD's read-modify-write outputs). */
    readCopies: Map<string, string>;
}

export class NRDPass extends RenderPass {
    private enabled = true;
    private method = DenoisingMethod.RelaxDiffuseSpecular;
    private outputSizeSelection = IOSize.Default;
    private worldSpaceMotion = true;
    private maxIntensity = 1000;
    private disocclusionThreshold = 2;
    private screenSize: [number, number] = [0, 0];
    private frameIndex = 0;
    private recreateDenoiser = false;
    /** Property values applied once the library (and its settings structs) is loaded. */
    private pendingSettings: [string, unknown][] = [];

    private library: NRDLibrary | null = null;
    private common: NRDSettings | null = null;
    private relaxDiffuseSpecular: NRDSettings | null = null;
    private relaxDiffuse: NRDSettings | null = null;
    private reblur: NRDSettings | null = null;
    private desc: NRDDenoiserDesc | null = null;
    private pipelines: Pipeline[] = [];
    private samplers: Sampler[] = [];
    private permanentTextures: Texture[] = [];
    private transientTextures: Texture[] = [];
    private packedDiffuse: Texture | null = null;
    private packedSpecular: Texture | null = null;
    private packRadiancePassRelax: ComputePass | null = null;
    private packRadiancePassReblur: ComputePass | null = null;
    private prevViewMatrix: { get(r: number, c: number): number } | null = null;
    private prevProjMatrix: { get(r: number, c: number): number } | null = null;
    private shadersRegistered = false;
    /** Read copies of the read-modify-write (gInOut_*) outputs, per texture. */
    private readCopies = new Map<Texture, Texture>();
    /** Scratch buffers of the buffer-backed mip outputs (setup patch), per texture and mip. */
    private mipBuffers = new Map<string, Buffer>();
    private mipCopyPasses = new Map<string, ComputePass>();

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [key, value] of props.entries()) {
            if (key === "enabled") this.enabled = Boolean(value);
            else if (key === "method") this.method = typeof value === "string" ? DenoisingMethod[value as keyof typeof DenoisingMethod] : Number(value);
            else if (key === "outputSize") this.outputSizeSelection = parseIOSize(value as string);
            else if (key === "worldSpaceMotion") this.worldSpaceMotion = Boolean(value);
            else if (key === "disocclusionThreshold") this.disocclusionThreshold = Number(value);
            else if (key === "maxIntensity") this.maxIntensity = Number(value);
            else this.pendingSettings.push([key, value]);
        }
        if (this.method === undefined || Number.isNaN(this.method)) throw new Error("NRDPass: unknown method");
    }

    /** Loads the NRD library and registers its shader sources (native links both at build time). */
    override async initAsync(): Promise<void> {
        if (!this.library) {
            this.library = await NRDLibrary.load();
            this.common = this.library.settings("CommonSettings");
            this.relaxDiffuseSpecular = this.library.settings("RelaxDiffuseSpecularSettings");
            this.relaxDiffuse = this.library.settings("RelaxDiffuseSettings");
            this.reblur = this.library.settings("ReblurSettings");
            for (const [k, v] of Object.entries(kRelaxDiffuseSpecularDefaults)) this.relaxDiffuseSpecular.set(k, v);
            for (const [k, v] of Object.entries(kRelaxDiffuseDefaults)) this.relaxDiffuse.set(k, v);
            for (const [key, value] of this.pendingSettings) this.applyProperty(key, value);
            this.pendingSettings = [];
        }
        if (!this.shadersRegistered) {
            const res = await fetch(kShaderFileList);
            if (!res.ok) throw new Error(`NRDPass: ${kShaderFileList} missing (run node scripts/setup-web.mjs)`);
            const list = (await res.json()) as { path: string; url: string }[];
            const files: Record<string, string> = {};
            await Promise.all(list.map(async ({ path, url }) => (files[path] = await (await fetch(url)).text())));
            this.device.programManager.addShaderFiles(files);
            this.shadersRegistered = true;
        }
    }

    /** Mirrors the per-method settings branch of the constructor's property parsing. */
    private applyProperty(key: string, value: unknown): void {
        const v = typeof value === "boolean" ? value : Number(value);
        if (this.method === DenoisingMethod.RelaxDiffuseSpecular || this.method === DenoisingMethod.ReblurDiffuseSpecular) {
            if (this.relaxDiffuseSpecular!.has(key)) this.relaxDiffuseSpecular!.set(key, v);
            else Logger.warning(`Unknown property '${key}' in NRD properties.`);
        } else if (this.method === DenoisingMethod.RelaxDiffuse) {
            const field = kRelaxDiffuseKeys[key];
            if (field) this.relaxDiffuse!.set(field, v);
            else Logger.warning(`Unknown property '${key}' in NRD properties.`);
        } else {
            Logger.warning(`Unknown property '${key}' in NRD properties.`);
        }
    }

    override getProperties(): Properties {
        const props: Record<string, unknown> = {
            enabled: this.enabled,
            method: DenoisingMethod[this.method],
            outputSize: IOSize[this.outputSizeSelection],
            worldSpaceMotion: this.worldSpaceMotion,
            disocclusionThreshold: this.disocclusionThreshold,
            maxIntensity: this.maxIntensity,
        };
        if (this.library) {
            if (this.method === DenoisingMethod.RelaxDiffuseSpecular || this.method === DenoisingMethod.ReblurDiffuseSpecular) {
                for (const k of kRelaxDiffuseSpecularKeys) props[k] = this.relaxDiffuseSpecular!.get(k);
            } else if (this.method === DenoisingMethod.RelaxDiffuse) {
                for (const [k, field] of Object.entries(kRelaxDiffuseKeys)) props[k] = this.relaxDiffuse!.get(field);
            }
        } else {
            for (const [k, v] of this.pendingSettings) props[k] = v;
        }
        return new Properties(props as ConstructorParameters<typeof Properties>[0]);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSizeSelection, this.screenSize, compileData.defaultTexDims);
        const out = (name: string, desc: string, format: ResourceFormat) =>
            r.addOutput(name, desc).format(format).texture2D(w, h).bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        if (this.method === DenoisingMethod.RelaxDiffuseSpecular || this.method === DenoisingMethod.ReblurDiffuseSpecular) {
            r.addInput(kInput.diffuseRadianceHitDist, "Diffuse radiance and hit distance");
            r.addInput(kInput.specularRadianceHitDist, "Specular radiance and hit distance");
            r.addInput(kInput.viewZ, "View Z");
            r.addInput(kInput.normWRoughnessMaterialID, "World normal, roughness, and material ID");
            r.addInput(kInput.mvec, "Motion vectors");
            out(kOutput.filteredDiffuseRadianceHitDist, "Filtered diffuse radiance and hit distance", ResourceFormat.RGBA16Float);
            out(kOutput.filteredSpecularRadianceHitDist, "Filtered specular radiance and hit distance", ResourceFormat.RGBA16Float);
        } else if (this.method === DenoisingMethod.RelaxDiffuse) {
            r.addInput(kInput.diffuseRadianceHitDist, "Diffuse radiance and hit distance");
            r.addInput(kInput.viewZ, "View Z");
            r.addInput(kInput.normWRoughnessMaterialID, "World normal, roughness, and material ID");
            r.addInput(kInput.mvec, "Motion vectors");
            out(kOutput.filteredDiffuseRadianceHitDist, "Filtered diffuse radiance and hit distance", ResourceFormat.RGBA16Float);
        } else if (this.method === DenoisingMethod.SpecularReflectionMv) {
            r.addInput(kInput.specularHitDist, "Specular hit distance");
            r.addInput(kInput.viewZ, "View Z");
            r.addInput(kInput.normWRoughnessMaterialID, "World normal, roughness, and material ID");
            r.addInput(kInput.mvec, "Motion vectors");
            out(kOutput.reflectionMvec, "Reflection motion vectors in screen space", ResourceFormat.RG32Float);
        } else {
            r.addInput(kInput.deltaPrimaryPosW, "Delta primary world position");
            r.addInput(kInput.deltaSecondaryPosW, "Delta secondary world position");
            r.addInput(kInput.mvec, "Motion vectors");
            out(kOutput.deltaMvec, "Delta motion vectors in screen space", ResourceFormat.RG32Float);
        }
        return r;
    }

    override compile(_ctx: RenderContext, compileData: CompileData): void {
        this.screenSize = calculateIOSize(this.outputSizeSelection, this.screenSize, compileData.defaultTexDims);
        if (this.screenSize[0] === 0 || this.screenSize[1] === 0) this.screenSize = compileData.defaultTexDims;
        this.frameIndex = 0;
        this.desc = null; // reinit on first execute (the library loads asynchronously)
    }

    override setScene(scene: Scene | null): void {
        super.setScene(scene);
    }

    private nrdMethod(): NRDMethod {
        switch (this.method) {
            case DenoisingMethod.RelaxDiffuseSpecular: return NRDMethod.RELAX_DIFFUSE_SPECULAR;
            case DenoisingMethod.RelaxDiffuse: return NRDMethod.RELAX_DIFFUSE;
            case DenoisingMethod.ReblurDiffuseSpecular: return NRDMethod.REBLUR_DIFFUSE_SPECULAR;
            case DenoisingMethod.SpecularReflectionMv: return NRDMethod.SPECULAR_REFLECTION_MV;
            default: return NRDMethod.SPECULAR_DELTA_MV;
        }
    }

    /** Mirrors reinit: denoiser, pools, samplers and pipelines. */
    private reinit(): void {
        const lib = this.library!;
        lib.createDenoiser(this.nrdMethod(), this.screenSize[0], this.screenSize[1]);
        const desc = (this.desc = lib.getDenoiserDesc());
        this.samplers = desc.samplers.map(({ sampler }) => {
            const clamp = sampler === NRDSampler.NEAREST_CLAMP || sampler === NRDSampler.LINEAR_CLAMP;
            const nearest = sampler === NRDSampler.NEAREST_CLAMP || sampler === NRDSampler.NEAREST_MIRRORED_REPEAT;
            const address = clamp ? TextureAddressingMode.Clamp : TextureAddressingMode.Mirror;
            const filter = nearest ? TextureFilteringMode.Point : TextureFilteringMode.Linear;
            return new Sampler(this.device, { magFilter: filter, minFilter: filter, mipFilter: TextureFilteringMode.Point, addressModeU: address, addressModeV: address, addressModeW: address });
        });
        const make = (t: { format: number; width: number; height: number; mipNum: number }, name: string) =>
            new Texture(this.device, { type: ResourceType.Texture2D, width: t.width, height: t.height, format: poolFormat(t.format), mipLevels: t.mipNum, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, name });
        this.permanentTextures = desc.permanentPool.map((t, i) => make(t, `NRD::permanent${i}`));
        this.transientTextures = desc.transientPool.map((t, i) => make(t, `NRD::transient${i}`));
        this.pipelines = desc.pipelines.map((p) => this.createPipeline(p.shader, p.entry));
        this.readCopies.clear();
        this.mipBuffers.clear();
        this.recreateDenoiser = false;
    }

    /** Mirrors createPipelines for one pipeline; the reflected declaration order gives the register tables. */
    private createPipeline(shader: string, entry: string): Pipeline {
        const pass = ComputePass.create(this.device, {
            path: `nrd/Shaders/Source/${shader}.hlsl`,
            csEntry: entry,
            defines: { NRD_COMPILER_DXC: 1, NRD_USE_OCT_NORMAL_ENCODING: 1, NRD_USE_MATERIAL_ID: 0 },
        });
        const params = (pass.getReflector().json as { parameters?: { name: string; binding?: { kind: string; index?: number }; type: { kind: string; baseShape?: string; access?: string } }[] }).parameters ?? [];
        const byIndex = params.filter((p) => p.binding?.kind === "descriptorTableSlot").sort((a, b) => (a.binding!.index ?? 0) - (b.binding!.index ?? 0));
        const isStorage = (p: (typeof params)[number]) => p.type.kind === "resource" && p.type.access !== undefined && p.type.access !== "read";
        // Matrix constants of the globalConstants cbuffer.
        type Field = { name: string; type: { kind: string; rowCount?: number; columnCount?: number }; binding?: { offset?: number } };
        const cb = params.find((p) => p.name === "globalConstants") as unknown as { type: { elementType?: { fields?: Field[] } } } | undefined;
        const matrixOffsets = (cb?.type.elementType?.fields ?? []).filter((f) => f.type.kind === "matrix" && f.type.rowCount === 4 && f.type.columnCount === 4).map((f) => f.binding?.offset ?? 0);
        const names = new Set(byIndex.map((p) => p.name));
        const readCopies = new Map<string, string>();
        for (const p of byIndex.filter(isStorage)) {
            const copy = p.name.endsWith("__out") ? `${p.name.slice(0, -5)}__in` : `${p.name}__in`;
            if (names.has(copy)) readCopies.set(p.name, copy);
        }
        return {
            pass,
            matrixOffsets,
            readCopies,
            // The __in read copies of the gInOut_* outputs (setup patch) are not NRD registers.
            textures: byIndex.filter((p) => p.type.kind === "resource" && !isStorage(p) && !p.name.endsWith("__in")).map((p) => p.name),
            storageTextures: byIndex.filter(isStorage).map((p) => p.name),
            samplers: byIndex.filter((p) => p.type.kind === "samplerState").map((p) => p.name),
        };
    }

    private inputTexture(renderData: RenderData, type: NRDResourceType): Texture | undefined {
        switch (type) {
            case NRDResourceType.IN_MV: return renderData.getTexture(kInput.mvec);
            case NRDResourceType.IN_NORMAL_ROUGHNESS: return renderData.getTexture(kInput.normWRoughnessMaterialID);
            case NRDResourceType.IN_VIEWZ: return renderData.getTexture(kInput.viewZ);
            // Packed copies (native packs the inputs in place).
            case NRDResourceType.IN_DIFF_RADIANCE_HITDIST: return this.packedDiffuse ?? undefined;
            case NRDResourceType.IN_SPEC_RADIANCE_HITDIST: return this.packedSpecular ?? undefined;
            case NRDResourceType.IN_SPEC_HITDIST: return renderData.getTexture(kInput.specularHitDist);
            case NRDResourceType.IN_DELTA_PRIMARY_POS: return renderData.getTexture(kInput.deltaPrimaryPosW);
            case NRDResourceType.IN_DELTA_SECONDARY_POS: return renderData.getTexture(kInput.deltaSecondaryPosW);
            case NRDResourceType.OUT_DIFF_RADIANCE_HITDIST: return renderData.getTexture(kOutput.filteredDiffuseRadianceHitDist);
            case NRDResourceType.OUT_SPEC_RADIANCE_HITDIST: return renderData.getTexture(kOutput.filteredSpecularRadianceHitDist);
            case NRDResourceType.OUT_REFLECTION_MV: return renderData.getTexture(kOutput.reflectionMvec);
            case NRDResourceType.OUT_DELTA_MV: return renderData.getTexture(kOutput.deltaMvec);
            default: return undefined;
        }
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        if (!this.library) throw new Error("NRDPass: initAsync() has not completed");
        if (this.enabled) this.executeInternal(ctx, renderData);
        else this.passThrough(ctx, renderData);
    }

    /** Mirrors execute's disabled branch: inputs blit through, motion vectors clear or copy. */
    private passThrough(ctx: RenderContext, renderData: RenderData): void {
        const blit = (src: string, dst: string) => ctx.blit(renderData.getTexture(src)!, renderData.getTexture(dst)!);
        if (this.method === DenoisingMethod.RelaxDiffuseSpecular || this.method === DenoisingMethod.ReblurDiffuseSpecular) {
            blit(kInput.diffuseRadianceHitDist, kOutput.filteredDiffuseRadianceHitDist);
            blit(kInput.specularRadianceHitDist, kOutput.filteredSpecularRadianceHitDist);
        } else if (this.method === DenoisingMethod.RelaxDiffuse) {
            blit(kInput.diffuseRadianceHitDist, kOutput.filteredDiffuseRadianceHitDist);
        } else {
            const out = this.method === DenoisingMethod.SpecularReflectionMv ? kOutput.reflectionMvec : kOutput.deltaMvec;
            if (this.worldSpaceMotion) ctx.clearTexture(renderData.getTexture(out)!);
            else blit(kInput.mvec, out);
        }
    }

    private packRadiance(ctx: RenderContext, renderData: RenderData, reblur: boolean): void {
        const [w, h] = this.screenSize;
        const tex = (name: string) =>
            new Texture(this.device, { type: ResourceType.Texture2D, width: w, height: h, format: ResourceFormat.RGBA32Float, mipLevels: 1, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, name });
        if (!this.packedDiffuse || this.packedDiffuse.width !== w || this.packedDiffuse.height !== h) {
            this.packedDiffuse = tex("NRD::packedDiffuse");
            this.packedSpecular = tex("NRD::packedSpecular");
        }
        const create = (method: number) =>
            ComputePass.create(this.device, { path: kShaderPackRadiance, defines: { NRD_USE_OCT_NORMAL_ENCODING: 1, NRD_USE_MATERIAL_ID: 0, NRD_METHOD: method } });
        const pass = reblur ? (this.packRadiancePassReblur ??= create(1)) : (this.packRadiancePassRelax ??= create(0));
        const cb = pass.getRootVar()["PerImageCB"] as ShaderVar;
        cb["gMaxIntensity"] = this.maxIntensity;
        const diffuse = renderData.getTexture(kInput.diffuseRadianceHitDist)!;
        cb["gDiffuseRadianceHitDist"] = diffuse;
        // RelaxDiffuse has no specular input: the specular slot packs the diffuse again (discarded).
        cb["gSpecularRadianceHitDist"] = renderData.getTexture(kInput.specularRadianceHitDist) ?? diffuse;
        cb["gPackedDiffuseRadianceHitDist"] = this.packedDiffuse;
        cb["gPackedSpecularRadianceHitDist"] = this.packedSpecular!;
        if (reblur) {
            const hd = ["A", "B", "C", "D"].map((k) => this.reblur!.get(`hitDistanceParameters.${k}`) as number);
            cb["gHitDistParams"] = hd;
            cb["gNormalRoughness"] = renderData.getTexture(kInput.normWRoughnessMaterialID)!;
            cb["gViewZ"] = renderData.getTexture(kInput.viewZ)!;
        }
        pass.execute(ctx, w, h);
    }

    private executeInternal(ctx: RenderContext, renderData: RenderData): void {
        const lib = this.library!;
        if (!this.desc || this.recreateDenoiser) this.reinit();
        const method = this.nrdMethod();
        if (this.method === DenoisingMethod.RelaxDiffuseSpecular || this.method === DenoisingMethod.RelaxDiffuse) this.packRadiance(ctx, renderData, false);
        else if (this.method === DenoisingMethod.ReblurDiffuseSpecular) this.packRadiance(ctx, renderData, true);
        // Native passes default-constructed settings for the motion-vector methods.
        lib.setMethodSettings(method);

        // Common settings (native executeInternal).
        const camera = this.scene!.camera;
        const view = camera.getViewMatrix();
        const data = camera.getData();
        const proj = perspective(camera.getFovY(), data.aspectRatio, data.nearZ, data.farZ);
        if (this.frameIndex === 0) {
            this.prevViewMatrix = view;
            this.prevProjMatrix = proj;
        }
        const common = this.common!;
        common.set("viewToClipMatrix", columnMajor(proj));
        common.set("viewToClipMatrixPrev", columnMajor(this.prevProjMatrix!));
        common.set("worldToViewMatrix", columnMajor(view));
        common.set("worldToViewMatrixPrev", columnMajor(this.prevViewMatrix!));
        // NRD's jitter convention: [-0.5; 0.5], sampleUv = pixelUv + cameraJitter.
        common.set("cameraJitter", [-data.jitterX, data.jitterY]);
        common.set("denoisingRange", kNRDDepthRange);
        common.set("disocclusionThreshold", this.disocclusionThreshold * 0.01);
        common.set("frameIndex", this.frameIndex);
        common.set("isMotionVectorInWorldSpace", this.worldSpaceMotion);
        this.prevViewMatrix = view;
        this.prevProjMatrix = proj;
        this.frameIndex++;

        for (const d of lib.getComputeDispatches()) this.dispatch(ctx, renderData, d);
    }

    /** Mirrors dispatch: constants, resources by descriptor range, samplers, grid. */
    private dispatch(ctx: RenderContext, renderData: RenderData, d: NRDDispatch): void {
        const desc = this.desc!;
        const pipelineDesc = desc.pipelines[d.pipelineIndex]!;
        const pipeline = this.pipelines[d.pipelineIndex]!;
        const root = pipeline.pass.getRootVar();
        if (pipelineDesc.hasConstantData) {
            // NRD uploads column-major matrices (native compiles with MatrixLayoutColumnMajor);
            // the WGSL kernels read row-major, so transpose each float4x4 in the blob.
            const blob = d.constants.slice();
            const f32 = new Float32Array(blob.buffer, 0, blob.byteLength >> 2);
            for (const off of pipeline.matrixOffsets) {
                const m = f32.slice(off >> 2, (off >> 2) + 16);
                for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) f32[(off >> 2) + r * 4 + c] = m[c * 4 + r]!;
            }
            pipeline.pass.getParameterBlock().setCBufferBlob("globalConstants", blob);
        }
        pipeline.samplers.forEach((name, i) => (root[name] = this.samplers[desc.samplers.findIndex((s) => s.register === i)] ?? this.samplers[i]!));
        const pendingMipCopies: { buffer: Buffer; texture: Texture; mip: number }[] = [];
        let resourceIndex = 0;
        for (const range of pipelineDesc.ranges) {
            for (let k = 0; k < range.count; k++) {
                const res = d.resources[resourceIndex++]!;
                let texture: Texture | undefined;
                if (res.type === NRDResourceType.TRANSIENT_POOL) texture = this.transientTextures[res.indexInPool];
                else if (res.type === NRDResourceType.PERMANENT_POOL) texture = this.permanentTextures[res.indexInPool];
                else texture = this.inputTexture(renderData, res.type);
                if (!texture) throw new Error(`NRDPass: no texture for NRD resource type ${NRDResourceType[res.type]} (${d.name})`);
                const names = range.type === NRDDescriptorType.TEXTURE ? pipeline.textures : pipeline.storageTextures;
                const name = names[range.base + k];
                if (!name) throw new Error(`NRDPass: ${pipelineDesc.shader} has no ${range.type === NRDDescriptorType.TEXTURE ? "t" : "u"}${range.base + k}`);
                // SRVs see the resource's mip range, UAVs its first mip (native getSRV/getUAV). Whole
                // textures bind as themselves, so storage formats retarget to them (ComputePass).
                const whole = res.mipOffset === 0 && (range.type === NRDDescriptorType.TEXTURE ? res.mipNum >= texture.mipCount : texture.mipCount === 1);
                const view = whole ? texture : range.type === NRDDescriptorType.TEXTURE ? texture.getSRV(res.mipOffset, res.mipNum) : texture.getUAV(res.mipOffset);
                if (name.endsWith("__buf")) {
                    // Buffer-backed mip output (setup patch): written to a buffer, copied after the dispatch.
                    const key = `${texture.name}@${res.mipOffset}`;
                    let buffer = this.mipBuffers.get(key);
                    if (!buffer) {
                        buffer = new Buffer(this.device, { size: this.screenSize[0] * this.screenSize[1] * 16, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: `NRD::${key}` });
                        this.mipBuffers.set(key, buffer);
                    }
                    root[name] = buffer;
                    pendingMipCopies.push({ buffer, texture, mip: res.mipOffset });
                    continue;
                }
                try {
                    root[name] = view;
                } catch {
                    /* declared but unused in this kernel (stripped) */
                }
                // Read-modify-write outputs (setup patch): the kernel reads the pre-dispatch state.
                const copyName = range.type === NRDDescriptorType.STORAGE_TEXTURE ? pipeline.readCopies.get(name) : undefined;
                if (copyName) {
                    let copy = this.readCopies.get(texture);
                    if (!copy) {
                        copy = new Texture(this.device, { type: ResourceType.Texture2D, width: texture.width, height: texture.height, format: texture.format, mipLevels: 1, bindFlags: ResourceBindFlags.ShaderResource, name: `${texture.name}::readCopy` });
                        this.readCopies.set(texture, copy);
                    }
                    ctx.copySubresource(copy, 0, 0, texture, res.mipOffset, 0);
                    try {
                        root[copyName] = copy;
                    } catch {
                        /* stripped */
                    }
                }
            }
        }
        if (pendingMipCopies.length > 0) root["gWebFalcorBufStride"] = this.screenSize[0];
        const [gx, gy] = pipeline.pass.getThreadGroupSize();
        pipeline.pass.execute(ctx, d.gridWidth * gx, d.gridHeight * gy);
        for (const c of pendingMipCopies) this.copyBufferToMip(ctx, c.buffer, c.texture, c.mip);
    }

    /** Copies a buffer-backed mip output into its texture mip (row stride = screen width). */
    private copyBufferToMip(ctx: RenderContext, buffer: Buffer, texture: Texture, mip: number): void {
        const kFormats: Partial<Record<ResourceFormat, [string, string, string]>> = {
            [ResourceFormat.RGBA16Float]: ["rgba16f", "float4", ""],
            [ResourceFormat.RGBA32Float]: ["rgba32f", "float4", ""],
            [ResourceFormat.R32Float]: ["r32f", "float", ".x"],
            [ResourceFormat.RG32Float]: ["rg32f", "float2", ".xy"],
        };
        const f = kFormats[texture.format];
        if (!f) throw new Error(`NRDPass: no mip copy for format ${ResourceFormat[texture.format]}`);
        let pass = this.mipCopyPasses.get(f[0]);
        if (!pass) {
            const source = `StructuredBuffer<float4> gSrc;\n[format("${f[0]}")] RWTexture2D<${f[1]}> gDst;\ncbuffer CB { uint2 gDim; uint gStride; };\n[numthreads(16, 16, 1)]\nvoid main(uint3 id: SV_DispatchThreadID)\n{\n    if (any(id.xy >= gDim)) return;\n    gDst[id.xy] = gSrc[id.y * gStride + id.x]${f[2]};\n}\n`;
            pass = ComputePass.create(this.device, { modules: [{ sources: [{ string: source, path: `WebFalcor/NRDMipCopy_${f[0]}.cs.slang` }] }], csEntry: "main" });
            this.mipCopyPasses.set(f[0], pass);
        }
        const [w, h] = [Math.max(1, texture.width >> mip), Math.max(1, texture.height >> mip)];
        const root = pass.getRootVar();
        root["gSrc"] = buffer;
        root["gDst"] = texture.mipCount === 1 ? texture : texture.getUAV(mip);
        const cb = root["CB"] as ShaderVar;
        cb["gDim"] = [w, h];
        cb["gStride"] = this.screenSize[0];
        pass.execute(ctx, w, h);
    }

    override renderUI(ui: UIWidgets): void {
        const v = this.library?.version ?? [3, 1, 0];
        ui.text(`NRD Library v${v[0]}.${v[1]}.${v[2]}`);
        ui.checkbox("Enabled", this.enabled, (x) => (this.enabled = x));
        if (this.method === DenoisingMethod.RelaxDiffuseSpecular || this.method === DenoisingMethod.ReblurDiffuseSpecular) {
            ui.dropdown("Denoising method", ["ReLAX", "ReBLUR"], this.method === DenoisingMethod.RelaxDiffuseSpecular ? "ReLAX" : "ReBLUR", (x: string) => {
                this.method = x === "ReLAX" ? DenoisingMethod.RelaxDiffuseSpecular : DenoisingMethod.ReblurDiffuseSpecular;
                this.recreateDenoiser = true;
            });
        }
        const motion = () => ui.text(this.worldSpaceMotion ? "Motion: world space" : "Motion: screen space");
        const common = () => {
            ui.text("Common:");
            motion();
            ui.slider("Disocclusion threshold (%)", this.disocclusionThreshold, 0, 5, 0.01, (x) => (this.disocclusionThreshold = x));
            ui.text("Pack radiance:");
            ui.slider("Max intensity", this.maxIntensity, 0, 100000, 1, (x) => (this.maxIntensity = x));
        };
        if (!this.library) return;
        const slider = (s: NRDSettings, label: string, field: string, min: number, max: number, step: number) =>
            ui.slider(label, s.get(field) as number, min, max, step, (x) => s.set(field, x));
        const check = (s: NRDSettings, label: string, field: string) => ui.checkbox(label, Boolean(s.get(field)), (x) => s.set(field, x));
        if (this.method === DenoisingMethod.RelaxDiffuseSpecular) {
            common();
            const s = this.relaxDiffuseSpecular!;
            const g = ui.group("ReLAX Diffuse/Specular");
            g.text("Prepass:");
            slider(s, "Specular blur radius", "specularPrepassBlurRadius", 0, 100, 1);
            slider(s, "Diffuse blur radius", "diffusePrepassBlurRadius", 0, 100, 1);
            g.text("Reprojection:");
            slider(s, "Specular max accumulated frames", "specularMaxAccumulatedFrameNum", 0, kRelaxMaxHistoryFrameNum, 1);
            slider(s, "Specular responsive max accumulated frames", "specularMaxFastAccumulatedFrameNum", 0, kRelaxMaxHistoryFrameNum, 1);
            slider(s, "Diffuse max accumulated frames", "diffuseMaxAccumulatedFrameNum", 0, kRelaxMaxHistoryFrameNum, 1);
            slider(s, "Diffuse responsive max accumulated frames", "diffuseMaxFastAccumulatedFrameNum", 0, kRelaxMaxHistoryFrameNum, 1);
            slider(s, "Specular variance boost", "specularVarianceBoost", 0, 8, 0.1);
            slider(s, "Diffuse history rejection normal threshold", "diffuseHistoryRejectionNormalThreshold", 0, 1, 0.01);
            check(s, "Reprojection test skipping without motion", "enableReprojectionTestSkippingWithoutMotion");
            check(s, "Specular virtual history clamping", "enableSpecularVirtualHistoryClamping");
            g.text("Disocclusion fix:");
            slider(s, "Edge stopping normal power", "disocclusionFixEdgeStoppingNormalPower", 0, 128, 0.1);
            slider(s, "Max kernel radius", "disocclusionFixMaxRadius", 0, 100, 1);
            slider(s, "Frames to fix", "disocclusionFixNumFramesToFix", 0, 100, 1);
            g.text("History clamping & antilag:");
            slider(s, "Color clamping sigma", "historyClampingColorBoxSigmaScale", 0, 10, 0.1);
            g.text("Spatial variance estimation:");
            slider(s, "History threshold", "spatialVarianceEstimationHistoryThreshold", 0, 10, 1);
            g.text("Firefly filter:");
            check(s, "Enable firefly filter", "enableAntiFirefly");
            g.text("Spatial filter:");
            slider(s, "A-trous iterations", "atrousIterationNum", 2, 8, 1);
            slider(s, "Specular luminance weight (sigma scale)", "specularPhiLuminance", 0, 10, 0.1);
            slider(s, "Diffuse luminance weight (sigma scale)", "diffusePhiLuminance", 0, 10, 0.1);
            slider(s, "Min luminance weight", "minLuminanceWeight", 0, 1, 0.01);
            slider(s, "Depth weight (relative fraction)", "depthThreshold", 0, 0.05, 0.01);
            slider(s, "Roughness weight (relative fraction)", "roughnessFraction", 0, 2, 0.01);
            slider(s, "Diffuse lobe angle fraction", "diffuseLobeAngleFraction", 0, 2, 0.1);
            slider(s, "Specular loba angle fraction", "specularLobeAngleFraction", 0, 2, 0.1);
            slider(s, "Specular normal weight (degrees of slack)", "specularLobeAngleSlack", 0, 180, 1);
            slider(s, "Roughness relaxation", "roughnessEdgeStoppingRelaxation", 0, 1, 0.01);
            slider(s, "Normal relaxation", "normalEdgeStoppingRelaxation", 0, 1, 0.01);
            slider(s, "Luminance relaxation", "luminanceEdgeStoppingRelaxation", 0, 1, 0.01);
            check(s, "Roughness edge stopping", "enableRoughnessEdgeStopping");
        } else if (this.method === DenoisingMethod.RelaxDiffuse) {
            common();
            const s = this.relaxDiffuse!;
            const g = ui.group("ReLAX Diffuse");
            g.text("Prepass:");
            slider(s, "Diffuse blur radius", "prepassBlurRadius", 0, 100, 1);
            g.text("Reprojection:");
            slider(s, "Diffuse max accumulated frames", "diffuseMaxAccumulatedFrameNum", 0, kRelaxMaxHistoryFrameNum, 1);
            slider(s, "Diffuse responsive max accumulated frames", "diffuseMaxFastAccumulatedFrameNum", 0, kRelaxMaxHistoryFrameNum, 1);
            slider(s, "Diffuse history rejection normal threshold", "diffuseHistoryRejectionNormalThreshold", 0, 1, 0.01);
            check(s, "Reprojection test skipping without motion", "enableReprojectionTestSkippingWithoutMotion");
            g.text("Disocclusion fix:");
            slider(s, "Edge stopping normal power", "disocclusionFixEdgeStoppingNormalPower", 0, 128, 0.1);
            slider(s, "Max kernel radius", "disocclusionFixMaxRadius", 0, 100, 1);
            slider(s, "Frames to fix", "disocclusionFixNumFramesToFix", 0, 100, 1);
            g.text("History clamping & antilag:");
            slider(s, "Color clamping sigma", "historyClampingColorBoxSigmaScale", 0, 10, 0.1);
            g.text("Spatial variance estimation:");
            slider(s, "History threshold", "spatialVarianceEstimationHistoryThreshold", 0, 10, 1);
            g.text("Firefly filter:");
            check(s, "Enable firefly filter", "enableAntiFirefly");
            g.text("Spatial filter:");
            slider(s, "A-trous iterations", "atrousIterationNum", 2, 8, 1);
            slider(s, "Diffuse luminance weight (sigma scale)", "diffusePhiLuminance", 0, 10, 0.1);
            slider(s, "Min luminance weight", "minLuminanceWeight", 0, 1, 0.01);
            slider(s, "Depth weight (relative fraction)", "depthThreshold", 0, 0.05, 0.01);
            slider(s, "Diffuse lobe angle fraction", "diffuseLobeAngleFraction", 0, 2, 0.1);
        } else if (this.method === DenoisingMethod.ReblurDiffuseSpecular) {
            common();
            const s = this.reblur!;
            const kEpsilon = 0.0001;
            const g = ui.group("ReBLUR Diffuse/Specular");
            const slt = g.group("Specular lobe trimming");
            slt.slider("A", s.get("specularLobeTrimmingParameters.A") as number, -256, 256, 0.01, (x) => s.set("specularLobeTrimmingParameters.A", x));
            slt.slider("B", s.get("specularLobeTrimmingParameters.B") as number, kEpsilon, 256, 0.01, (x) => s.set("specularLobeTrimmingParameters.B", x));
            slt.slider("C", s.get("specularLobeTrimmingParameters.C") as number, 1, 256, 0.01, (x) => s.set("specularLobeTrimmingParameters.C", x));
            const hd = g.group("Hit distance");
            for (const [k, lo, hi] of [["A", -256, 256], ["B", kEpsilon, 256], ["C", 1, 256], ["D", -256, 0]] as const) hd.slider(k, s.get(`hitDistanceParameters.${k}`) as number, lo, hi, 0.01, (x) => s.set(`hitDistanceParameters.${k}`, x));
            for (const [label, prefix, dark] of [["Antilag intensity", "antilagIntensitySettings", 256], ["Antilag hit distance", "antilagHitDistanceSettings", 1]] as const) {
                const a = g.group(label);
                a.slider("Threshold min", s.get(`${prefix}.thresholdMin`) as number, 0, 1, 0.01, (x) => s.set(`${prefix}.thresholdMin`, x));
                a.slider("Threshold max", s.get(`${prefix}.thresholdMax`) as number, 0, 1, 0.01, (x) => s.set(`${prefix}.thresholdMax`, x));
                a.slider("Sigma scale", s.get(`${prefix}.sigmaScale`) as number, kEpsilon, 16, 0.01, (x) => s.set(`${prefix}.sigmaScale`, x));
                a.slider("Sensitivity to darkness", s.get(`${prefix}.sensitivityToDarkness`) as number, kEpsilon, dark, 0.01, (x) => s.set(`${prefix}.sensitivityToDarkness`, x));
                a.checkbox("Enable", Boolean(s.get(`${prefix}.enable`)), (x) => s.set(`${prefix}.enable`, x));
            }
            slider(s, "Max accumulated frame num", "maxAccumulatedFrameNum", 0, kReblurMaxHistoryFrameNum, 1);
            slider(s, "Blur radius", "blurRadius", 0, 256, 0.01);
            slider(s, "Min converged state base radius scale", "minConvergedStateBaseRadiusScale", 0, 1, 0.01);
            slider(s, "Max adaptive radius scale", "maxAdaptiveRadiusScale", 0, 10, 0.01);
            slider(s, "Normal weight (fraction of lobe)", "lobeAngleFraction", 0, 1, 0.01);
            slider(s, "Roughness weight (fraction)", "roughnessFraction", 0, 1, 0.01);
            slider(s, "Responsive accumulation roughness threshold", "responsiveAccumulationRoughnessThreshold", 0, 1, 0.01);
            slider(s, "Stabilization strength", "stabilizationStrength", 0, 1, 0.01);
            slider(s, "History fix strength", "historyFixStrength", 0, 1, 0.01);
            slider(s, "Plane distance sensitivity", "planeDistanceSensitivity", kEpsilon, 16, 0.001);
            slider(s, "Input mix", "inputMix", 0, 1, 0.01);
            slider(s, "Residual noise level", "residualNoiseLevel", 0.01, 0.1, 0.01);
            check(s, "Antifirefly", "enableAntiFirefly");
            check(s, "Reference accumulation", "enableReferenceAccumulation");
            check(s, "Performance mode", "enablePerformanceMode");
            check(s, "Material test for diffuse", "enableMaterialTestForDiffuse");
            check(s, "Material test for specular", "enableMaterialTestForSpecular");
        } else {
            motion();
        }
    }
}

registerRenderPass("NRD", (device, props) => new NRDPass(device, props));
