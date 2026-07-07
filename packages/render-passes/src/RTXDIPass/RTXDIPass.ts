/**
 * Standalone ReSTIR direct-illumination pass mirroring
 * Source/RenderPasses/RTXDIPass (PrepareSurfaceData + FinalShading around the
 * Rendering/RTXDI module, which carries the actual RTXDI SDK resampling).
 *
 * Web divergences (documented, DESIGN.md §RenderPasses):
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
