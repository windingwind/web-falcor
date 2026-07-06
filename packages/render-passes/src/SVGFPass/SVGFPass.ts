/**
 * SVGF denoiser mirroring Source/RenderPasses/SVGFPass: demodulated
 * illumination reprojection, moments filtering and a-trous wavelet
 * decomposition. All five kernels run verbatim (no overrides).
 */

import {
    Fbo,
    FullScreenPass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type ShaderVar,
} from "@web-falcor/falcor";

const kPackLinearZAndNormalShader = "RenderPasses/SVGFPass/SVGFPackLinearZAndNormal.ps.slang";
const kReprojectShader = "RenderPasses/SVGFPass/SVGFReproject.ps.slang";
const kAtrousShader = "RenderPasses/SVGFPass/SVGFAtrous.ps.slang";
const kFilterMomentShader = "RenderPasses/SVGFPass/SVGFFilterMoments.ps.slang";
const kFinalModulateShader = "RenderPasses/SVGFPass/SVGFFinalModulate.ps.slang";

const kInternalPrevLinearZAndNormal = "Previous Linear Z and Packed Normal";
const kInternalPrevLighting = "Previous Lighting";
const kInternalPrevMoments = "Previous Moments";

/** One render target set mirroring native's Fbo usage (texture + Fbo pair). */
interface TargetFbo {
    fbo: Fbo;
    textures: Texture[];
}

export class SVGFPass extends RenderPass {
    private filterEnabled = true;
    private filterIterations = 4;
    private feedbackTap = 1;
    private varianceEpsilon = 1e-4;
    private phiColor = 10.0;
    private phiNormal = 128.0;
    private alpha = 0.05;
    private momentsAlpha = 0.2;

    private packLinearZAndNormal: FullScreenPass | null = null;
    private reprojection: FullScreenPass | null = null;
    private atrous: FullScreenPass | null = null;
    private filterMoments: FullScreenPass | null = null;
    private finalModulate: FullScreenPass | null = null;

    private curReprojFbo: TargetFbo | null = null;
    private prevReprojFbo: TargetFbo | null = null;
    private linearZAndNormalFbo: TargetFbo | null = null;
    private pingPongFbo: [TargetFbo, TargetFbo] | null = null;
    private filteredPastFbo: TargetFbo | null = null;
    private filteredIlluminationFbo: TargetFbo | null = null;
    private finalFbo: TargetFbo | null = null;
    private buffersNeedClear = true;

    constructor(device: Device, props: Properties) {
        super(device);
        this.filterEnabled = props.get("Enabled", true);
        this.filterIterations = props.get("Iterations", 4);
        this.feedbackTap = props.get("FeedbackTap", 1);
        this.varianceEpsilon = props.get("VarianceEpsilon", 1e-4);
        this.phiColor = props.get("PhiColor", 10.0);
        this.phiNormal = props.get("PhiNormal", 128.0);
        this.alpha = props.get("Alpha", 0.05);
        this.momentsAlpha = props.get("MomentsAlpha", 0.2);
    }

    override getProperties(): Properties {
        return new Properties({
            Enabled: this.filterEnabled,
            Iterations: this.filterIterations,
            FeedbackTap: this.feedbackTap,
            VarianceEpsilon: this.varianceEpsilon,
            PhiColor: this.phiColor,
            PhiNormal: this.phiNormal,
            Alpha: this.alpha,
            MomentsAlpha: this.momentsAlpha,
        });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("Albedo", "Albedo");
        r.addInput("Color", "Color");
        r.addInput("Emission", "Emission");
        r.addInput("WorldPosition", "World Position");
        r.addInput("WorldNormal", "World Normal");
        r.addInput("PositionNormalFwidth", "PositionNormalFwidth");
        r.addInput("LinearZ", "LinearZ");
        r.addInput("MotionVec", "Motion vectors");
        r.addInternal(kInternalPrevLinearZAndNormal, "Previous Linear Z and Packed Normal")
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        r.addInternal(kInternalPrevLighting, "Previous Filtered Lighting")
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        r.addInternal(kInternalPrevMoments, "Previous Moments")
            .format(ResourceFormat.RG32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        r.addOutput("Filtered image", "Filtered image").format(ResourceFormat.RGBA16Float);
        return r;
    }

    override compile(_ctx: RenderContext, compileData: CompileData): void {
        this.allocateFbos(compileData.defaultTexDims);
        this.buffersNeedClear = true;
    }

    private makeTargetFbo(dims: [number, number], formats: ResourceFormat[], name: string): TargetFbo {
        const fbo = new Fbo();
        const textures = formats.map(
            (format, i) =>
                new Texture(this.device, {
                    type: ResourceType.Texture2D,
                    width: dims[0],
                    height: dims[1],
                    format,
                    bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
                    name: `SVGF::${name}${formats.length > 1 ? `.${i}` : ""}`,
                }),
        );
        textures.forEach((t, i) => fbo.attachColorTarget(t, i));
        return { fbo, textures };
    }

    /** Mirrors SVGFPass::allocateFbos. */
    private allocateFbos(dims: [number, number]): void {
        const reprojFormats = [ResourceFormat.RGBA32Float, ResourceFormat.RG32Float, ResourceFormat.R16Float];
        this.curReprojFbo = this.makeTargetFbo(dims, reprojFormats, "curReproj");
        this.prevReprojFbo = this.makeTargetFbo(dims, reprojFormats, "prevReproj");
        this.linearZAndNormalFbo = this.makeTargetFbo(dims, [ResourceFormat.RGBA32Float], "linearZAndNormal");
        this.pingPongFbo = [
            this.makeTargetFbo(dims, [ResourceFormat.RGBA32Float], "pingPong0"),
            this.makeTargetFbo(dims, [ResourceFormat.RGBA32Float], "pingPong1"),
        ];
        this.filteredPastFbo = this.makeTargetFbo(dims, [ResourceFormat.RGBA32Float], "filteredPast");
        this.filteredIlluminationFbo = this.makeTargetFbo(dims, [ResourceFormat.RGBA32Float], "filteredIllumination");
        this.finalFbo = this.makeTargetFbo(dims, [ResourceFormat.RGBA32Float], "final");
        this.buffersNeedClear = true;
    }

    private clearBuffers(ctx: RenderContext, renderData: RenderData): void {
        const clearFbo = (t: TargetFbo | null) => t?.textures.forEach((tex) => ctx.clearTexture(tex));
        clearFbo(this.pingPongFbo![0]);
        clearFbo(this.pingPongFbo![1]);
        clearFbo(this.linearZAndNormalFbo);
        clearFbo(this.filteredPastFbo);
        clearFbo(this.curReprojFbo);
        clearFbo(this.prevReprojFbo);
        clearFbo(this.filteredIlluminationFbo);
        ctx.clearTexture(renderData.getTexture(kInternalPrevLinearZAndNormal)!);
        ctx.clearTexture(renderData.getTexture(kInternalPrevLighting)!);
        ctx.clearTexture(renderData.getTexture(kInternalPrevMoments)!);
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const albedo = renderData.getTexture("Albedo")!;
        const color = renderData.getTexture("Color")!;
        const emission = renderData.getTexture("Emission")!;
        const posNormalFwidth = renderData.getTexture("PositionNormalFwidth")!;
        const linearZ = renderData.getTexture("LinearZ")!;
        const worldNormal = renderData.getTexture("WorldNormal")!;
        const motionVec = renderData.getTexture("MotionVec")!;
        const output = renderData.getTexture("Filtered image")!;

        if (!this.packLinearZAndNormal) {
            this.packLinearZAndNormal = FullScreenPass.create(this.device, { path: kPackLinearZAndNormalShader });
            this.reprojection = FullScreenPass.create(this.device, { path: kReprojectShader });
            this.atrous = FullScreenPass.create(this.device, { path: kAtrousShader });
            this.filterMoments = FullScreenPass.create(this.device, { path: kFilterMomentShader });
            this.finalModulate = FullScreenPass.create(this.device, { path: kFinalModulateShader });
        }

        if (this.buffersNeedClear) {
            this.clearBuffers(ctx, renderData);
            this.buffersNeedClear = false;
        }

        if (!this.filterEnabled) {
            ctx.blit(color, output);
            return;
        }

        // Pack linear z + derivative + octahedral normal.
        {
            const cb = this.packLinearZAndNormal.getRootVar()["PerImageCB"] as ShaderVar;
            cb["gLinearZ"] = linearZ;
            cb["gNormal"] = worldNormal;
            this.packLinearZAndNormal.execute(ctx, this.linearZAndNormalFbo!.fbo);
        }

        // Demodulate + temporal reprojection (3 MRTs: illumination, moments, history).
        {
            const cb = this.reprojection!.getRootVar()["PerImageCB"] as ShaderVar;
            cb["gMotion"] = motionVec;
            cb["gColor"] = color;
            cb["gEmission"] = emission;
            cb["gAlbedo"] = albedo;
            cb["gPositionNormalFwidth"] = posNormalFwidth;
            cb["gPrevIllum"] = this.filteredPastFbo!.textures[0]!;
            cb["gPrevMoments"] = this.prevReprojFbo!.textures[1]!;
            cb["gLinearZAndNormal"] = this.linearZAndNormalFbo!.textures[0]!;
            cb["gPrevLinearZAndNormal"] = renderData.getTexture(kInternalPrevLinearZAndNormal)!;
            cb["gPrevHistoryLength"] = this.prevReprojFbo!.textures[2]!;
            cb["gAlpha"] = this.alpha;
            cb["gMomentsAlpha"] = this.momentsAlpha;
            this.reprojection!.execute(ctx, this.curReprojFbo!.fbo);
        }

        // First cross-bilateral filter + variance estimate.
        {
            const cb = this.filterMoments!.getRootVar()["PerImageCB"] as ShaderVar;
            cb["gIllumination"] = this.curReprojFbo!.textures[0]!;
            cb["gHistoryLength"] = this.curReprojFbo!.textures[2]!;
            cb["gLinearZAndNormal"] = this.linearZAndNormalFbo!.textures[0]!;
            cb["gMoments"] = this.curReprojFbo!.textures[1]!;
            cb["gPhiColor"] = this.phiColor;
            cb["gPhiNormal"] = this.phiNormal;
            this.filterMoments!.execute(ctx, this.pingPongFbo![0].fbo);
        }

        // A-trous wavelet decomposition.
        {
            const cb = this.atrous!.getRootVar()["PerImageCB"] as ShaderVar;
            cb["gAlbedo"] = albedo;
            cb["gHistoryLength"] = this.curReprojFbo!.textures[2]!;
            cb["gLinearZAndNormal"] = this.linearZAndNormalFbo!.textures[0]!;
            cb["gPhiColor"] = this.phiColor;
            cb["gPhiNormal"] = this.phiNormal;
            for (let i = 0; i < this.filterIterations; i++) {
                const curTarget = this.pingPongFbo![1];
                cb["gIllumination"] = this.pingPongFbo![0].textures[0]!;
                cb["gStepSize"] = 1 << i;
                this.atrous!.execute(ctx, curTarget.fbo);
                if (i === Math.min(this.feedbackTap, this.filterIterations - 1)) {
                    ctx.blit(curTarget.textures[0]!, this.filteredPastFbo!.textures[0]!);
                }
                [this.pingPongFbo![0], this.pingPongFbo![1]] = [this.pingPongFbo![1], this.pingPongFbo![0]];
            }
            if (this.feedbackTap < 0) {
                ctx.blit(this.curReprojFbo!.textures[0]!, this.filteredPastFbo!.textures[0]!);
            }
        }

        // Modulate filtered illumination with albedo and re-add emission.
        {
            const cb = this.finalModulate!.getRootVar()["PerImageCB"] as ShaderVar;
            cb["gAlbedo"] = albedo;
            cb["gEmission"] = emission;
            cb["gIllumination"] = this.pingPongFbo![0].textures[0]!;
            this.finalModulate!.execute(ctx, this.finalFbo!.fbo);
        }

        ctx.blit(this.finalFbo!.textures[0]!, output);

        // Swap resources for the next frame.
        [this.curReprojFbo, this.prevReprojFbo] = [this.prevReprojFbo, this.curReprojFbo];
        ctx.blit(this.linearZAndNormalFbo!.textures[0]!, renderData.getTexture(kInternalPrevLinearZAndNormal)!);
    }
}

registerRenderPass("SVGFPass", (device, props) => new SVGFPass(device, props));
