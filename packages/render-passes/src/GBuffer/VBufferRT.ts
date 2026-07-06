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
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    SAMPLE_GENERATOR_DEFAULT,
    SampleGenerator,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang";

export class VBufferRT extends RenderPass {
    private pass: ComputePass | null = null;
    private frameCount = 0;
    private useAlphaTest = false;
    private sampleGenerator: SampleGenerator;
    private cameraJitterGenerator: CPUSampleGenerator | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.useAlphaTest = props.get("useAlphaTest", false);
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_DEFAULT);
        // Mirrors GBufferBase::updateSamplePattern (Center -> no generator).
        const pattern = props.get<string>("samplePattern", "Center");
        const count = props.get("sampleCount", 16);
        if (pattern === "Stratified") this.cameraJitterGenerator = new StratifiedSamplePattern(count);
        else if (pattern === "Halton") this.cameraJitterGenerator = new HaltonSamplePattern(count);
        else if (pattern === "DirectX") this.cameraJitterGenerator = new DxSamplePattern(count);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
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
        this.pass = null;
        this.frameCount = 0;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const vbuffer = renderData.getTexture("vbuffer")!;
        // Mirrors GBufferBase::updateFrameDim (first jitter sample lands next frame).
        this.scene.camera.setPatternGenerator(
            this.cameraJitterGenerator,
            new float2(Math.fround(1 / vbuffer.width), Math.fround(1 / vbuffer.height)),
        );
        if (!this.pass) {
            const defines = this.scene.getSceneDefines().addAll({
                USE_ALPHA_TEST: this.useAlphaTest ? 1 : 0,
                RAY_FLAGS: 0,
                COMPUTE_DEPTH_OF_FIELD: 0,
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
