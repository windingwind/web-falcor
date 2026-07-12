/**
 * Invalid pixel detection pass mirroring
 * Source/RenderPasses/DebugPasses/InvalidPixelDetectionPass: NaN pixels
 * render red, Inf green, valid black (unmodified upstream shader).
 */

import {
    Fbo,
    FullScreenPass,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type Properties,
    type RenderContext,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/DebugPasses/InvalidPixelDetectionPass/InvalidPixelDetection.ps.slang";

export class InvalidPixelDetectionPass extends RenderPass {
    private pass: FullScreenPass | null = null;
    private fbo = new Fbo();

    constructor(device: Device, _props: Properties) {
        super(device);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("src", "Input image to be checked").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Output where pixels are red if NaN, green if Inf, and black otherwise")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const src = renderData.getTexture("src")!;
        const dst = renderData.getTexture("dst")!;
        this.pass ??= FullScreenPass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        root["gTexture"] = src;
        this.fbo.attachColorTarget(dst, 0);
        this.pass.execute(ctx, this.fbo);
    }
}

registerRenderPass("InvalidPixelDetectionPass", (device, props) => new InvalidPixelDetectionPass(device, props));
