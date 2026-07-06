/**
 * Temporal AA mirroring Source/RenderPasses/TAA: neighborhood-clamped
 * exponential history accumulation with Catmull-Rom history sampling.
 * Kernel unmodified except the bool->uint cbuffer override.
 */

import {
    Fbo,
    FullScreenPass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceType,
    Texture,
    TextureFilteringMode,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type Sampler,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/TAA/TAA.ps.slang";

export class TAA extends RenderPass {
    private pass: FullScreenPass | null = null;
    private fbo = new Fbo();
    private prevColor: Texture | null = null;
    private linearSampler: Sampler;
    private alpha = 0.1;
    private colorBoxSigma = 1.0;
    private antiFlicker = true;

    constructor(device: Device, props: Properties) {
        super(device);
        this.alpha = props.get("alpha", 0.1);
        this.colorBoxSigma = props.get("colorBoxSigma", 1.0);
        this.antiFlicker = props.get("antiFlicker", true);
        this.linearSampler = device.createSampler({
            minFilter: TextureFilteringMode.Linear,
            magFilter: TextureFilteringMode.Linear,
            mipFilter: TextureFilteringMode.Linear,
        });
    }

    override getProperties(): Properties {
        return new Properties({ alpha: this.alpha, colorBoxSigma: this.colorBoxSigma, antiFlicker: this.antiFlicker });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("motionVecs", "Screen-space motion vectors");
        r.addInput("colorIn", "Color-buffer of the current frame");
        r.addOutput("colorOut", "Anti-aliased color buffer");
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const colorIn = renderData.getTexture("colorIn")!;
        const colorOut = renderData.getTexture("colorOut")!;
        const motionVec = renderData.getTexture("motionVecs")!;
        this.allocatePrevColor(colorOut);
        this.fbo.attachColorTarget(colorOut, 0);

        if (!this.pass) this.pass = FullScreenPass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        const cb = root["PerFrameCB"]!;
        cb["gAlpha"] = this.alpha;
        cb["gColorBoxSigma"] = this.colorBoxSigma;
        cb["gAntiFlicker"] = this.antiFlicker ? 1 : 0;
        root["gTexColor"] = colorIn;
        root["gTexMotionVec"] = motionVec;
        root["gTexPrevColor"] = this.prevColor!;
        root["gSampler"] = this.linearSampler;

        this.pass.execute(ctx, this.fbo);
        ctx.blit(colorOut, this.prevColor!);
    }

    /** Mirrors TAA::allocatePrevColor (history matches colorOut shape/format). */
    private allocatePrevColor(colorOut: Texture): void {
        const stale =
            !this.prevColor ||
            this.prevColor.width !== colorOut.width ||
            this.prevColor.height !== colorOut.height ||
            this.prevColor.format !== colorOut.format;
        if (stale) {
            this.prevColor = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: colorOut.width,
                height: colorOut.height,
                format: colorOut.format,
                bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
                name: "TAA::prevColor",
            });
        }
    }
}

registerRenderPass("TAA", (device, props) => new TAA(device, props));
