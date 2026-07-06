/**
 * CrossFade pass mirroring Source/RenderPasses/Utils/CrossFade: fades between
 * inputs A and B, either auto (frame-counted after reset) or by a fixed
 * factor. Uses the WebFalcor override shader (write-only output texture).
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
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/Utils/CrossFade/CrossFade.cs.slang";

export class CrossFade extends RenderPass {
    private outputFormat = ResourceFormat.RGBA32Float;
    private enableAutoFade = true;
    private waitFrameCount = 10;
    private fadeFrameCount = 100;
    private fadeFactor = 0.5;
    private mixFrame = 0;
    private frameDim: [number, number] = [0, 0];
    private pass: ComputePass | null = null;
    private dummy: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        const fmt = props.getOpt<string | number>("outputFormat");
        if (fmt !== undefined) this.outputFormat = (typeof fmt === "string" ? ResourceFormat[fmt as keyof typeof ResourceFormat] : fmt) ?? this.outputFormat;
        this.enableAutoFade = props.get("enableAutoFade", true);
        this.waitFrameCount = props.get("waitFrameCount", 10);
        this.fadeFrameCount = props.get("fadeFrameCount", 100);
        this.fadeFactor = props.get("fadeFactor", 0.5);
    }

    override getProperties(): Properties {
        return new Properties({
            enableAutoFade: this.enableAutoFade,
            waitFrameCount: this.waitFrameCount,
            fadeFrameCount: this.fadeFrameCount,
            fadeFactor: this.fadeFactor,
        });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("A", "Input A").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("B", "Input B").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("out", "Output").bindFlags(ResourceBindFlags.UnorderedAccess).format(this.outputFormat);
        return r;
    }

    override compile(_ctx: RenderContext, compileData: CompileData): void {
        this.frameDim = compileData.defaultTexDims;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.mixFrame = 0;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        // Native resets on refresh flags / scene updates; on the web only scene
        // (re)binds reset (no Mogwai UI mutation channels yet).
        this.mixFrame++;
        const mix = this.enableAutoFade
            ? Math.min(Math.max((this.mixFrame - this.waitFrameCount) / this.fadeFrameCount, 0), 1)
            : Math.min(Math.max(this.fadeFactor, 0), 1);
        const scaleA = 1 - mix;
        const scaleB = mix;

        const output = renderData.getTexture("out")!;
        this.outputFormat = output.format;

        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        this.dummy ??= new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 1,
            height: 1,
            format: ResourceFormat.RGBA32Float,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "CrossFade::dummy",
        });

        const root = this.pass.getRootVar();
        root["CB"]["frameDim"] = this.frameDim;
        root["CB"]["scaleA"] = scaleA;
        root["CB"]["scaleB"] = scaleB;
        root["A"] = renderData.getTexture("A") ?? this.dummy;
        root["B"] = renderData.getTexture("B") ?? this.dummy;
        root["output"] = output;
        this.pass.execute(ctx, this.frameDim[0], this.frameDim[1]);
    }
}

registerRenderPass("CrossFade", (device, props) => new CrossFade(device, props));
