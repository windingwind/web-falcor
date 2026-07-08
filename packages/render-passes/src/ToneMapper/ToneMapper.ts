/**
 * Tone mapping pass mirroring Source/RenderPasses/ToneMapper, using the
 * unmodified upstream ToneMapping.ps.slang. Auto-exposure (luminance pass)
 * lands with the camera/exposure work in M5; manual exposure is complete.
 */

import {
    IOSize,
    parseIOSize,
    calculateIOSize,
    Fbo,
    FullScreenPass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/ToneMapper/ToneMapping.ps.slang";

export enum ToneMapOperator {
    Linear = 0,
    Reinhard = 1,
    ReinhardModified = 2,
    HejiHableAlu = 3,
    HableUc2 = 4,
    Aces = 5,
}

export class ToneMapper extends RenderPass {
    private operator = ToneMapOperator.Aces;
    private exposureCompensation = 0;
    private clamp = true;
    private outputFormat = ResourceFormat.RGBA8UnormSrgb;
    private outputSize = IOSize.Default;
    private pass: FullScreenPass | null = null;
    private fbo = new Fbo();

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        this.outputSize = parseIOSize(props.getOpt("outputSize"), this.outputSize);
        const op = props.getOpt<string | number>("operator");
        if (op !== undefined) {
            this.operator = (typeof op === "string" ? ToneMapOperator[op as keyof typeof ToneMapOperator] : op) ?? ToneMapOperator.Aces;
            this.pass = null;
        }
        this.exposureCompensation = props.get("exposureCompensation", 0);
        this.clamp = props.get("clamp", true);
        const fmt = props.getOpt<string | number>("outputFormat");
        if (fmt !== undefined) {
            this.outputFormat = (typeof fmt === "string" ? ResourceFormat[fmt as keyof typeof ResourceFormat] : fmt) ?? this.outputFormat;
        }
    }

    override getProperties(): Properties {
        return new Properties({
            operator: ToneMapOperator[this.operator]!,
            exposureCompensation: this.exposureCompensation,
            clamp: this.clamp,
        });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Exposure Compensation", this.exposureCompensation, -8, 8, 0.1, (v) => (this.exposureCompensation = v));
        const ops = Object.keys(ToneMapOperator).filter((k) => isNaN(Number(k)));
        ui.dropdown("Operator", ops, ToneMapOperator[this.operator]!, (v) => {
            this.operator = ToneMapOperator[v as keyof typeof ToneMapOperator];
            this.pass = null; // operator is a shader define — rebuild next execute
        });
        ui.checkbox("Clamp Output", this.clamp, (v) => {
            this.clamp = v;
            this.pass = null; // clamp is a shader define — rebuild next execute
        });
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, [512, 512], compileData.defaultTexDims);
        r.addInput("src", "Source texture").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Tone-mapped output")
            .texture2D(w, h)
            .format(this.outputFormat)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const src = renderData.getTexture("src")!;
        const dst = renderData.getTexture("dst")!;

        if (!this.pass) {
            const defines: Record<string, string | number> = { _TONE_MAPPER_OPERATOR: this.operator };
            if (this.clamp) defines["_TONE_MAPPER_CLAMP"] = 1;
            this.pass = FullScreenPass.create(this.device, { path: kShaderFile, defines });
        }

        // Exposure compensation folds into the color transform (mirrors upstream
        // ToneMapper::updateColorTransform: exposure scale * white balance).
        const scale = Math.pow(2, this.exposureCompensation);
        // float3x4 row-major rows.
        const colorTransform = [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0];

        const root = this.pass.getRootVar();
        root["gColorTex"] = src;
        root["gColorSampler"] = this.device.createSampler();
        root["PerImageCB"]["gParams"]["whiteScale"] = 11.2;
        root["PerImageCB"]["gParams"]["whiteMaxLuminance"] = 1.0;
        root["PerImageCB"]["gParams"]["colorTransform"] = colorTransform;

        this.fbo.attachColorTarget(dst, 0);
        this.pass.execute(ctx, this.fbo);
    }
}

registerRenderPass("ToneMapper", (device, props) => new ToneMapper(device, props));
