/**
 * Tone mapping pass mirroring Source/RenderPasses/ToneMapper, using the
 * unmodified upstream ToneMapping.ps.slang and Luminance.ps.slang
 * (auto-exposure via log-luminance mip chain).
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
    ResourceType,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/ToneMapper/ToneMapping.ps.slang";
const kLuminanceFile = "RenderPasses/ToneMapper/Luminance.ps.slang";

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
    private autoExposure = false;
    private fNumber = 1;
    private shutter = 1;
    private filmSpeed = 100;
    private clamp = true;
    private outputFormat = ResourceFormat.RGBA8UnormSrgb;
    private outputSize = IOSize.Default;
    private pass: FullScreenPass | null = null;
    private fbo = new Fbo();
    private luminancePass: FullScreenPass | null = null;
    private luminanceTex: Texture | null = null;
    private luminanceFbo = new Fbo();

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
        this.fNumber = props.get("fNumber", this.fNumber);
        this.shutter = props.get("shutter", this.shutter);
        this.filmSpeed = props.get("filmSpeed", this.filmSpeed);
        const autoExposure = props.get("autoExposure", this.autoExposure);
        if (autoExposure !== this.autoExposure) {
            this.autoExposure = autoExposure;
            this.pass = null; // shader define — rebuild next execute
        }
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
            autoExposure: this.autoExposure,
            fNumber: this.fNumber,
            shutter: this.shutter,
            filmSpeed: this.filmSpeed,
            clamp: this.clamp,
        });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Exposure Compensation", this.exposureCompensation, -8, 8, 0.1, (v) => (this.exposureCompensation = v));
        ui.checkbox("Auto Exposure", this.autoExposure, (v) => {
            this.autoExposure = v;
            this.pass = null; // shader define — rebuild next execute
        });
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
            if (this.autoExposure) defines["_TONE_MAPPER_AUTO_EXPOSURE"] = 1;
            this.pass = FullScreenPass.create(this.device, { path: kShaderFile, defines });
        }

        if (this.autoExposure) this.runLuminancePass(ctx, src);

        // Exposure folds into the color transform (mirrors updateColorTransform:
        // white balance * 2^EC * manual physical exposure when auto is off).
        const manualExposureScale = this.autoExposure ? 1 : this.filmSpeed / 100 / (this.shutter * this.fNumber * this.fNumber);
        const scale = Math.pow(2, this.exposureCompensation) * manualExposureScale;
        // float3x4 row-major rows.
        const colorTransform = [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0];

        const root = this.pass.getRootVar();
        root["gColorTex"] = src;
        root["gColorSampler"] = this.device.createSampler();
        root["PerImageCB"]["gParams"]["whiteScale"] = 11.2;
        root["PerImageCB"]["gParams"]["whiteMaxLuminance"] = 1.0;
        root["PerImageCB"]["gParams"]["colorTransform"] = colorTransform;
        if (this.autoExposure) {
            root["gLuminanceTexSampler"] = this.device.createSampler();
            root["gLuminanceTex"] = this.luminanceTex!;
        }

        this.fbo.attachColorTarget(dst, 0);
        this.pass.execute(ctx, this.fbo);
    }

    /** Mirrors ToneMapper::createLuminanceFbo + luminance pass + generateMips:
     *  log2-luminance into a pow2 texture, mip chain average, read at LOD 16. */
    private runLuminancePass(ctx: RenderContext, src: Texture): void {
        const w = 1 << Math.floor(Math.log2(src.width));
        const h = 1 << Math.floor(Math.log2(src.height));
        // Upstream uses R32Float for fp32 sources; linear-filtering r32float needs float32-filterable.
        const format = this.device.hasFeature("float32-filterable") ? ResourceFormat.R32Float : ResourceFormat.R16Float;
        if (!this.luminanceTex || this.luminanceTex.width !== w || this.luminanceTex.height !== h || this.luminanceTex.format !== format) {
            this.luminanceTex = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: w,
                height: h,
                format,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget,
                name: "ToneMapper::luminance",
            });
            this.luminanceFbo.attachColorTarget(this.luminanceTex, 0);
        }

        this.luminancePass ??= FullScreenPass.create(this.device, { path: kLuminanceFile });
        const root = this.luminancePass.getRootVar();
        root["gColorTex"] = src;
        root["gColorSampler"] = this.device.createSampler();
        this.luminancePass.execute(ctx, this.luminanceFbo);
        this.luminanceTex.generateMips(ctx);
    }
}

registerRenderPass("ToneMapper", (device, props) => new ToneMapper(device, props));
