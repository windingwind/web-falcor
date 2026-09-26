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
    type Scene,
    type UIWidgets,
    Logger,
    RuntimeError,
} from "@web-falcor/falcor";

import { calculateWhiteBalanceTransformRGB_Rec709, invertMat3, mulMat3Vec, type Mat3 } from "./ColorUtils.js";

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

/** Mirrors ToneMapper::ExposureMode. */
export enum ExposureMode {
    AperturePriority = 0,
    ShutterPriority = 1,
}
// Native clamps (ToneMapper.cpp).
const kExposureCompensationMin = -12, kExposureCompensationMax = 12;
const kFilmSpeedMin = 1, kFilmSpeedMax = 6400;
const kFNumberMin = 0.1, kFNumberMax = 100;
const kShutterMin = 0.1, kShutterMax = 10000;
const kExposureValueMin = Math.log2(kShutterMin * kFNumberMin * kFNumberMin), kExposureValueMax = Math.log2(kShutterMax * kFNumberMax * kFNumberMax);
const kWhitePointMin = 1905, kWhitePointMax = 25000;
const kIdentity3: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export class ToneMapper extends RenderPass {
    private mOperator = ToneMapOperator.Aces;
    private mExposureCompensation = 0;
    private mAutoExposure = false;
    private mFNumber = 1;
    private mShutter = 1;
    private mFilmSpeed = 100;
    private mExposureMode = ExposureMode.AperturePriority;
    private mExposureValue = 0; // log2(shutter * fNumber^2), kept in sync like native updateExposureValue
    private mWhiteBalance = false;
    private mWhitePoint = 6500;
    private whiteBalanceTransform: Mat3 = kIdentity3;
    private sourceWhite: [number, number, number] = [1, 1, 1];
    private mWhiteMaxLuminance = 1.0;
    private mWhiteScale = 11.2;
    private mClamp = true;
    private outputFormat = ResourceFormat.RGBA8UnormSrgb;
    /** Set by the 'outputFormat' property (native's Unknown default is omitted from getProperties). */
    private outputFormatSet = false;
    /** ToneMapper::mUseSceneMetadata: take film speed, f-number and shutter from the scene. */
    private useSceneMetadata = true;
    private outputSize = IOSize.Default;
    /** Native kFixedOutputSize default (used when outputSize == Fixed). */
    private fixedOutputSize: [number, number] = [512, 512];
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
        const fixed = props.getOpt<number[] | { x: number; y: number }>("fixedOutputSize");
        if (fixed) this.fixedOutputSize = Array.isArray(fixed) ? [fixed[0]!, fixed[1]!] : [fixed.x, fixed.y];
        const op = props.getOpt<string | number>("operator");
        if (op !== undefined) {
            this.mOperator = (typeof op === "string" ? ToneMapOperator[op as keyof typeof ToneMapOperator] : op) ?? ToneMapOperator.Aces;
            this.pass = null;
        }
        this.mExposureCompensation = Math.min(kExposureCompensationMax, Math.max(kExposureCompensationMin, props.get("exposureCompensation", 0)));
        const mode = props.getOpt<string | number>("exposureMode");
        if (mode !== undefined) this.mExposureMode = (typeof mode === "string" ? ExposureMode[mode as keyof typeof ExposureMode] : mode) ?? this.mExposureMode;
        this.setFNumber(props.get("fNumber", this.mFNumber));
        this.setShutter(props.get("shutter", this.mShutter));
        this.mFilmSpeed = Math.min(kFilmSpeedMax, Math.max(kFilmSpeedMin, props.get("filmSpeed", this.mFilmSpeed)));
        // Native parseProperties has no case for 'exposureValue' (only the python property setter applies it);
        // the upstream image-test graphs pass it and the oracles match with it ignored.
        if (props.has("exposureValue")) Logger.warning("Unknown property 'exposureValue' in a ToneMapping properties.");
        this.mWhiteBalance = props.get("whiteBalance", this.mWhiteBalance);
        this.mWhitePoint = Math.min(kWhitePointMax, Math.max(kWhitePointMin, props.get("whitePoint", this.mWhitePoint)));
        this.mWhiteMaxLuminance = props.get("whiteMaxLuminance", this.mWhiteMaxLuminance);
        this.mWhiteScale = props.get("whiteScale", this.mWhiteScale);
        this.useSceneMetadata = props.get("useSceneMetadata", this.useSceneMetadata);
        this.updateWhiteBalanceTransform();
        const autoExposure = props.get("autoExposure", this.mAutoExposure);
        if (autoExposure !== this.mAutoExposure) {
            this.mAutoExposure = autoExposure;
            this.pass = null; // shader define — rebuild next execute
        }
        this.mClamp = props.get("clamp", true);
        const fmt = props.getOpt<string | number>("outputFormat");
        if (fmt !== undefined) {
            this.outputFormat = (typeof fmt === "string" ? ResourceFormat[fmt as keyof typeof ResourceFormat] : fmt) ?? this.outputFormat;
            this.outputFormatSet = true;
        }
    }

    override getProperties(): Properties {
        // Native's key order.
        const props: NonNullable<ConstructorParameters<typeof Properties>[0]> = { outputSize: IOSize[this.outputSize]! };
        if (this.outputSize === IOSize.Fixed) props.fixedOutputSize = this.fixedOutputSize;
        if (this.outputFormatSet) props.outputFormat = ResourceFormat[this.outputFormat]!;
        Object.assign(props, {
            useSceneMetadata: this.useSceneMetadata,
            exposureCompensation: this.mExposureCompensation,
            autoExposure: this.mAutoExposure,
            filmSpeed: this.mFilmSpeed,
            whiteBalance: this.mWhiteBalance,
            whitePoint: this.mWhitePoint,
            operator: ToneMapOperator[this.mOperator]!,
            clamp: this.mClamp,
            whiteMaxLuminance: this.mWhiteMaxLuminance,
            whiteScale: this.mWhiteScale,
            fNumber: this.mFNumber,
            shutter: this.mShutter,
            exposureMode: ExposureMode[this.mExposureMode]!,
        });
        return new Properties(props);
    }

    /** Python properties (ToneMapper's pybind11 binding over its setters; operator/exposureMode by enum name). */
    get exposureCompensation(): number {
        return this.mExposureCompensation;
    }
    set exposureCompensation(v: number) {
        this.mExposureCompensation = Math.min(kExposureCompensationMax, Math.max(kExposureCompensationMin, v));
    }
    get autoExposure(): boolean {
        return this.mAutoExposure;
    }
    set autoExposure(v: boolean) {
        this.mAutoExposure = v;
        this.pass = null; // shader define — rebuild next execute
    }
    get exposureValue(): number {
        return this.mExposureValue;
    }
    set exposureValue(v: number) {
        this.setExposureValue(v);
    }
    get filmSpeed(): number {
        return this.mFilmSpeed;
    }
    set filmSpeed(v: number) {
        this.mFilmSpeed = Math.min(kFilmSpeedMax, Math.max(kFilmSpeedMin, v));
    }
    get whiteBalance(): boolean {
        return this.mWhiteBalance;
    }
    set whiteBalance(v: boolean) {
        this.mWhiteBalance = v;
        this.updateWhiteBalanceTransform();
    }
    get whitePoint(): number {
        return this.mWhitePoint;
    }
    set whitePoint(v: number) {
        this.setWhitePoint(v);
    }
    get operator(): string {
        return ToneMapOperator[this.mOperator]!;
    }
    set operator(name: string) {
        const op = ToneMapOperator[name as keyof typeof ToneMapOperator];
        if (op === undefined) throw new RuntimeError(`Invalid enum name '${name}'`);
        if (op !== this.mOperator) {
            this.mOperator = op;
            this.pass = null;
        }
    }
    get clamp(): boolean {
        return this.mClamp;
    }
    set clamp(v: boolean) {
        if (v !== this.mClamp) {
            this.mClamp = v;
            this.pass = null;
        }
    }
    get whiteMaxLuminance(): number {
        return this.mWhiteMaxLuminance;
    }
    set whiteMaxLuminance(v: number) {
        this.mWhiteMaxLuminance = v;
    }
    get whiteScale(): number {
        return this.mWhiteScale;
    }
    set whiteScale(v: number) {
        this.mWhiteScale = Math.max(0.001, v);
    }
    get fNumber(): number {
        return this.mFNumber;
    }
    set fNumber(v: number) {
        this.setFNumber(v);
    }
    get shutter(): number {
        return this.mShutter;
    }
    set shutter(v: number) {
        this.setShutter(v);
    }
    get exposureMode(): string {
        return ExposureMode[this.mExposureMode]!;
    }
    set exposureMode(name: string) {
        const mode = ExposureMode[name as keyof typeof ExposureMode];
        if (mode === undefined) throw new RuntimeError(`Invalid enum name '${name}'`);
        this.mExposureMode = mode;
    }

    /** ToneMapper::setScene: the scene's camera metadata, unless disabled. */
    override setScene(scene: Scene | null): void {
        super.setScene(scene);
        const meta = scene?.metadata;
        if (!meta || !this.useSceneMetadata) return;
        if (meta.filmISO !== undefined) this.mFilmSpeed = Math.min(kFilmSpeedMax, Math.max(kFilmSpeedMin, meta.filmISO));
        if (meta.fNumber !== undefined) this.setFNumber(meta.fNumber);
        if (meta.shutterSpeed !== undefined) this.setShutter(meta.shutterSpeed);
    }

    /** Mirrors ToneMapper::setFNumber / setShutter / setExposureValue / updateExposureValue. */
    setFNumber(fNumber: number): void {
        this.mFNumber = Math.min(kFNumberMax, Math.max(kFNumberMin, fNumber));
        this.mExposureValue = Math.log2(this.mShutter * this.mFNumber * this.mFNumber);
    }
    setShutter(shutter: number): void {
        this.mShutter = Math.min(kShutterMax, Math.max(kShutterMin, shutter));
        this.mExposureValue = Math.log2(this.mShutter * this.mFNumber * this.mFNumber);
    }
    setExposureValue(ev: number): void {
        this.mExposureValue = Math.min(kExposureValueMax, Math.max(kExposureValueMin, ev));
        if (this.mExposureMode === ExposureMode.AperturePriority) {
            this.mShutter = Math.min(kShutterMax, Math.max(kShutterMin, Math.pow(2, this.mExposureValue) / (this.mFNumber * this.mFNumber)));
        } else {
            this.mFNumber = Math.min(kFNumberMax, Math.max(kFNumberMin, Math.sqrt(Math.pow(2, this.mExposureValue) / this.mShutter)));
        }
    }
    getExposureValue(): number {
        return this.mExposureValue;
    }
    setWhitePoint(kelvin: number): void {
        this.mWhitePoint = Math.min(kWhitePointMax, Math.max(kWhitePointMin, kelvin));
        this.updateWhiteBalanceTransform();
    }
    /** Mirrors updateWhiteBalanceTransform (also derives the source white shown in the UI). */
    private updateWhiteBalanceTransform(): void {
        this.whiteBalanceTransform = this.mWhiteBalance ? calculateWhiteBalanceTransformRGB_Rec709(this.mWhitePoint) : kIdentity3;
        this.sourceWhite = mulMat3Vec(invertMat3(this.whiteBalanceTransform), [1, 1, 1]);
    }

    override renderUI(ui: UIWidgets): void {
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
        const exposure = ui.group("Exposure");
        exposure.slider("Exposure Compensation", this.mExposureCompensation, kExposureCompensationMin, kExposureCompensationMax, 0.1, (v) => (this.mExposureCompensation = v));
        exposure.checkbox("Auto Exposure", this.mAutoExposure, (v) => {
            this.mAutoExposure = v;
            this.pass = null; // shader define — rebuild next execute
        });
        exposure.dropdown("Exposure mode", ["AperturePriority", "ShutterPriority"], ExposureMode[this.mExposureMode]!, (v) => (this.mExposureMode = ExposureMode[v as keyof typeof ExposureMode]));
        exposure.slider("Exposure Value (EV)", this.mExposureValue, kExposureValueMin, kExposureValueMax, 0.1, (v) => this.setExposureValue(v));
        exposure.slider("Film Speed (ISO)", this.mFilmSpeed, kFilmSpeedMin, kFilmSpeedMax, 0.1, (v) => (this.mFilmSpeed = v));
        exposure.slider("f-Number", this.mFNumber, kFNumberMin, kFNumberMax, 0.1, (v) => this.setFNumber(v));
        exposure.slider("Shutter", this.mShutter, kShutterMin, kShutterMax, 0.1, (v) => this.setShutter(v));
        const grading = ui.group("Color Grading");
        grading.checkbox("White Balance", this.mWhiteBalance, (v) => {
            this.mWhiteBalance = v;
            this.updateWhiteBalanceTransform();
        });
        grading.slider("White Point (K)", this.mWhitePoint, kWhitePointMin, kWhitePointMax, 5, (v) => this.setWhitePoint(v));
        const w = this.sourceWhite;
        const wMax = Math.max(w[0], w[1], w[2]);
        grading.text(`Source white (normalized): ${(w[0] / wMax).toFixed(3)}, ${(w[1] / wMax).toFixed(3)}, ${(w[2] / wMax).toFixed(3)}`);
        const tone = ui.group("Tonemapping");
        const ops = Object.keys(ToneMapOperator).filter((k) => isNaN(Number(k)));
        tone.dropdown("Operator", ops, ToneMapOperator[this.mOperator]!, (v) => {
            this.mOperator = ToneMapOperator[v as keyof typeof ToneMapOperator];
            this.pass = null; // operator is a shader define — rebuild next execute
        });
        tone.slider("White Luminance", this.mWhiteMaxLuminance, 0.1, 100, 0.2, (v) => (this.mWhiteMaxLuminance = v)); // ReinhardModified
        tone.slider("Linear White", this.mWhiteScale, 0, 100, 0.01, (v) => (this.mWhiteScale = v)); // HableUc2
        tone.checkbox("Clamp Output", this.mClamp, (v) => {
            this.mClamp = v;
            this.pass = null; // clamp is a shader define — rebuild next execute
        });
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, this.fixedOutputSize, compileData.defaultTexDims);
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
            const defines: Record<string, string | number> = { _TONE_MAPPER_OPERATOR: this.mOperator };
            if (this.mClamp) defines["_TONE_MAPPER_CLAMP"] = 1;
            if (this.mAutoExposure) defines["_TONE_MAPPER_AUTO_EXPOSURE"] = 1;
            this.pass = FullScreenPass.create(this.device, { path: kShaderFile, defines });
        }

        if (this.mAutoExposure) this.runLuminancePass(ctx, src);

        // Exposure folds into the color transform (mirrors updateColorTransform:
        // white balance * 2^EC * manual physical exposure when auto is off).
        const manualExposureScale = this.mAutoExposure ? 1 : this.mFilmSpeed / 100 / (this.mShutter * this.mFNumber * this.mFNumber);
        const scale = Math.pow(2, this.mExposureCompensation) * manualExposureScale;
        // float3x4 row-major rows: whiteBalanceTransform * exposureScale * manualExposureScale (updateColorTransform).
        const m = this.whiteBalanceTransform;
        const colorTransform = [m[0] * scale, m[1] * scale, m[2] * scale, 0, m[3] * scale, m[4] * scale, m[5] * scale, 0, m[6] * scale, m[7] * scale, m[8] * scale, 0];

        const root = this.pass.getRootVar();
        root["gColorTex"] = src;
        root["gColorSampler"] = this.device.createSampler();
        root["PerImageCB"]["gParams"]["whiteScale"] = this.mWhiteScale;
        root["PerImageCB"]["gParams"]["whiteMaxLuminance"] = this.mWhiteMaxLuminance;
        root["PerImageCB"]["gParams"]["colorTransform"] = colorTransform;
        if (this.mAutoExposure) {
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
