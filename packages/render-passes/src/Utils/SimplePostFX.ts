/**
 * Simple post FX pass mirroring Source/RenderPasses/SimplePostFX: bloom via an
 * 8-level pyramid, star lobes, vignette, chromatic aberration, barrel
 * distortion, color grading. Web divergence (documented in the shader
 * override): upsampling ping-pongs a second pyramid because WGSL forbids
 * rgba16float read_write storage and WebGPU forbids sample+store of one
 * texture in a pass; border-mode sampling is emulated in-shader.
 */

import {
    IOSize,
    parseIOSize,
    calculateIOSize,
    ComputePass,
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
    float3,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/SimplePostFX/SimplePostFX.cs.slang";
const kNumLevels = 8;

function toVec3(v: unknown, fallback: [number, number, number]): [number, number, number] {
    if (Array.isArray(v)) return [v[0] as number, v[1] as number, v[2] as number];
    const o = v as { x?: number; y?: number; z?: number };
    if (o && typeof o.x === "number") return [o.x, o.y!, o.z!];
    return fallback;
}

export class SimplePostFX extends RenderPass {
    private mEnabled = true;
    private mWipe = 0;
    private mBloomAmount = 0;
    private mStarAmount = 0;
    private mStarAngle = 0.1;
    private mVignetteAmount = 0;
    private mChromaticAberrationAmount = 0;
    private mBarrelDistortAmount = 0;
    private mSaturationCurve: [number, number, number] = [1, 1, 1];
    private mColorOffset: [number, number, number] = [0.5, 0.5, 0.5];
    private mColorScale: [number, number, number] = [0.5, 0.5, 0.5];
    private mColorPower: [number, number, number] = [0.5, 0.5, 0.5];
    private mColorOffsetScalar = 0;
    private mColorScaleScalar = 0;
    private mColorPowerScalar = 0;
    private outputSize = IOSize.Default;
    private fixedOutputSize: [number, number] = [512, 512];

    private downsamplePass: ComputePass;
    private upsamplePass: ComputePass;
    private postFXPass: ComputePass;
    /** pyramid[0..kNumLevels]: downsample chain; upPyramid[0..kNumLevels-1]: upsample ping-pong (web divergence). */
    private pyramid: (Texture | null)[] = new Array<Texture | null>(kNumLevels + 1).fill(null);
    private upPyramid: (Texture | null)[] = new Array<Texture | null>(kNumLevels).fill(null);

    constructor(device: Device, props: Properties) {
        super(device);
        this.mEnabled = props.get("enabled", true);
        this.mWipe = props.get("wipe", 0);
        this.mBloomAmount = props.get("bloomAmount", 0);
        this.mStarAmount = props.get("starAmount", 0);
        this.mStarAngle = props.get("starAngle", 0.1);
        this.mVignetteAmount = props.get("vignetteAmount", 0);
        this.mChromaticAberrationAmount = props.get("chromaticAberrationAmount", 0);
        this.mBarrelDistortAmount = props.get("barrelDistortAmount", 0);
        this.mSaturationCurve = toVec3(props.getOpt("saturationCurve"), [1, 1, 1]);
        this.mColorOffset = toVec3(props.getOpt("colorOffset"), [0.5, 0.5, 0.5]);
        this.mColorScale = toVec3(props.getOpt("colorScale"), [0.5, 0.5, 0.5]);
        this.mColorPower = toVec3(props.getOpt("colorPower"), [0.5, 0.5, 0.5]);
        this.mColorOffsetScalar = props.get("colorOffsetScalar", 0);
        this.mColorScaleScalar = props.get("colorScaleScalar", 0);
        this.mColorPowerScalar = props.get("colorPowerScalar", 0);
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        const fixed = props.getOpt<number[]>("fixedOutputSize");
        if (fixed) this.fixedOutputSize = [fixed[0]!, fixed[1]!];

        this.downsamplePass = ComputePass.create(device, { path: kShaderFile, csEntry: "downsample" });
        this.upsamplePass = ComputePass.create(device, { path: kShaderFile, csEntry: "upsample" });
        this.postFXPass = ComputePass.create(device, { path: kShaderFile, csEntry: "runPostFX" });
    }

    /** Mirrors SimplePostFX::renderUI. */
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
        ui.checkbox("Enable post fx", this.mEnabled, (v) => (this.mEnabled = v));
        ui.slider("Wipe", this.mWipe, 0, 1, 0.001, (v) => (this.mWipe = v));
        const lens = ui.group("Lens FX");
        lens.slider("Bloom", this.mBloomAmount, 0, 1, 0.001, (v) => (this.mBloomAmount = v));
        lens.slider("Bloom Star", this.mStarAmount, 0, 1, 0.001, (v) => (this.mStarAmount = v));
        lens.slider("Star Angle", this.mStarAngle, 0, 1, 0.001, (v) => (this.mStarAngle = v));
        lens.slider("Vignette", this.mVignetteAmount, 0, 1, 0.001, (v) => (this.mVignetteAmount = v));
        lens.slider("Chromatic Aberration", this.mChromaticAberrationAmount, 0, 1, 0.001, (v) => (this.mChromaticAberrationAmount = v));
        lens.slider("Barrel Distortion", this.mBarrelDistortAmount, 0, 1, 0.001, (v) => (this.mBarrelDistortAmount = v));
        lens.button("reset this group", () => {
            this.mBloomAmount = 0;
            this.mStarAmount = 0;
            this.mStarAngle = 0.1;
            this.mVignetteAmount = 0;
            this.mChromaticAberrationAmount = 0;
            this.mBarrelDistortAmount = 0;
        });
        const sat = ui.group("Saturation");
        const satNames = ["Shadow Saturation", "Midtone Saturation", "Hilight Saturation"];
        satNames.forEach((label, i) => sat.slider(label, this.mSaturationCurve[i]!, 0, 2, 0.001, (v) => (this.mSaturationCurve[i] = v)));
        sat.button("reset this group", () => (this.mSaturationCurve = [1, 1, 1]));
        const luma = ui.group("Offset/Power/Scale (luma)");
        luma.slider("Luma Offset (Shadows)", this.mColorOffsetScalar, -1, 1, 0.001, (v) => (this.mColorOffsetScalar = v));
        luma.slider("Luma Power (Midtones)", this.mColorPowerScalar, -1, 1, 0.001, (v) => (this.mColorPowerScalar = v));
        luma.slider("Luma Scale (Hilights)", this.mColorScaleScalar, -1, 1, 0.001, (v) => (this.mColorScaleScalar = v));
        luma.button("reset this group", () => {
            this.mColorOffsetScalar = 0;
            this.mColorPowerScalar = 0;
            this.mColorScaleScalar = 0;
        });
        const color = ui.group("Offset/Power/Scale (color)");
        const rgb = (label: string, get: () => [number, number, number]) => {
            ["R", "G", "B"].forEach((c, i) => color.slider(`${label} ${c}`, get()[i]!, 0, 1, 0.001, (v) => (get()[i] = v)));
        };
        rgb("Color Offset (Shadows)", () => this.mColorOffset);
        rgb("Color Power (Midtones)", () => this.mColorPower);
        rgb("Color Scale (Hilights)", () => this.mColorScale);
        color.button("reset this group", () => {
            this.mColorOffset = [0.5, 0.5, 0.5];
            this.mColorPower = [0.5, 0.5, 0.5];
            this.mColorScale = [0.5, 0.5, 0.5];
        });
    }

    /** Python properties (SimplePostFX's plain getters/setters; the curves are float3). */
    get enabled(): boolean {
        return this.mEnabled;
    }
    set enabled(v: boolean) {
        this.mEnabled = v;
    }
    get wipe(): number {
        return this.mWipe;
    }
    set wipe(v: number) {
        this.mWipe = v;
    }
    get bloomAmount(): number {
        return this.mBloomAmount;
    }
    set bloomAmount(v: number) {
        this.mBloomAmount = v;
    }
    get starAmount(): number {
        return this.mStarAmount;
    }
    set starAmount(v: number) {
        this.mStarAmount = v;
    }
    get starAngle(): number {
        return this.mStarAngle;
    }
    set starAngle(v: number) {
        this.mStarAngle = v;
    }
    get vignetteAmount(): number {
        return this.mVignetteAmount;
    }
    set vignetteAmount(v: number) {
        this.mVignetteAmount = v;
    }
    get chromaticAberrationAmount(): number {
        return this.mChromaticAberrationAmount;
    }
    set chromaticAberrationAmount(v: number) {
        this.mChromaticAberrationAmount = v;
    }
    get barrelDistortAmount(): number {
        return this.mBarrelDistortAmount;
    }
    set barrelDistortAmount(v: number) {
        this.mBarrelDistortAmount = v;
    }
    get colorOffsetScalar(): number {
        return this.mColorOffsetScalar;
    }
    set colorOffsetScalar(v: number) {
        this.mColorOffsetScalar = v;
    }
    get colorScaleScalar(): number {
        return this.mColorScaleScalar;
    }
    set colorScaleScalar(v: number) {
        this.mColorScaleScalar = v;
    }
    get colorPowerScalar(): number {
        return this.mColorPowerScalar;
    }
    set colorPowerScalar(v: number) {
        this.mColorPowerScalar = v;
    }
    get saturationCurve(): float3 {
        return new float3(...this.mSaturationCurve);
    }
    set saturationCurve(v: unknown) {
        this.mSaturationCurve = toVec3(v, this.mSaturationCurve);
    }
    get colorOffset(): float3 {
        return new float3(...this.mColorOffset);
    }
    set colorOffset(v: unknown) {
        this.mColorOffset = toVec3(v, this.mColorOffset);
    }
    get colorScale(): float3 {
        return new float3(...this.mColorScale);
    }
    set colorScale(v: unknown) {
        this.mColorScale = toVec3(v, this.mColorScale);
    }
    get colorPower(): float3 {
        return new float3(...this.mColorPower);
    }
    set colorPower(v: unknown) {
        this.mColorPower = toVec3(v, this.mColorPower);
    }

    override getProperties(): Properties {
        return new Properties({
            enabled: this.mEnabled,
            wipe: this.mWipe,
            bloomAmount: this.mBloomAmount,
            starAmount: this.mStarAmount,
            starAngle: this.mStarAngle,
            vignetteAmount: this.mVignetteAmount,
            chromaticAberrationAmount: this.mChromaticAberrationAmount,
            barrelDistortAmount: this.mBarrelDistortAmount,
            saturationCurve: this.mSaturationCurve,
            colorOffset: this.mColorOffset,
            colorScale: this.mColorScale,
            colorPower: this.mColorPower,
            colorOffsetScalar: this.mColorOffsetScalar,
            colorScaleScalar: this.mColorScaleScalar,
            colorPowerScalar: this.mColorPowerScalar,
        });
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, this.fixedOutputSize, compileData.defaultTexDims);
        r.addInput("src", "Source texture").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "post-effected output texture")
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess)
            .format(ResourceFormat.RGBA32Float)
            .texture2D(w, h);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const src = renderData.getTexture("src")!;
        const dst = renderData.getTexture("dst")!;

        if (this.mEnabled && (src.width !== dst.width || src.height !== dst.height)) {
            throw new Error("SimplePostFX I/O sizes don't match.");
        }
        const [width, height] = [src.width, src.height];

        const isDefault =
            this.mBloomAmount === 0 &&
            this.mChromaticAberrationAmount === 0 &&
            this.mBarrelDistortAmount === 0 &&
            this.mSaturationCurve.every((v) => v === 1) &&
            this.mColorOffset.every((v) => v === 0.5) &&
            this.mColorScale.every((v) => v === 0.5) &&
            this.mColorPower.every((v) => v === 0.5) &&
            this.mColorOffsetScalar === 0 &&
            this.mColorScaleScalar === 0 &&
            this.mColorPowerScalar === 0;
        if (!this.mEnabled || this.mWipe >= 1 || isDefault) {
            ctx.blit(src, dst);
            return;
        }

        if (this.mBloomAmount > 0) {
            this.preparePyramids(width, height);
            {
                const root = this.downsamplePass.getRootVar();
                for (let level = 0; level < kNumLevels; level++) {
                    const res = [Math.max(1, width >> (level + 1)), Math.max(1, height >> (level + 1))];
                    const srcTex = level ? this.pyramid[level]! : src;
                    root["PerFrameCB"]["gResolution"] = res;
                    root["PerFrameCB"]["gInvRes"] = [1 / res[0]!, 1 / res[1]!];
                    root["PerFrameCB"]["gSrcRes"] = [srcTex.width, srcTex.height];
                    root["gSrc"] = srcTex;
                    root["gDstMip"] = this.pyramid[level + 1]!;
                    this.downsamplePass.execute(ctx, res[0]!, res[1]!);
                }
            }
            {
                const root = this.upsamplePass.getRootVar();
                root["PerFrameCB"]["gBloomAmount"] = this.mBloomAmount;
                for (let level = kNumLevels - 1; level >= 0; level--) {
                    const res = [Math.max(1, width >> level), Math.max(1, height >> level)];
                    const invres = [1 / res[0]!, 1 / res[1]!];
                    const bloomed = level === kNumLevels - 1 ? this.pyramid[level + 1]! : this.upPyramid[level + 1]!;
                    root["PerFrameCB"]["gResolution"] = res;
                    root["PerFrameCB"]["gInvRes"] = invres;
                    root["PerFrameCB"]["gSrcRes"] = [bloomed.width, bloomed.height];
                    const wantStar = level === 1 || level === 2;
                    root["PerFrameCB"]["gStar"] = wantStar ? this.mStarAmount : 0;
                    if (wantStar) {
                        let ang = this.mStarAngle;
                        root["PerFrameCB"]["gStarDir1"] = [Math.sin(ang) * invres[0]! * 2, Math.cos(ang) * invres[1]! * 2];
                        ang += Math.PI / 3;
                        root["PerFrameCB"]["gStarDir2"] = [Math.sin(ang) * invres[0]! * 2, Math.cos(ang) * invres[1]! * 2];
                        ang += Math.PI / 3;
                        root["PerFrameCB"]["gStarDir3"] = [Math.sin(ang) * invres[0]! * 2, Math.cos(ang) * invres[1]! * 2];
                    }
                    root["PerFrameCB"]["gInPlace"] = level > 0 ? 1 : 0;
                    root["gBloomed"] = bloomed;
                    root["gDstPrev"] = this.pyramid[level]!;
                    root["gSrc"] = src;
                    root["gDstMip"] = this.upPyramid[level]!;
                    this.upsamplePass.execute(ctx, res[0]!, res[1]!);
                }
            }
        }

        {
            const root = this.postFXPass.getRootVar();
            root["PerFrameCB"]["gResolution"] = [width, height];
            root["PerFrameCB"]["gInvRes"] = [1 / width, 1 / height];
            root["PerFrameCB"]["gSrcRes"] = [width, height];
            root["PerFrameCB"]["gVignetteAmount"] = this.mVignetteAmount;
            root["PerFrameCB"]["gChromaticAberrationAmount"] = this.mChromaticAberrationAmount / 64;
            const barrel = this.mBarrelDistortAmount * 0.125;
            root["PerFrameCB"]["gBarrelDistort"] = [1 / (1 + 4 * barrel), barrel];
            const [sx, sy, sz] = this.mSaturationCurve;
            const cy = sy - sx;
            const cz = sz - sx;
            const A = 2 * cz - 4 * cy;
            root["PerFrameCB"]["gSaturationCurve"] = [A, cz - A, sx];
            root["PerFrameCB"]["gColorOffset"] = this.mColorOffset.map((v) => v + this.mColorOffsetScalar - 0.5);
            const scaleMult = Math.pow(2, 1 + 2 * this.mColorScaleScalar);
            root["PerFrameCB"]["gColorScale"] = this.mColorScale.map((v) => v * scaleMult);
            root["PerFrameCB"]["gColorPower"] = this.mColorPower.map((v) => Math.pow(2, 3 * (0.5 - v - this.mColorPowerScalar)));
            root["PerFrameCB"]["gWipe"] = this.mWipe * width;
            root["gBloomed"] = this.mBloomAmount > 0 ? this.upPyramid[0]! : src;
            root["gSrc"] = src;
            root["gDst"] = dst;
            this.postFXPass.execute(ctx, width, height);
        }
    }

    /** Mirrors preparePostFX (plus the web-only up-pyramid). */
    private preparePyramids(width: number, height: number): void {
        const make = (w: number, h: number, name: string) =>
            new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: w,
                height: h,
                format: ResourceFormat.RGBA16Float,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                name,
            });
        for (let res = 0; res < kNumLevels + 1; res++) {
            const w = Math.max(1, width >> res);
            const h = Math.max(1, height >> res);
            if (!this.pyramid[res] || this.pyramid[res]!.width !== w || this.pyramid[res]!.height !== h) {
                this.pyramid[res] = make(w, h, `SimplePostFX::pyramid[${res}]`);
            }
            if (res < kNumLevels && (!this.upPyramid[res] || this.upPyramid[res]!.width !== w || this.upPyramid[res]!.height !== h)) {
                this.upPyramid[res] = make(w, h, `SimplePostFX::upPyramid[${res}]`);
            }
        }
    }
}

registerRenderPass("SimplePostFX", (device, props) => new SimplePostFX(device, props));
