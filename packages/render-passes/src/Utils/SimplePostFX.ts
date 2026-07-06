/**
 * Simple post FX pass mirroring Source/RenderPasses/SimplePostFX: bloom via an
 * 8-level pyramid, star lobes, vignette, chromatic aberration, barrel
 * distortion, color grading. Web divergence (documented in the shader
 * override): upsampling ping-pongs a second pyramid because WGSL forbids
 * rgba16float read_write storage and WebGPU forbids sample+store of one
 * texture in a pass; border-mode sampling is emulated in-shader.
 */

import {
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
    private enabled = true;
    private wipe = 0;
    private bloomAmount = 0;
    private starAmount = 0;
    private starAngle = 0.1;
    private vignetteAmount = 0;
    private chromaticAberrationAmount = 0;
    private barrelDistortAmount = 0;
    private saturationCurve: [number, number, number] = [1, 1, 1];
    private colorOffset: [number, number, number] = [0.5, 0.5, 0.5];
    private colorScale: [number, number, number] = [0.5, 0.5, 0.5];
    private colorPower: [number, number, number] = [0.5, 0.5, 0.5];
    private colorOffsetScalar = 0;
    private colorScaleScalar = 0;
    private colorPowerScalar = 0;

    private downsamplePass: ComputePass;
    private upsamplePass: ComputePass;
    private postFXPass: ComputePass;
    /** pyramid[0..kNumLevels]: downsample chain; upPyramid[0..kNumLevels-1]: upsample ping-pong (web divergence). */
    private pyramid: (Texture | null)[] = new Array<Texture | null>(kNumLevels + 1).fill(null);
    private upPyramid: (Texture | null)[] = new Array<Texture | null>(kNumLevels).fill(null);

    constructor(device: Device, props: Properties) {
        super(device);
        this.enabled = props.get("enabled", true);
        this.wipe = props.get("wipe", 0);
        this.bloomAmount = props.get("bloomAmount", 0);
        this.starAmount = props.get("starAmount", 0);
        this.starAngle = props.get("starAngle", 0.1);
        this.vignetteAmount = props.get("vignetteAmount", 0);
        this.chromaticAberrationAmount = props.get("chromaticAberrationAmount", 0);
        this.barrelDistortAmount = props.get("barrelDistortAmount", 0);
        this.saturationCurve = toVec3(props.getOpt("saturationCurve"), [1, 1, 1]);
        this.colorOffset = toVec3(props.getOpt("colorOffset"), [0.5, 0.5, 0.5]);
        this.colorScale = toVec3(props.getOpt("colorScale"), [0.5, 0.5, 0.5]);
        this.colorPower = toVec3(props.getOpt("colorPower"), [0.5, 0.5, 0.5]);
        this.colorOffsetScalar = props.get("colorOffsetScalar", 0);
        this.colorScaleScalar = props.get("colorScaleScalar", 0);
        this.colorPowerScalar = props.get("colorPowerScalar", 0);
        // 'outputSize'/'fixedOutputSize' accepted; IOSize plumbing lands with HalfRes.

        this.downsamplePass = ComputePass.create(device, { path: kShaderFile, csEntry: "downsample" });
        this.upsamplePass = ComputePass.create(device, { path: kShaderFile, csEntry: "upsample" });
        this.postFXPass = ComputePass.create(device, { path: kShaderFile, csEntry: "runPostFX" });
    }

    override getProperties(): Properties {
        return new Properties({
            enabled: this.enabled,
            wipe: this.wipe,
            bloomAmount: this.bloomAmount,
            starAmount: this.starAmount,
            starAngle: this.starAngle,
            vignetteAmount: this.vignetteAmount,
            chromaticAberrationAmount: this.chromaticAberrationAmount,
            barrelDistortAmount: this.barrelDistortAmount,
            saturationCurve: this.saturationCurve,
            colorOffset: this.colorOffset,
            colorScale: this.colorScale,
            colorPower: this.colorPower,
            colorOffsetScalar: this.colorOffsetScalar,
            colorScaleScalar: this.colorScaleScalar,
            colorPowerScalar: this.colorPowerScalar,
        });
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
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

        if (this.enabled && (src.width !== dst.width || src.height !== dst.height)) {
            throw new Error("SimplePostFX I/O sizes don't match.");
        }
        const [width, height] = [src.width, src.height];

        const isDefault =
            this.bloomAmount === 0 &&
            this.chromaticAberrationAmount === 0 &&
            this.barrelDistortAmount === 0 &&
            this.saturationCurve.every((v) => v === 1) &&
            this.colorOffset.every((v) => v === 0.5) &&
            this.colorScale.every((v) => v === 0.5) &&
            this.colorPower.every((v) => v === 0.5) &&
            this.colorOffsetScalar === 0 &&
            this.colorScaleScalar === 0 &&
            this.colorPowerScalar === 0;
        if (!this.enabled || this.wipe >= 1 || isDefault) {
            ctx.blit(src, dst);
            return;
        }

        if (this.bloomAmount > 0) {
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
                root["PerFrameCB"]["gBloomAmount"] = this.bloomAmount;
                for (let level = kNumLevels - 1; level >= 0; level--) {
                    const res = [Math.max(1, width >> level), Math.max(1, height >> level)];
                    const invres = [1 / res[0]!, 1 / res[1]!];
                    const bloomed = level === kNumLevels - 1 ? this.pyramid[level + 1]! : this.upPyramid[level + 1]!;
                    root["PerFrameCB"]["gResolution"] = res;
                    root["PerFrameCB"]["gInvRes"] = invres;
                    root["PerFrameCB"]["gSrcRes"] = [bloomed.width, bloomed.height];
                    const wantStar = level === 1 || level === 2;
                    root["PerFrameCB"]["gStar"] = wantStar ? this.starAmount : 0;
                    if (wantStar) {
                        let ang = this.starAngle;
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
            root["PerFrameCB"]["gVignetteAmount"] = this.vignetteAmount;
            root["PerFrameCB"]["gChromaticAberrationAmount"] = this.chromaticAberrationAmount / 64;
            const barrel = this.barrelDistortAmount * 0.125;
            root["PerFrameCB"]["gBarrelDistort"] = [1 / (1 + 4 * barrel), barrel];
            const [sx, sy, sz] = this.saturationCurve;
            const cy = sy - sx;
            const cz = sz - sx;
            const A = 2 * cz - 4 * cy;
            root["PerFrameCB"]["gSaturationCurve"] = [A, cz - A, sx];
            root["PerFrameCB"]["gColorOffset"] = this.colorOffset.map((v) => v + this.colorOffsetScalar - 0.5);
            const scaleMult = Math.pow(2, 1 + 2 * this.colorScaleScalar);
            root["PerFrameCB"]["gColorScale"] = this.colorScale.map((v) => v * scaleMult);
            root["PerFrameCB"]["gColorPower"] = this.colorPower.map((v) => Math.pow(2, 3 * (0.5 - v - this.colorPowerScalar)));
            root["PerFrameCB"]["gWipe"] = this.wipe * width;
            root["gBloomed"] = this.bloomAmount > 0 ? this.upPyramid[0]! : src;
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
