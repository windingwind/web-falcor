/**
 * FLIP error-metric pass mirroring Source/RenderPasses/FLIPPass. The LDR path
 * (default) is complete. HDR auto-exposure needs a synchronous luminance
 * readback natively; on the web it would land a frame late (like ColorMapPass
 * auto-range) — deferred until an HDR-FLIP graph needs it, as is pooled-value
 * reduction (UI-only). Monitor info uses the native headless defaults
 * (useRealMonitorInfo has no browser equivalent for physical size).
 *
 * Note: native binds gClampInput from mUseMagma (upstream quirk) — replicated
 * for 1:1 output parity.
 */

import {
    ComputePass,
    Logger,
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

const kShaderFile = "RenderPasses/FLIPPass/FLIPPass.cs.slang";

export enum FLIPToneMapperType {
    ACES = 0,
    Hable = 1,
    Reinhard = 2,
}

export class FLIPPass extends RenderPass {
    private enabled = true;
    private isHDR = false;
    private toneMapper = FLIPToneMapperType.ACES;
    private useCustomExposureParameters = false;
    private startExposure = 0;
    private stopExposure = 0;
    private exposureDelta = 0;
    private numExposures = 2;
    private useMagma = true;
    private clampInput = false;
    private monitorWidthPixels = 3840;
    private monitorWidthMeters = 0.7;
    private monitorDistanceMeters = 0.7;
    private computePooledFLIPValues = false;

    private pass: ComputePass;
    private errorMapDisplay: Texture | null = null;
    private exposureMapDisplay: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.enabled = props.get("enabled", true);
        this.isHDR = props.get("isHDR", false);
        const tm = props.getOpt<string | number>("toneMapper");
        if (tm !== undefined) this.toneMapper = (typeof tm === "string" ? FLIPToneMapperType[tm as keyof typeof FLIPToneMapperType] : tm) ?? FLIPToneMapperType.ACES;
        this.useCustomExposureParameters = props.get("useCustomExposureParameters", false);
        this.startExposure = props.get("startExposure", 0);
        this.stopExposure = props.get("stopExposure", 0);
        this.numExposures = props.get("numExposures", 2);
        this.useMagma = props.get("useMagma", true);
        this.clampInput = props.get("clampInput", false);
        this.monitorWidthPixels = props.get("monitorWidthPixels", 3840);
        this.monitorWidthMeters = props.get("monitorWidthMeters", 0.7);
        this.monitorDistanceMeters = props.get("monitorDistanceMeters", 0.7);
        this.computePooledFLIPValues = props.get("computePooledFLIPValues", false);
        // 'useRealMonitorInfo' accepted; headless defaults used (see header).

        this.pass = ComputePass.create(device, { path: kShaderFile, defines: { TONE_MAPPER: this.toneMapper } });
    }

    override getProperties(): Properties {
        return new Properties({
            enabled: this.enabled,
            useMagma: this.useMagma,
            clampInput: this.clampInput,
            isHDR: this.isHDR,
            toneMapper: this.toneMapper,
            useCustomExposureParameters: this.useCustomExposureParameters,
            startExposure: this.startExposure,
            stopExposure: this.stopExposure,
            numExposures: this.numExposures,
            monitorWidthPixels: this.monitorWidthPixels,
            monitorWidthMeters: this.monitorWidthMeters,
            monitorDistanceMeters: this.monitorDistanceMeters,
            computePooledFLIPValues: this.computePooledFLIPValues,
        });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("testImage", "Test image").bindFlags(ResourceBindFlags.ShaderResource).texture2D(0, 0);
        r.addInput("referenceImage", "Reference image").bindFlags(ResourceBindFlags.ShaderResource).texture2D(0, 0);
        r.addOutput("errorMap", "FLIP error map for computations")
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
            .texture2D(0, 0);
        r.addOutput("errorMapDisplay", "FLIP error map for display")
            .format(ResourceFormat.RGBA8UnormSrgb)
            .bindFlags(ResourceBindFlags.RenderTarget)
            .texture2D(0, 0);
        r.addOutput("exposureMapDisplay", "HDR-FLIP exposure map for display")
            .format(ResourceFormat.RGBA8UnormSrgb)
            .bindFlags(ResourceBindFlags.RenderTarget)
            .texture2D(0, 0);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.enabled) return;
        const test = renderData.getTexture("testImage")!;
        const reference = renderData.getTexture("referenceImage")!;
        const errorMap = renderData.getTexture("errorMap")!;
        const errorMapDisplayOut = renderData.getTexture("errorMapDisplay")!;
        const exposureMapDisplayOut = renderData.getTexture("exposureMapDisplay")!;

        const [w, h] = [reference.width, reference.height];
        if (!this.errorMapDisplay || this.errorMapDisplay.width !== w || this.errorMapDisplay.height !== h) {
            const make = (name: string) =>
                new Texture(this.device, {
                    type: ResourceType.Texture2D,
                    width: w,
                    height: h,
                    format: ResourceFormat.RGBA32Float,
                    bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource,
                    name,
                });
            this.errorMapDisplay = make("FLIPPass::errorMapDisplay");
            this.exposureMapDisplay = make("FLIPPass::exposureMapDisplay");
        }

        if (this.isHDR && !this.useCustomExposureParameters) {
            Logger.warning("FLIPPass: HDR auto-exposure needs a synchronous readback; not ported yet (LDR path is 1:1).");
        }

        const root = this.pass.getRootVar();
        root["gTestImage"] = test;
        root["gReferenceImage"] = reference;
        root["gFLIPErrorMap"] = errorMap;
        root["gFLIPErrorMapDisplay"] = this.errorMapDisplay;
        root["gExposureMapDisplay"] = this.exposureMapDisplay;
        const cb = root["PerFrameCB"];
        cb["gIsHDR"] = this.isHDR ? 1 : 0;
        cb["gUseMagma"] = this.useMagma ? 1 : 0;
        cb["gClampInput"] = this.useMagma ? 1 : 0; // native quirk: bound from mUseMagma
        cb["gResolution"] = [w, h];
        cb["gMonitorWidthPixels"] = this.monitorWidthPixels;
        cb["gMonitorWidthMeters"] = this.monitorWidthMeters;
        cb["gMonitorDistance"] = this.monitorDistanceMeters;
        cb["gStartExposure"] = this.startExposure;
        cb["gExposureDelta"] = this.exposureDelta;
        cb["gNumExposures"] = this.numExposures;
        this.pass.execute(ctx, w, h);

        ctx.blit(this.errorMapDisplay, errorMapDisplayOut);
        ctx.blit(this.exposureMapDisplay!, exposureMapDisplayOut);

        if (this.computePooledFLIPValues) {
            Logger.warning("FLIPPass: pooled FLIP values (UI-only) not ported yet.");
        }
    }
}

registerRenderPass("FLIPPass", (device, props) => new FLIPPass(device, props));
