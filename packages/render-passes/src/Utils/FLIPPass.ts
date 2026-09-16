/**
 * FLIP error-metric pass mirroring Source/RenderPasses/FLIPPass: LDR + HDR.
 * Web divergence (docs §9): the HDR auto-exposure parameters come from an
 * async luminance readback (native blocks), so they land ~1 frame late —
 * the first HDR frame renders with the previous (or default) exposure range.
 * Pooled FLIP values (average/min/max) likewise land asynchronously on
 * `averageFLIP`/`minFLIP`/`maxFLIP`. Monitor info uses the native headless
 * defaults (useRealMonitorInfo has no browser equivalent for physical size).
 *
 * Note: native binds gClampInput from mUseMagma (upstream quirk) — replicated
 * for 1:1 output parity.
 */

import {
    Buffer,
    ComputePass,
    MemoryType,
    ParallelReduction,
    ParallelReductionType,
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

const kShaderFile = "RenderPasses/FLIPPass/FLIPPass.cs.slang";
const kLuminanceShaderFile = "RenderPasses/FLIPPass/ComputeLuminance.cs.slang";

/** Mirrors solveSecondDegree (FLIPPass.cpp). */
function solveSecondDegree(a: number, b: number, c: number): [number, number] {
    if (a === 0) {
        const x = -c / b;
        return [x, x];
    }
    const d1 = -0.5 * (b / a);
    const d2 = Math.sqrt(d1 * d1 - c / a);
    return [d1 - d2, d1 + d2];
}

/** Mirrors computeMedianMax (FLIPPass.cpp): median + max of the luminance values. */
export function computeMedianMax(values: Float32Array): [number, number] {
    const sorted = values.slice().sort();
    const n = sorted.length;
    const median = n & 1 ? sorted[n >> 1]! : (sorted[n / 2 - 1]! + sorted[n / 2]!) * 0.5;
    return [median, sorted[n - 1]!];
}

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
    private luminancePass: ComputePass | null = null;
    private luminanceBuffer: Buffer | null = null;
    private reduction: ParallelReduction | null = null;
    private errorMapDisplay: Texture | null = null;
    private exposureMapDisplay: Texture | null = null;
    private exposureReadbackInFlight = false;
    private pooledReadbackInFlight = false;

    /** Mirrors mAverageFLIP/mMinFLIP/mMaxFLIP (async readback; NaN until the first landing). */
    averageFLIP = NaN;
    minFLIP = NaN;
    maxFLIP = NaN;

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

    /** Mirrors FLIPPass::renderUI (tone mapper is a shader define -> kernel rebuild). */
    override renderUI(ui: UIWidgets): void {
        ui.checkbox("Enabled", this.enabled, (v) => (this.enabled = v));
        ui.text("FLIP Settings:");
        ui.checkbox("Use Magma", this.useMagma, (v) => (this.useMagma = v));
        ui.checkbox("Clamp input", this.clampInput, (v) => (this.clampInput = v));
        ui.checkbox("Input is HDR", this.isHDR, (v) => (this.isHDR = v));
        const tms = Object.keys(FLIPToneMapperType).filter((k) => isNaN(Number(k)));
        ui.dropdown("Tone mapper", tms, FLIPToneMapperType[this.toneMapper]!, (v) => {
            this.toneMapper = FLIPToneMapperType[v as keyof typeof FLIPToneMapperType];
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines: { TONE_MAPPER: this.toneMapper } });
        });
        ui.checkbox("Use custom exposure parameters", this.useCustomExposureParameters, (v) => (this.useCustomExposureParameters = v));
        const custom = (set: (v: number) => void) => (v: number) => {
            set(v);
            this.exposureDelta = (this.stopExposure - this.startExposure) / (this.numExposures - 1);
        };
        ui.slider("Start exposure", this.startExposure, -20, 20, 0.01, custom((v) => (this.startExposure = v)));
        ui.slider("Stop exposure", this.stopExposure, -20, 20, 0.01, custom((v) => (this.stopExposure = v)));
        ui.slider("Number of exposures", this.numExposures, 2, 20, 1, custom((v) => (this.numExposures = Math.round(v))));
        ui.checkbox("Per-frame metrics", this.computePooledFLIPValues, (v) => (this.computePooledFLIPValues = v));
        if (this.computePooledFLIPValues && Number.isFinite(this.averageFLIP)) {
            ui.text(`Average: ${this.averageFLIP.toFixed(4)}  Min: ${this.minFLIP.toFixed(4)}  Max: ${this.maxFLIP.toFixed(4)}`);
        }
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

        if (this.useCustomExposureParameters) {
            // Mirrors the native UI path: delta derived from the custom start/stop range.
            this.exposureDelta = (this.stopExposure - this.startExposure) / (this.numExposures - 1);
        } else if (this.isHDR) {
            // HDR auto-exposure from the reference luminance. Like native, the
            // parameters computed from this frame's readback apply to the NEXT
            // frame (native also copies the members into the cbuffer before
            // recomputing them); the web readback is just async as well.
            if (!this.luminanceBuffer || this.luminanceBuffer.size !== w * h * 4) {
                this.luminanceBuffer = new Buffer(this.device, {
                    size: w * h * 4,
                    structSize: 4,
                    bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                    memoryType: MemoryType.DeviceLocal,
                    name: "FLIPPass::luminance",
                });
            }
            this.luminancePass ??= ComputePass.create(this.device, { path: kLuminanceShaderFile, csEntry: "computeLuminance" });
            const lroot = this.luminancePass.getRootVar();
            lroot["gInputImage"] = reference;
            lroot["gOutputLuminance"] = this.luminanceBuffer;
            lroot["PerFrameCB"]["gResolution"] = [w, h];
            this.luminancePass.execute(ctx, w, h);
            if (!this.exposureReadbackInFlight) {
                this.exposureReadbackInFlight = true;
                const n = w * h;
                void this.luminanceBuffer
                    .getBlob()
                    .then((bytes) => {
                        const [median, max] = computeMedianMax(new Float32Array(bytes.buffer, bytes.byteOffset, n));
                        this.computeExposureParameters(median, max);
                    })
                    .finally(() => {
                        this.exposureReadbackInFlight = false;
                    });
            }
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

        // Mean/min/max FLIP via parallel reduction (alpha channel holds the FLIP value).
        if (this.computePooledFLIPValues && !this.pooledReadbackInFlight) {
            this.reduction ??= new ParallelReduction(this.device);
            this.pooledReadbackInFlight = true;
            const n = w * h;
            void Promise.all([
                this.reduction.execute(ctx, errorMap, ParallelReductionType.Sum),
                this.reduction.execute(ctx, errorMap, ParallelReductionType.MinMax),
            ])
                .then(([sum, minMax]) => {
                    this.averageFLIP = sum[3]! / n;
                    this.minFLIP = minMax[3]!;
                    this.maxFLIP = minMax[7]!;
                })
                .finally(() => {
                    this.pooledReadbackInFlight = false;
                });
        }
    }

    /** Current HDR exposure parameters (auto-computed values land async). */
    getExposureParameters(): { startExposure: number; stopExposure: number; exposureDelta: number; numExposures: number } {
        return {
            startExposure: this.startExposure,
            stopExposure: this.stopExposure,
            exposureDelta: this.exposureDelta,
            numExposures: this.numExposures,
        };
    }

    /** Mirrors FLIPPass::computeExposureParameters (tone-mapper-specific range solve). */
    private computeExposureParameters(Ymedian: number, Ymax: number): void {
        let tm: number[];
        if (this.toneMapper === FLIPToneMapperType.Reinhard) {
            tm = [0, 1, 0, 0, 1, 1];
        } else if (this.toneMapper === FLIPToneMapperType.ACES) {
            // 0.6 is pre-exposure cancellation.
            tm = [0.6 * 0.6 * 2.51, 0.6 * 0.03, 0, 0.6 * 0.6 * 2.43, 0.6 * 0.59, 0.14];
        } else {
            tm = [0.231683, 0.013791, 0, 0.18, 0.3, 0.018];
        }
        const t = 0.85;
        const [, xMax] = solveSecondDegree(tm[0]! - t * tm[3]!, tm[1]! - t * tm[4]!, tm[2]! - t * tm[5]!);

        this.startExposure = Math.log2(xMax / Ymax);
        this.stopExposure = Math.log2(xMax / Ymedian);
        this.numExposures = Math.max(2, Math.ceil(this.stopExposure - this.startExposure));
        this.exposureDelta = (this.stopExposure - this.startExposure) / (this.numExposures - 1);
    }
}

registerRenderPass("FLIPPass", (device, props) => new FLIPPass(device, props));
