/**
 * Error measurement pass mirroring Source/RenderPasses/ErrorMeasurePass:
 * per-pixel |source - reference| (optionally squared/averaged) with the mean
 * error reduced on the GPU. The reference comes from the optional input or a
 * loaded image file (EXR/HDR/browser-decodable). Web divergence (docs §9):
 * measurements surface on `measurements` via async readback instead of the
 * native csv file.
 */

import {
    ComputePass,
    ParallelReduction,
    ParallelReductionType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    RuntimeError,
    FieldFlags,
    Texture,
    decodeExr,
    decodeHdr,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/ErrorMeasurePass/ErrorMeasurer.cs.slang";
const kMediaBaseUrl = "/Falcor/media/";

export class ErrorMeasurePass extends RenderPass {
    private referenceImagePath = "";
    private ignoreBackground = true;
    private computeSquaredDifference = true;
    private computeAverage = false;
    private selectedOutput = "Source"; // Source | Reference | Difference (native default)
    private pass: ComputePass | null = null;
    private reduction: ParallelReduction | null = null;
    private differenceTexture: Texture | null = null;
    private referenceTexture: Texture | null = null;
    private dummyWorldPos: Texture | null = null;
    private readbackInFlight = false;

    /** Mirrors ErrorMeasurePass::mMeasurements (async readback; ~1 frame late). */
    measurements: { valid: boolean; error: [number, number, number]; avgError: number } = { valid: false, error: [0, 0, 0], avgError: 0 };

    /** Mirrors mRunningError/mRunningAvgError: exponential moving average, one
     * step per landed measurement (native steps per frame; the readback is the
     * web's measurement cadence). runningAvgError < 0 means no sample yet. */
    runningError: [number, number, number] = [0, 0, 0];
    runningAvgError = -1;
    private reportRunningError = true;
    private runningErrorSigma = 0.995;

    constructor(device: Device, props: Properties) {
        super(device);
        this.referenceImagePath = props.get("ReferenceImagePath", "");
        this.ignoreBackground = props.get("IgnoreBackground", true);
        this.computeSquaredDifference = props.get("ComputeSquaredDifference", true);
        this.computeAverage = props.get("ComputeAverage", false);
        this.selectedOutput = props.get("SelectedOutputId", "Source");
        this.reportRunningError = props.get("ReportRunningError", true);
        this.runningErrorSigma = props.get("RunningErrorSigma", 0.995);
        // 'MeasurementsFilePath' accepted: no file IO on the web (docs §9).
    }

    override async initAsync(): Promise<void> {
        this.runningAvgError = -1; // Mirrors loadReference: running error restarts with a new reference.
        if (!this.referenceImagePath) return;
        const url = kMediaBaseUrl + this.referenceImagePath;
        const res = await fetch(url);
        if (!res.ok) throw new RuntimeError(`ErrorMeasurePass: failed to fetch reference '${url}' (${res.status})`);
        const lower = this.referenceImagePath.toLowerCase();
        if (lower.endsWith(".exr")) {
            const exr = decodeExr(await res.arrayBuffer());
            this.referenceTexture = this.device.createTexture2D(exr.width, exr.height, ResourceFormat.RGBA32Float, 1, 1, exr.data);
        } else if (lower.endsWith(".hdr")) {
            const hdr = decodeHdr(new Uint8Array(await res.arrayBuffer()));
            this.referenceTexture = this.device.createTexture2D(hdr.width, hdr.height, ResourceFormat.RGBA32Float, 1, 1, hdr.data);
        } else {
            const bitmap = await createImageBitmap(await res.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
            const tex = this.device.createTexture2D(bitmap.width, bitmap.height, ResourceFormat.RGBA8UnormSrgb, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
            this.device.gpuDevice.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex.gpuTexture }, [bitmap.width, bitmap.height]);
            this.referenceTexture = tex;
        }
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("Source", "Source image").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("Reference", "Reference image (optional)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("WorldPosition", "World-space position").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("Output", "Output image")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const source = renderData.getTexture("Source")!;
        const output = renderData.getTexture("Output")!;
        const reference = renderData.getTexture("Reference") ?? this.referenceTexture;
        if (!reference) {
            this.measurements.valid = false;
            ctx.blit(source, output);
            return;
        }

        const [w, h] = [source.width, source.height];
        if (!this.differenceTexture || this.differenceTexture.width !== w || this.differenceTexture.height !== h) {
            this.differenceTexture = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: w,
                height: h,
                format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                mipLevels: 1,
                name: "ErrorMeasurePass::difference",
            });
        }

        const worldPos = renderData.getTexture("WorldPosition");
        if (!worldPos && !this.dummyWorldPos) {
            this.dummyWorldPos = this.device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array([0, 0, 0, 0]));
        }

        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        root["gReference"] = reference;
        root["gSource"] = source;
        root["gWorldPosition"] = worldPos ?? this.dummyWorldPos!;
        root["PerFrameCB"]["gResolution"] = [w, h];
        // Native requires the world-position input for background rejection.
        root["PerFrameCB"]["gIgnoreBackground"] = this.ignoreBackground && worldPos ? 1 : 0;
        root["PerFrameCB"]["gComputeDiffSqr"] = this.computeSquaredDifference ? 1 : 0;
        root["PerFrameCB"]["gComputeAverage"] = this.computeAverage ? 1 : 0;
        root["gResult"] = this.differenceTexture;
        this.pass.execute(ctx, w, h);

        // Mean error via GPU reduction (native runReductionPasses); async
        // readback lands ~1 frame later.
        this.reduction ??= new ParallelReduction(this.device);
        if (!this.readbackInFlight) {
            this.readbackInFlight = true;
            void this.reduction.execute(ctx, this.differenceTexture, ParallelReductionType.Sum).then((sum) => {
                const n = w * h;
                const error: [number, number, number] = [sum[0]! / n, sum[1]! / n, sum[2]! / n];
                const avgError = (error[0] + error[1] + error[2]) / 3;
                this.measurements = { valid: true, error, avgError };
                // Mirrors the native running-error update (endFrame).
                if (this.runningAvgError < 0) {
                    this.runningError = [...error];
                    this.runningAvgError = avgError;
                } else {
                    const s = this.runningErrorSigma;
                    this.runningError = [
                        s * this.runningError[0] + (1 - s) * error[0],
                        s * this.runningError[1] + (1 - s) * error[1],
                        s * this.runningError[2] + (1 - s) * error[2],
                    ];
                    this.runningAvgError = s * this.runningAvgError + (1 - s) * avgError;
                }
                this.readbackInFlight = false;
            });
        }

        switch (this.selectedOutput) {
            case "Reference":
                ctx.blit(reference, output);
                break;
            case "Difference":
                ctx.blit(this.differenceTexture, output);
                break;
            default:
                ctx.blit(source, output);
        }
    }

    override renderUI(ui: UIWidgets): void {
        ui.dropdown("Output", ["Source", "Reference", "Difference"], this.selectedOutput, (v) => (this.selectedOutput = v));
        ui.checkbox("Ignore background", this.ignoreBackground, (v) => (this.ignoreBackground = v));
        ui.checkbox("Compute squared difference", this.computeSquaredDifference, (v) => (this.computeSquaredDifference = v));
        ui.checkbox("Compute average", this.computeAverage, (v) => (this.computeAverage = v));
        ui.checkbox("Report running error", this.reportRunningError, (v) => {
            this.reportRunningError = v;
            if (v) this.runningAvgError = -1; // native resets the average on enable
        });
        if (!this.measurements.valid) {
            ui.text("No measurements (reference missing)");
            return;
        }
        const useRunning = this.reportRunningError && this.runningAvgError >= 0;
        const err = useRunning ? this.runningError : this.measurements.error;
        const avg = useRunning ? this.runningAvgError : this.measurements.avgError;
        ui.text(`${useRunning ? "Running avg" : "Avg"} error: ${avg.toExponential(3)} (${err.map((x) => x.toExponential(2)).join(", ")})`);
    }
}

registerRenderPass("ErrorMeasurePass", (device, props) => new ErrorMeasurePass(device, props));
