/**
 * Temporal accumulation pass mirroring Source/RenderPasses/AccumulatePass.
 * Shader is the WebFalcor override (buffer-backed state); logic matches upstream.
 * Double precision mode is impossible in WGSL (no fp64/i64) and maps to
 * SingleCompensated with a warning (parity matrix §8.2).
 */

import {
    IOSize,
    parseIOSize,
    calculateIOSize,
    Buffer,
    ComputePass,
    Logger,
    MemoryType,
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
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/AccumulatePass/Accumulate.cs.slang";

export enum AccumulatePrecision {
    Double = 0,
    Single = 1,
    SingleCompensated = 2,
}

export class AccumulatePass extends RenderPass {
    private enabled = true;
    private outputSize = IOSize.Default;
    private precision = AccumulatePrecision.Single;
    private autoReset = true;
    private frameCount = 0;
    private pass: ComputePass | null = null;
    private lastFrameSum: Buffer | null = null;
    private lastFrameCorr: Buffer | null = null;
    private dims: [number, number] = [0, 0];

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        this.enabled = props.get("enabled", true);
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        const mode = props.getOpt<string | number>("precisionMode");
        if (mode !== undefined) {
            const parsed = typeof mode === "string" ? AccumulatePrecision[mode as keyof typeof AccumulatePrecision] : mode;
            this.precision = parsed ?? AccumulatePrecision.Single;
        }
        if (this.precision === AccumulatePrecision.Double) {
            Logger.warning("AccumulatePass: Double precision unavailable in WGSL; using SingleCompensated (docs §8.2)");
            this.precision = AccumulatePrecision.SingleCompensated;
        }
        this.autoReset = props.get("autoReset", true);
    }

    override getProperties(): Properties {
        return new Properties({ enabled: this.enabled, precisionMode: AccumulatePrecision[this.precision]!, autoReset: this.autoReset });
    }

    reset(): void {
        this.frameCount = 0;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, [512, 512], compileData.defaultTexDims);
        r.addInput("input", "Input data to be temporally accumulated").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("output", "Accumulated output")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const input = renderData.getTexture("input")!;
        const output = renderData.getTexture("output")!;
        const [w, h] = [output.width, output.height];

        if (this.dims[0] !== w || this.dims[1] !== h) {
            this.dims = [w, h];
            const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            this.lastFrameSum = new Buffer(this.device, { size: w * h * 16, structSize: 16, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name: "AccumulatePass::sum" });
            this.lastFrameCorr = new Buffer(this.device, { size: w * h * 16, structSize: 16, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name: "AccumulatePass::corr" });
            this.frameCount = 0;
        }
        if (this.frameCount === 0) {
            ctx.clearBuffer(this.lastFrameSum!);
            ctx.clearBuffer(this.lastFrameCorr!);
        }

        if (!this.pass) {
            const entry = this.precision === AccumulatePrecision.SingleCompensated ? "accumulateSingleCompensated" : "accumulateSingle";
            this.pass = ComputePass.create(this.device, { path: kShaderFile, csEntry: entry, defines: { _INPUT_FORMAT: 0 } });
        }

        const root = this.pass.getRootVar();
        root["PerFrameCB"]["gResolution"] = [w, h];
        root["PerFrameCB"]["gAccumCount"] = this.frameCount;
        root["PerFrameCB"]["gAccumulate"] = this.enabled;
        root["PerFrameCB"]["gMovingAverageMode"] = false;
        root["gCurFrame"] = input;
        root["gOutputFrame"] = output;
        root["gLastFrameSum"] = this.lastFrameSum!;
        root["gLastFrameCorr"] = this.lastFrameCorr!;

        this.pass.execute(ctx, w, h);
        if (this.enabled) this.frameCount++;
    }
}

registerRenderPass("AccumulatePass", (device, props) => new AccumulatePass(device, props));
