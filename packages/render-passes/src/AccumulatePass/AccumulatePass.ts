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
    type Scene,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/AccumulatePass/Accumulate.cs.slang";

export enum AccumulatePrecision {
    Double = 0,
    Single = 1,
    SingleCompensated = 2,
}

/** Mirrors AccumulatePass::OverflowMode. */
export enum AccumulateOverflowMode {
    Stop = 0,
    Reset = 1,
    EMA = 2,
}

export class AccumulatePass extends RenderPass {
    private enabled = true;
    /** 0 = unlimited; otherwise frames beyond it follow overflowMode (native mMaxFrameCount). */
    private maxFrameCount = 0;
    private overflowMode = AccumulateOverflowMode.Stop;
    private outputSize = IOSize.Default;
    /** Native kFixedOutputSize default (used when outputSize == Fixed). */
    private fixedOutputSize: [number, number] = [512, 512];
    private precision = AccumulatePrecision.Single;
    private autoReset = true;
    /** Scene update version + camera state of the previous frame (autoReset). */
    private lastSceneKey: string | undefined;
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
        const fixed = props.getOpt<number[] | { x: number; y: number }>("fixedOutputSize");
        if (fixed) this.fixedOutputSize = Array.isArray(fixed) ? [fixed[0]!, fixed[1]!] : [fixed.x, fixed.y];
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
        this.maxFrameCount = props.get("maxFrameCount", 0);
        const overflow = props.getOpt<string | number>("overflowMode");
        if (overflow !== undefined) this.overflowMode = (typeof overflow === "string" ? AccumulateOverflowMode[overflow as keyof typeof AccumulateOverflowMode] : overflow) ?? this.overflowMode;
    }

    override getProperties(): Properties {
        return new Properties({ enabled: this.enabled, precisionMode: AccumulatePrecision[this.precision]!, autoReset: this.autoReset, outputSize: IOSize[this.outputSize]!, fixedOutputSize: this.fixedOutputSize, maxFrameCount: this.maxFrameCount, overflowMode: AccumulateOverflowMode[this.overflowMode]! });
    }

    reset(): void {
        this.frameCount = 0;
    }

    /** Mirrors AccumulatePass::setScene: accumulation restarts with a new scene. */
    override setScene(scene: Scene | null): void {
        super.setScene(scene);
        this.reset();
        this.lastSceneKey = undefined;
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
        ui.checkbox("Enabled", this.enabled, (v) => (this.enabled = v));
        ui.button("Reset", () => this.reset());
        ui.checkbox("Auto Reset", this.autoReset, (v) => (this.autoReset = v));
        ui.dropdown("Precision", ["Single", "SingleCompensated"], AccumulatePrecision[this.precision]!, (v) => {
            this.precision = AccumulatePrecision[v as keyof typeof AccumulatePrecision];
            this.pass = null; // precision selects the compute entry point — rebuild
            this.reset();
        });
        // Native: the frame limit is not supported in SingleCompensated mode.
        ui.slider("Max Frames", this.maxFrameCount, 0, 4096, 1, (v) => {
            this.maxFrameCount = Math.round(v);
            this.reset();
        });
        ui.dropdown("Overflow Mode", ["Stop", "Reset", "EMA"], AccumulateOverflowMode[this.overflowMode]!, (v) => {
            this.overflowMode = AccumulateOverflowMode[v as keyof typeof AccumulateOverflowMode];
            this.reset();
        });
        ui.text(`Frames accumulated ${this.frameCount}`);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, this.fixedOutputSize, compileData.defaultTexDims);
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

        // AccumulatePass::execute autoReset: any scene change except camera jitter/history restarts accumulation.
        const scene = this.scene;
        if (this.autoReset && scene) {
            const key = `${scene.updateVersion}|${scene.camera.getChangeKey()}`;
            if (this.lastSceneKey !== undefined && key !== this.lastSceneKey) this.reset();
            this.lastSceneKey = key;
        }

        // Mirrors the native overflow handling once maxFrameCount frames were accumulated.
        const limited = this.maxFrameCount > 0 && this.precision !== AccumulatePrecision.SingleCompensated;
        if (limited && this.frameCount === this.maxFrameCount) {
            if (this.overflowMode === AccumulateOverflowMode.Stop) return; // retain the accumulated image
            if (this.overflowMode === AccumulateOverflowMode.Reset) this.reset();
            // EMA: keep blending with the constant weight 1 / (maxFrameCount + 1) below.
        }

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
        // With a frame limit the kernel runs as a moving average: weight 1/(count+1) equals the running mean until
        // the count stops at the limit, then it becomes an exponential moving average (native semantics).
        root["PerFrameCB"]["gMovingAverageMode"] = limited ? 1 : 0;
        root["gCurFrame"] = input;
        root["gOutputFrame"] = output;
        root["gLastFrameSum"] = this.lastFrameSum!;
        root["gLastFrameCorr"] = this.lastFrameCorr!;

        this.pass.execute(ctx, w, h);
        if (this.enabled && (!limited || this.frameCount < this.maxFrameCount)) this.frameCount++;
    }
}

registerRenderPass("AccumulatePass", (device, props) => new AccumulatePass(device, props));
