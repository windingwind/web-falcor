/**
 * MotionStats: one-pass statistics of the screen-space motion vectors, for passes that react to motion (speed lines,
 * impact frames, datamosh) and for clip metadata. Deterministic: a fixed-order reduction (bitwise reproducible) plus
 * exact integer histograms.
 *
 * Motion of a pixel m = -mvec * frameDim, in pixels per frame (mvec = previous - current position in UV units, as the
 * G-buffer passes write it). Angles are atan2(m.y, m.x) in pixel coordinates (x right, y down), in [-pi, pi).
 *
 * Inputs : mvec (G-buffer motion vectors), mask (optional; a pixel counts when mask.w != 0, e.g. GBuffer posW = geometry only).
 * Outputs: stats (4 x 1 RGBA32Float)
 *            texel 0 = (mean m.x, mean m.y, mean |m|, max |m|)
 *            texel 1 = (rms |m|, moving share (|m| > minMotion), dominant direction angle, share of moving pixels in it)
 *            texel 2 = (angle of the mean vector (0 if it is zero), pixel count, |mean m|, moving pixel count)
 *            texel 3 = (magRange, minMotion, magBins, dirBins)
 *          histogram (max(magBins, dirBins) x 2 RGBA32Float): row 0 = magnitude bins over [0, magRange) (last bin takes
 *            overflow), row 1 = direction bins over [-pi, pi) of the moving pixels; .x = count / pixel count, .y = count.
 * Params : magBins (1..64, 16), dirBins (1..64, 16), magRange (pixels, 32), minMotion (pixels, 0.5).
 * Statistics only: the pass does not change any image.
 */

import {
    Buffer,
    ComputePass,
    FieldFlags,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/MotionStats/MotionStats.cs.slang";
/** Matches the maximum histogram bin count in MotionStats.cs.slang. */
const kMaxBins = 64;

/** Reduces motion vectors into magnitude, direction and summary statistics. */
export class MotionStats extends RenderPass {
    private mParameters: { magBins: number; dirBins: number; magRange: number; minMotion: number } = {
        magBins: 16,
        dirBins: 16,
        magRange: 32,
        minMotion: 0.5,
    };
    private mClearPass: ComputePass | null = null;
    private mReducePass: ComputePass | null = null;
    private mFinalPass: ComputePass | null = null;
    private mPartials: Buffer | null = null;
    private mHistogram: Buffer | null = null;
    private mDummyMask: Texture | null = null;
    private mDummyHist: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.mParameters) as (keyof typeof this.mParameters)[])
            this.mParameters[k] = Number(props.get(k, this.mParameters[k]));
        this.mParameters.magBins = MotionStats.bins(this.mParameters.magBins);
        this.mParameters.dirBins = MotionStats.bins(this.mParameters.dirBins);
        if (!(this.mParameters.magRange > 0)) throw new Error(`MotionStats: magRange must be > 0 (got ${this.mParameters.magRange})`);
        if (!(this.mParameters.minMotion >= 0)) throw new Error(`MotionStats: minMotion must be >= 0 (got ${this.mParameters.minMotion})`);
    }

    private static bins(v: number): number {
        const n = Math.round(v);
        if (!(n >= 1 && n <= kMaxBins)) throw new Error(`MotionStats: bin counts must be in 1..${kMaxBins} (got ${v})`);
        return n;
    }

    override getProperties(): Properties {
        return new Properties({ ...this.mParameters });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Magnitude range (px)", this.mParameters.magRange, 1, 256, 1, (v) => (this.mParameters.magRange = v));
        ui.slider("Min motion (px)", this.mParameters.minMotion, 0, 8, 0.05, (v) => (this.mParameters.minMotion = v));
        ui.text(`${this.mParameters.magBins} magnitude bins, ${this.mParameters.dirBins} direction bins`);
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("mvec", "Screen-space motion vectors (UV units, previous - current)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("mask", "Optional mask: pixel counted when w != 0 (e.g. GBuffer posW)")
            .bindFlags(ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        r.addOutput("stats", "Motion statistics (4 x 1, see MotionStats.ts)")
            .texture2D(4, 1)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        r.addOutput("histogram", "Magnitude (row 0) and direction (row 1) histograms")
            .texture2D(Math.max(this.mParameters.magBins, this.mParameters.dirBins), 2)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        return r;
    }

    private buffer(old: Buffer | null, elements: number, structSize: number, name: string): Buffer {
        if (old && old.size >= elements * structSize) return old;
        return new Buffer(this.device, {
            size: elements * structSize,
            structSize,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name,
        });
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const mvec = rd.getTexture("mvec");
        const stats = rd.getTexture("stats");
        if (!mvec || !stats) return;
        const mask = rd.getTexture("mask");
        let histOut = rd.getTexture("histogram");
        const W = mvec.width;
        const H = mvec.height;
        const gx = Math.ceil(W / 16);
        const gy = Math.ceil(H / 16);
        this.mPartials = this.buffer(this.mPartials, 2 * gx * gy, 16, "MotionStats::partials");
        this.mHistogram = this.buffer(this.mHistogram, 2 * kMaxBins, 4, "MotionStats::hist");
        const placeholder = (w: number, h: number) =>
            this.device.createTexture2D(
                w,
                h,
                ResourceFormat.RGBA32Float,
                1,
                1,
                undefined,
                ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            );
        this.mDummyMask ??= placeholder(1, 1);
        if (!histOut) histOut = this.mDummyHist ??= placeholder(kMaxBins, 2);
        if (!this.mClearPass) {
            this.mClearPass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "clearMain" });
            this.mReducePass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "reduceMain" });
            this.mFinalPass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "finalMain" });
        }
        const passes = [this.mClearPass, this.mReducePass!, this.mFinalPass!];
        for (const pass of passes) {
            const root = pass.getRootVar();
            const cb = root["CB"]!;
            cb["gFrameDim"] = [W, H];
            cb["gGroupsX"] = gx;
            cb["gNumGroups"] = gx * gy;
            cb["gMagBins"] = this.mParameters.magBins;
            cb["gDirBins"] = this.mParameters.dirBins;
            cb["gMagRange"] = this.mParameters.magRange;
            cb["gMinMotion"] = this.mParameters.minMotion;
            cb["gUseMask"] = mask ? 1 : 0;
            root["gMvec"] = mvec;
            root["gMask"] = mask ?? this.mDummyMask;
            root["gPartials"] = this.mPartials;
            root["gHist"] = this.mHistogram;
            root["gStats"] = stats;
            root["gHistOut"] = histOut;
        }
        this.mClearPass.execute(ctx, 2 * kMaxBins);
        this.mReducePass!.execute(ctx, gx * 16, gy * 16);
        this.mFinalPass!.execute(ctx, 256);
    }
}

registerRenderPass("MotionStats", (device, props) => new MotionStats(device, props), {
    desc: "Motion-vector statistics: mean / max / rms motion, moving share, dominant direction and histograms",
});
