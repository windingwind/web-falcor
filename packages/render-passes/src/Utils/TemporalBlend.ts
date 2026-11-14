/**
 * TemporalBlend: exponential moving average of a colour input over frames, with optional motion-vector reprojection.
 *
 * Example user of the history helpers (History.ts): the previous output lives in a PingPongTextures pair driven by a
 * HistoryTracker, so the pass starts from an empty history on a fresh context, after resetHistory(), after
 * Camera.resetHistory() and after a primed camera cut; on such a frame the output is the input.
 *
 * Inputs : src (colour), mvec (optional; GBufferRaster/GBufferRT mvec, previous minus current position in UV units).
 * Output : dst (RGBA32Float).
 * Params : alpha (history weight in [0, 1); 0 = pass-through, bit for bit).
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    FieldFlags,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { HistoryTracker, PingPongTextures } from "./History.js";

const kShaderFile = "RenderPasses/Utils/TemporalBlend.cs.slang";

export class TemporalBlend extends RenderPass {
    private p = { alpha: 0.8 };
    private pass: ComputePass | null = null;
    private readonly history = new HistoryTracker();
    private readonly pp: PingPongTextures;
    private dummy: import("@web-falcor/falcor").Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.p.alpha = Number(props.get("alpha", this.p.alpha));
        this.pp = new PingPongTextures(device, ResourceFormat.RGBA32Float, "TemporalBlend::history");
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("History weight", this.p.alpha, 0, 0.99, 0.01, (v) => (this.p.alpha = v));
    }

    /** Drops the history: the next frame outputs the input. */
    resetHistory(): void {
        this.history.reset();
    }

    /** Frames since the last history reset. */
    get frameIndex(): number {
        return this.history.frameIndex;
    }

    /** The ping-pong pair (tests). */
    get historyTextures() {
        return this.pp.textures;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("src", "Colour").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("mvec", "Motion vectors (UV, previous minus current)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("dst", "Blended colour").texture2D(w, h).format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const W = dst.width, H = dst.height;
        const frame = this.history.begin(this.scene, `${W}x${H}`);
        const { read, write } = this.pp.get(ctx, W, H, frame);
        const mvec = rd.getTexture("mvec");
        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        this.dummy ??= this.device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(4), ResourceBindFlags.ShaderResource);
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        cb["gFrameDim"] = [W, H];
        cb["gAlpha"] = Math.min(Math.max(this.p.alpha, 0), 1);
        cb["gHistoryValid"] = frame.frameIndex > 0 ? 1 : 0;
        cb["gHasMvec"] = mvec ? 1 : 0;
        root["gSrc"] = src;
        root["gMvec"] = mvec ?? this.dummy;
        root["gHistory"] = read;
        root["gDst"] = dst;
        root["gHistoryOut"] = write;
        this.pass.execute(ctx, W, H);
    }
}

registerRenderPass("TemporalBlend", (device, props) => new TemporalBlend(device, props), {
    desc: "Exponential blend with the reprojected previous output; history resets on fresh contexts and camera cuts",
});
