/**
 * HiZBuild: hierarchical min/max depth pyramid (Hi-Z) for screen-space tracing (reflections, GI, contact shadows,
 * occlusion tests).
 *
 * Output `hiz` is an RG32Float texture with a mip chain: r = minimum, g = maximum of the input depth over each texel's
 * footprint. Mip 0 copies the selected channel of `depth`; each further mip halves the size (floor, at least 1) and
 * reduces the previous one in a separate dispatch (one bind per mip, so it stays within WebGPU's storage-texture
 * limits). For odd sizes the last texel of a row/column also covers the left-over source texel, so the min/max bounds
 * are conservative at every size. With standard depth (near = 0) the r channel is the nearest depth (classic Hi-Z),
 * with reversed depth it is the g channel.
 *
 * Input  : depth (any float texture: GBufferRaster/VBufferRaster depth, GBufferRT depth, linearZ, ...).
 * Params : channel (input channel, default 0), mipCount (0 = full chain down to 1x1).
 */

import {
    ComputePass,
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
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/Utils/HiZBuild.cs.slang";

/** Mip count of a full chain for a w x h base level. */
export function hiZMipCount(w: number, h: number): number {
    return Math.floor(Math.log2(Math.max(w, h, 1))) + 1;
}

/** Builds a min/max depth mip pyramid for screen-space tracing. */
export class HiZBuild extends RenderPass {
    private mParameters: { channel: number; mipCount: number } = { channel: 0, mipCount: 0 };
    private mCopyPass: ComputePass | null = null;
    private mReducePass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.mParameters.channel = Number(props.get("channel", this.mParameters.channel));
        this.mParameters.mipCount = Number(props.get("mipCount", this.mParameters.mipCount));
        if (![0, 1, 2, 3].includes(this.mParameters.channel))
            throw new Error(`HiZBuild: channel must be 0..3 (got ${this.mParameters.channel})`);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.mParameters });
    }

    override renderUI(_ui: UIWidgets): void {}

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        const full = hiZMipCount(w, h);
        const mips = this.mParameters.mipCount > 0 ? Math.min(this.mParameters.mipCount, full) : full;
        r.addInput("depth", "Depth (selected channel)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("hiz", "Min (r) / max (g) depth pyramid")
            .texture2D(w, h, 1, mips)
            .format(ResourceFormat.RG32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const depth = rd.getTexture("depth");
        const hiz = rd.getTexture("hiz");
        if (!depth || !hiz) return;
        this.mCopyPass ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: "copyMip0" });
        this.mReducePass ??= ComputePass.create(this.device, { path: kShaderFile, csEntry: "reduce" });
        const W = hiz.width;
        const H = hiz.height;
        {
            const root = this.mCopyPass.getRootVar();
            root["CB"]!["gDstDim"] = [W, H];
            root["CB"]!["gSrcDim"] = [W, H];
            root["CB"]!["gChannel"] = this.mParameters.channel;
            root["gDepth"] = depth;
            root["gDstMip"] = hiz.getUAV(0);
            this.mCopyPass.execute(ctx, W, H);
        }
        const root = this.mReducePass.getRootVar();
        for (let mip = 1; mip < hiz.mipCount; mip++) {
            const sw = Math.max(1, W >> (mip - 1));
            const sh = Math.max(1, H >> (mip - 1));
            const dw = Math.max(1, W >> mip);
            const dh = Math.max(1, H >> mip);
            root["CB"]!["gDstDim"] = [dw, dh];
            root["CB"]!["gSrcDim"] = [sw, sh];
            root["CB"]!["gChannel"] = this.mParameters.channel;
            root["gSrcMip"] = hiz.getSRV(mip - 1, 1);
            root["gDstMip"] = hiz.getUAV(mip);
            this.mReducePass.execute(ctx, dw, dh);
        }
    }
}

registerRenderPass("HiZBuild", (device, props) => new HiZBuild(device, props), {
    desc: "Hierarchical min/max depth pyramid (one dispatch per mip)",
});
