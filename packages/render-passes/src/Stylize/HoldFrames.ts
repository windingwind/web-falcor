/**
 * HoldFrames: animation "on twos/threes". Of every `hold` frames only the first (frame index % hold == phase) takes a new
 * input image; the others repeat it (limited-animation hold counts). The frame index is the pass's own execute counter.
 * hold = 1 is the identity.
 *
 * Inputs : src. Params : hold (1-8), phase.
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
    type PropertyValue,
    type RenderContext,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput } from "./StyleCommon.js";

const kShaderFile = "RenderPasses/Stylize/HoldFrames.cs.slang";

/** Retains a source frame for the configured hold interval and phase. */
export class HoldFrames extends RenderPass {
    private mParameters: { hold: number; phase: number } = { hold: 2, phase: 0 };
    private mCopy: ComputePass | null = null;
    private mHistory: Texture | null = null;
    private mFrame: number = 0;

    constructor(device: Device, props: Properties) {
        super(device);
        this.mParameters.hold = props.get("hold", 2);
        this.mParameters.phase = props.get("phase", 0);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.mParameters } as Record<string, PropertyValue>);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "Colour");
        addColorOutput(r, "dst", "Colour, held for `hold` frames", w, h);
        return r;
    }

    private run(ctx: RenderContext, input: Texture, output: Texture): void {
        this.mCopy ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.mCopy.getRootVar();
        root["CB"]!["gFrameDim"] = [output.width, output.height];
        root["gIn"] = input;
        root["gOut"] = output;
        this.mCopy.execute(ctx, output.width, output.height);
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src");
        const dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const hold = Math.min(Math.max(Math.round(this.mParameters.hold), 1), 8);
        const W = dst.width;
        const H = dst.height;
        if (!this.mHistory || this.mHistory.width !== W || this.mHistory.height !== H) {
            this.mHistory = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: W,
                height: H,
                format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource,
                name: "HoldFrames::history",
            });
            this.mFrame = 0;
        }
        const fresh = hold === 1 || this.mFrame === 0 || (((this.mFrame - Math.round(this.mParameters.phase)) % hold) + hold) % hold === 0;
        if (fresh) {
            this.run(ctx, src, dst);
            if (hold > 1) this.run(ctx, src, this.mHistory);
        } else {
            this.run(ctx, this.mHistory, dst);
        }
        this.mFrame++;
    }
}

registerRenderPass("HoldFrames", (device, props) => new HoldFrames(device, props), {
    desc: "Hold every new frame for N frames (animation on twos/threes)",
});
