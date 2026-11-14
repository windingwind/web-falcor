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

export class HoldFrames extends RenderPass {
    private p = { hold: 2, phase: 0 };
    private copy: ComputePass | null = null;
    private history: Texture | null = null;
    private frame = 0;

    constructor(device: Device, props: Properties) {
        super(device);
        this.p.hold = props.get("hold", 2);
        this.p.phase = props.get("phase", 0);
    }

    override getProperties(): Properties { return new Properties({ ...this.p } as Record<string, PropertyValue>); }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "Colour");
        addColorOutput(r, "dst", "Colour, held for `hold` frames", w, h);
        return r;
    }

    private run(ctx: RenderContext, input: Texture, output: Texture): void {
        this.copy ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.copy.getRootVar();
        root["CB"]!["gFrameDim"] = [output.width, output.height];
        root["gIn"] = input;
        root["gOut"] = output;
        this.copy.execute(ctx, output.width, output.height);
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const hold = Math.min(Math.max(Math.round(this.p.hold), 1), 8);
        const W = dst.width, H = dst.height;
        if (!this.history || this.history.width !== W || this.history.height !== H) {
            this.history = new Texture(this.device, { type: ResourceType.Texture2D, width: W, height: H, format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource, name: "HoldFrames::history" });
            this.frame = 0;
        }
        const fresh = hold === 1 || this.frame === 0 || (((this.frame - Math.round(this.p.phase)) % hold) + hold) % hold === 0;
        if (fresh) {
            this.run(ctx, src, dst);
            if (hold > 1) this.run(ctx, src, this.history);
        } else {
            this.run(ctx, this.history, dst);
        }
        this.frame++;
    }
}

registerRenderPass("HoldFrames", (device, props) => new HoldFrames(device, props), { desc: "Hold every new frame for N frames (animation on twos/threes)" });
