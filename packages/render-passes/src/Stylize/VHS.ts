/**
 * VHS: analogue tape look. In YIQ, luma is sampled with a random per-line horizontal jitter (lineJitter px) and an extra
 * shift and noise inside a tracking band that rolls down the frame (tracking, trackingSpeed in frame heights per frame);
 * chroma is low-passed horizontally (chromaBlur px box) and shifted (chromaOffset px); tape noise on luma.
 * Time = the pass's frame counter (+ frameOffset), so the effect animates over a clip and is deterministic per frame index.
 * out = lerp(src, vhs, strength); strength = 0 is the identity.
 *
 * Inputs : src. Params : chromaBlur, chromaOffset, lineJitter, tracking, trackingSpeed, noise, seed, frameOffset, strength.
 */

import { Properties, registerRenderPass, type Device, type RenderContext, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, type PortSpec } from "./StyleCommon.js";

const kDefaults = { chromaBlur: 4, chromaOffset: 2, lineJitter: 1, tracking: 0.5, trackingSpeed: 0.02, noise: 0.05, seed: 0, frameOffset: 0, strength: 1 };

export class VHS extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/VHS.cs.slang";
    protected readonly outputDesc = "VHS-filtered colour";
    protected readonly ports: readonly PortSpec[] = [["src", "Colour (linear)", false, "gSrc"]];
    private frame = 0;

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        cb["gFrame"] = (this.frame + Math.round(this.p.frameOffset)) >>> 0;
        cb["gSeed"] = Math.round(this.p.seed) >>> 0;
        cb["gChromaBlur"] = this.p.chromaBlur;
        cb["gChromaOffset"] = this.p.chromaOffset;
        cb["gLineJitter"] = this.p.lineJitter;
        cb["gTracking"] = this.p.tracking;
        cb["gTrackingSpeed"] = this.p.trackingSpeed;
        cb["gNoise"] = this.p.noise;
        cb["gStrength"] = this.p.strength;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        super.execute(ctx, rd);
        this.frame++;
    }
}

registerRenderPass("VHS", (device, props) => new VHS(device, props), { desc: "VHS tape: chroma bleed/offset, line jitter, rolling tracking band, noise" });
