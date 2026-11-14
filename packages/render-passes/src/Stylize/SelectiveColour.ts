/**
 * SelectiveColour: keep one hue band in colour, desaturate the rest ("colour key" / sin-city red).
 * The mask is a soft band around `keepHue` (OKLCh hue angle in degrees, of `src` or, with source "albedo", of the
 * lighting-independent albedo), gated by a minimum chroma so greys never count as the key colour.
 * Saturation is scaled by satKeep inside the band and by satRest outside (around Rec. 709 luma, in linear RGB).
 * satKeep = satRest = 1 is the identity. OKLCh hues of pure sRGB primaries: red ~29, yellow ~110, green ~142, cyan ~195,
 * blue ~264, magenta ~328 degrees.
 *
 * Inputs : src, albedo (optional, for source "albedo").
 * Params : keepHue (deg), hueWidth (deg), softness (deg), minChroma (OKLab chroma), satKeep, satRest, source {src|albedo}.
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, enumIndex, type PortSpec } from "./StyleCommon.js";

const kSources = ["src", "albedo"] as const;
const kDefaults = {
    keepHue: 29,
    hueWidth: 20,
    softness: 15,
    minChroma: 0.05,
    satKeep: 1,
    satRest: 0,
    source: "src" as string,
};

export class SelectiveColour extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/SelectiveColour.cs.slang";
    protected readonly outputDesc = "Selectively desaturated colour";
    protected readonly ports: readonly PortSpec[] = [
        ["src", "Colour (linear)", false, "gSrc"],
        ["albedo", "Albedo for the hue mask (optional)", true, "gAlbedo"],
    ];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        enumIndex("SelectiveColour", "source", this.p.source, kSources);
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        cb["gKeepHue"] = this.p.keepHue;
        cb["gHueWidth"] = this.p.hueWidth;
        cb["gSoftness"] = this.p.softness;
        cb["gMinChroma"] = this.p.minChroma;
        cb["gSatKeep"] = this.p.satKeep;
        cb["gSatRest"] = this.p.satRest;
        cb["gSource"] = enumIndex("SelectiveColour", "source", this.p.source, kSources);
    }
}

registerRenderPass("SelectiveColour", (device, props) => new SelectiveColour(device, props), {
    desc: "Keep one hue band (OKLCh) in colour and desaturate the rest",
});
