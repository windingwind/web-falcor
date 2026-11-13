/**
 * CRT: cathode-ray-tube display simulation. Barrel curvature; Gaussian scanline beams over `sourceLines` virtual lines,
 * wider for dark and narrower for bright pixels (beamMax / beamMin, in line units, energy-normalised); a phosphor mask at
 * output resolution (aperture grille stripes, staggered slot mask or diagonal dot triads); halation glow; vignette;
 * brightness compensation. out = lerp(src, crt, strength); strength = 0 is the identity.
 * Model after the public-domain crt-lottes shader family (ideas; code written for this pass).
 *
 * Inputs : src. Params : sourceLines, beamMin, beamMax, mask {none|aperture|slot|dot}, maskStrength, curvature, halation,
 *          vignette, brightness, strength.
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, enumIndex, type PortSpec } from "./StyleCommon.js";

const kMasks = ["none", "aperture", "slot", "dot"] as const;
const kDefaults = { sourceLines: 240, beamMin: 0.3, beamMax: 0.6, mask: "aperture" as string, maskStrength: 0.5, curvature: 0.1,
    halation: 0.1, vignette: 0.5, brightness: 1.3, strength: 1 };

export class CRT extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/CRT.cs.slang";
    protected readonly outputDesc = "CRT-filtered colour";
    protected readonly ports: readonly PortSpec[] = [["src", "Colour (linear)", false, "gSrc"]];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        enumIndex("CRT", "mask", this.p.mask, kMasks);
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        cb["gSourceLines"] = Math.max(this.p.sourceLines, 8);
        cb["gBeamMin"] = Math.max(this.p.beamMin, 0.05);
        cb["gBeamMax"] = Math.max(this.p.beamMax, 0.05);
        cb["gMask"] = enumIndex("CRT", "mask", this.p.mask, kMasks);
        cb["gMaskStrength"] = this.p.maskStrength;
        cb["gCurvature"] = this.p.curvature;
        cb["gHalation"] = this.p.halation;
        cb["gVignette"] = this.p.vignette;
        cb["gBrightness"] = this.p.brightness;
        cb["gStrength"] = this.p.strength;
    }
}

registerRenderPass("CRT", (device, props) => new CRT(device, props), { desc: "CRT display: curvature, scanline beams, phosphor mask, halation" });
