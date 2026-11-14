/**
 * ShadowTint: replace shadowed areas by a hue-shifted, darker version of the albedo instead of a plain darkening
 * (painted / anime-style shadow colours, optionally cooler or warmer with distance).
 * Shadow mask from s = saturate(N.L) times an optional visibility (e.g. DeferredLighting `visibility`):
 * m = 1 - smoothstep(threshold - softness, threshold + softness, s). Shadow colour = albedo in OKLCh with the hue rotated
 * by hueShift (degrees), lightness scaled by (1 - valueDrop), chroma by satKeep, multiplied by a tint that blends from
 * nearTint to farTint over view depth [near, far], times brightness. out = lerp(src, shadow colour, m * strength);
 * strength = 0 is the identity. OKLab: B. Ottosson (2020), matrices re-typed from the article.
 *
 * Inputs : src, albedo, normW, posW, visibility (optional). Params : threshold, softness, hueShift, valueDrop, satKeep,
 *          nearTint, farTint, near, far, brightness, strength, lightDir ([0,0,0] = first directional light).
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, cameraData, keyLightDir, parseColour, type PortSpec } from "./StyleCommon.js";

const kDefaults = {
    threshold: 0.2, softness: 0.05, hueShift: -15, valueDrop: 0.3, satKeep: 1.1,
    nearTint: [0.85, 0.9, 1.1] as number[], farTint: [0.8, 1.0, 0.95] as number[], near: 0, far: 0,
    brightness: 1, strength: 1, lightDir: [0, 0, 0] as number[],
};

export class ShadowTint extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/ShadowTint.cs.slang";
    protected readonly outputDesc = "Colour with tinted shadows";
    protected override readonly needsScene = true;
    protected readonly ports: readonly PortSpec[] = [
        ["src", "Colour", false, "gSrc"],
        ["albedo", "Albedo", false, "gAlbedo"],
        ["normW", "World-space shading normal", false, "gNormW"],
        ["posW", "World-space position (w = 0: background)", false, "gPosW"],
        ["visibility", "Light visibility in r (optional)", true, "gVisibility"],
    ];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        const cam = cameraData(this.scene!);
        cb["gThreshold"] = this.p.threshold;
        cb["gSoftness"] = this.p.softness;
        cb["gLightDir"] = keyLightDir(this.scene, this.p.lightDir);
        cb["gHueShift"] = (this.p.hueShift * Math.PI) / 180;
        cb["gNearTint"] = parseColour(this.p.nearTint);
        cb["gValueDrop"] = this.p.valueDrop;
        cb["gFarTint"] = parseColour(this.p.farTint);
        cb["gSatKeep"] = this.p.satKeep;
        cb["gCamPos"] = cam.pos;
        cb["gNear"] = this.p.near;
        cb["gCamFwd"] = cam.fwd;
        cb["gFar"] = this.p.far;
        cb["gBrightness"] = this.p.brightness;
        cb["gStrength"] = this.p.strength;
    }
}

registerRenderPass("ShadowTint", (device, props) => new ShadowTint(device, props), {
    desc: "Shadows as hue-shifted, darker albedo with a near/far tint (painted shadow colours)",
});
