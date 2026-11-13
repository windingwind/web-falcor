/**
 * RimLight: an additive rim term for stylized shading.
 *  - mode "fresnel": rim = (1 - N.V)^power, optionally biased towards upward-facing normals (`upBias`), the illustrative
 *    rim of e.g. Team Fortress 2 ("Illustrative Rendering in Team Fortress 2", Mitchell, Francke & Eng, NPAR 2007);
 *  - mode "depth_offset": a constant-width screen-space rim. The depth `widthPx` pixels away along the screen-space normal
 *    is compared with the pixel's own depth; a jump of more than `depthThreshold` (relative) or background marks the
 *    pixel as rim (the screen-space rim common in anime-style renderers).
 * Both are scaled by lerp(1, saturate(N.L), lightFacing), so the rim can be restricted to the lit side.
 * out = src + intensity * rim * colour * lerp(1, albedo, tint); background pixels pass through. intensity = 0 is the identity.
 *
 * Inputs : src (colour), posW, normW (GBuffer), albedo (optional, e.g. GBuffer diffuseOpacity).
 * Params : mode {fresnel|depth_offset}, colour [r,g,b] (linear), intensity, power, up [x,y,z], upBias, widthPx,
 *          depthThreshold, lightDir ([0,0,0] = first directional light), lightFacing, tint.
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, cameraData, enumIndex, keyLightDir, type PortSpec } from "./StyleCommon.js";

const kModes = ["fresnel", "depth_offset"] as const;
const kDefaults = {
    mode: "fresnel" as string,
    colour: [1, 1, 1] as number[],
    intensity: 0.5,
    power: 4,
    up: [0, 1, 0] as number[],
    upBias: 0,
    widthPx: 3,
    depthThreshold: 0.05,
    lightDir: [0, 0, 0] as number[],
    lightFacing: 0,
    tint: 0,
};

export class RimLight extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/RimLight.cs.slang";
    protected readonly outputDesc = "Colour with rim light added";
    protected override readonly needsScene = true;
    protected readonly ports: readonly PortSpec[] = [
        ["src", "Colour", false, "gSrc"],
        ["posW", "World-space position (w = 0: background)", false, "gPosW"],
        ["normW", "World-space shading normal", false, "gNormW"],
        ["albedo", "Albedo for tinting (optional)", true, "gAlbedo"],
    ];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        enumIndex("RimLight", "mode", this.p.mode, kModes);
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        const cam = cameraData(this.scene!);
        const up = this.p.up.map(Number);
        const ul = Math.hypot(up[0]!, up[1]!, up[2]!) || 1;
        cb["gMode"] = enumIndex("RimLight", "mode", this.p.mode, kModes);
        cb["gIntensity"] = this.p.intensity;
        cb["gColour"] = this.p.colour.map(Number);
        cb["gPower"] = this.p.power;
        cb["gUp"] = up.map((x) => x / ul);
        cb["gUpBias"] = this.p.upBias;
        cb["gCamPos"] = cam.pos;
        cb["gCamFwd"] = cam.fwd;
        cb["gWidthPx"] = this.p.widthPx;
        cb["gDepthThreshold"] = this.p.depthThreshold;
        cb["gLightDir"] = keyLightDir(this.scene, this.p.lightDir);
        cb["gLightFacing"] = this.p.lightFacing;
        cb["gViewRow0"] = cam.viewRows[0];
        cb["gViewRow1"] = cam.viewRows[1];
        cb["gTint"] = this.p.tint;
    }
}

registerRenderPass("RimLight", (device, props) => new RimLight(device, props), {
    desc: "Rim light: fresnel (with up bias) or constant-width screen-space depth-offset rim",
});
