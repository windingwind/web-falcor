/**
 * LightShafts: screen-space crepuscular rays ("god rays") by radial light scattering towards the sun's screen position
 * (Mitchell, "Volumetric Light Scattering as a Post-Process", GPU Gems 3, ch. 13, 2007).
 * The occlusion mask is the sky (posW.w = 0) above a luminance threshold; geometry occludes. The sun position comes from the
 * first directional light of the scene (projected through the camera), or from `lightDir`, or from `sunUV` if set
 * (>= 0, screen uv with y down). Rays fade out as the sun moves behind the camera. exposure = 0 is the identity.
 *
 * Inputs : src, posW. Params : samples (8-128), density (0-1, fraction of the way to the sun), decay, weight, exposure,
 *          threshold (sky luminance), tint [r,g,b], lightDir ([0,0,0] = scene light), sunUV ([-1,-1] = from the light).
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, cameraData, keyLightDir, type PortSpec } from "./StyleCommon.js";

const kDefaults = {
    samples: 64,
    density: 0.8,
    decay: 0.97,
    weight: 0.04,
    exposure: 1,
    threshold: 0.5,
    tint: [1, 0.95, 0.85] as number[],
    lightDir: [0, 0, 0] as number[],
    sunUV: [-1, -1] as number[],
};

/** Screen uv (y down) of a direction, and how much it faces the camera (0 behind .. 1 straight ahead). */
export function projectDirection(viewProjRows: number[][], camPos: number[], camFwd: number[], dir: number[]): { uv: [number, number]; facing: number } {
    const far = 1e4;
    const w4 = [camPos[0]! + dir[0]! * far, camPos[1]! + dir[1]! * far, camPos[2]! + dir[2]! * far, 1];
    const c = viewProjRows.map((r) => r[0]! * w4[0]! + r[1]! * w4[1]! + r[2]! * w4[2]! + r[3]! * w4[3]!);
    const facing = dir[0]! * camFwd[0]! + dir[1]! * camFwd[1]! + dir[2]! * camFwd[2]!;
    if (c[3]! <= 1e-6) return { uv: [0.5, 0.5], facing: 0 };
    const ndc = [c[0]! / c[3]!, c[1]! / c[3]!];
    return { uv: [0.5 + 0.5 * ndc[0]!, 0.5 - 0.5 * ndc[1]!], facing: Math.min(Math.max(facing * 4, 0), 1) };
}

export class LightShafts extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/LightShafts.cs.slang";
    protected readonly outputDesc = "Colour with light shafts added";
    protected override readonly needsScene = true;
    protected readonly ports: readonly PortSpec[] = [
        ["src", "Colour", false, "gSrc"],
        ["posW", "World-space position (w = 0: sky)", false, "gPosW"],
    ];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        const cam = cameraData(this.scene!);
        let uv: [number, number], facing: number;
        const manual = this.p.sunUV.map(Number);
        if (manual[0]! >= 0 && manual[1]! >= 0) { uv = [manual[0]!, manual[1]!]; facing = 1; }
        else ({ uv, facing } = projectDirection(cam.viewProjRows, cam.pos, cam.fwd, keyLightDir(this.scene, this.p.lightDir)));
        cb["gSunUV"] = uv;
        cb["gTint"] = this.p.tint.map(Number);
        cb["gSunFacing"] = facing;
        cb["gSamples"] = Math.min(Math.max(Math.round(this.p.samples), 1), 256);
        cb["gDensity"] = this.p.density;
        cb["gDecay"] = this.p.decay;
        cb["gWeight"] = this.p.weight;
        cb["gExposure"] = this.p.exposure;
        cb["gThreshold"] = this.p.threshold;
    }
}

registerRenderPass("LightShafts", (device, props) => new LightShafts(device, props), {
    desc: "Screen-space radial light shafts towards the sun (occluded by geometry)",
});
