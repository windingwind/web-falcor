/**
 * GradientFog: distance fog whose colour follows a gradient over distance, with optional exponential height falloff.
 * Fog amount f = maxOpacity * (1 - exp(-tau)), tau = density * the integral of exp(-heightFalloff * (height - heightRef))
 * along the view ray from `start` to the surface (closed form; heightFalloff = 0 gives plain exponential distance fog).
 * Fog colour = piecewise-linear interpolation between 2-4 colour stops placed at world distances (e.g. warm near, blue
 * far for aerial perspective, or a stepped ramp for layered silhouettes). Background: skyMode "fog" blends the last stop
 * at maxOpacity, "keep" leaves the sky untouched. density = 0 or maxOpacity = 0 is the identity.
 * Height fog integral as in common engine fog models (e.g. Bevy's atmospheric fog, MIT/Apache-2.0; Wenzel, "Real-time
 * atmospheric effects in games", 2006); written for this pass.
 *
 * Inputs : src, posW. Params : stops ([[distance, "#RRGGBB" or [r,g,b]], ...], 2-4, ascending), density (per world unit),
 *          start, heightFalloff, heightRef, up [x,y,z], skyMode {fog|keep}, maxOpacity.
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, cameraData, enumIndex, parseColour, type PortSpec } from "./StyleCommon.js";

const kSkyModes = ["fog", "keep"] as const;
const kMaxStops = 4;
type Stop = [number, string | number[]];

const kDefaults = {
    stops: [[0, "#e6c9a8"], [60, "#8fa6c4"]] as unknown as number[],
    density: 0.02,
    start: 0,
    heightFalloff: 0,
    heightRef: 0,
    up: [0, 1, 0] as number[],
    skyMode: "fog" as string,
    maxOpacity: 1,
};

export class GradientFog extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/GradientFog.cs.slang";
    protected readonly outputDesc = "Fogged colour";
    protected override readonly needsScene = true;
    protected readonly ports: readonly PortSpec[] = [
        ["src", "Colour", false, "gSrc"],
        ["posW", "World-space position (w = 0: background)", false, "gPosW"],
    ];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        enumIndex("GradientFog", "skyMode", this.p.skyMode, kSkyModes);
        this.fogStops();
    }

    fogStops(): [number, [number, number, number]][] {
        const s = this.p.stops as unknown as Stop[];
        if (!Array.isArray(s) || s.length < 1 || s.length > kMaxStops) throw new Error(`GradientFog: 1-${kMaxStops} stops`);
        const out = s.map(([d, c]) => [Number(d), parseColour(c)] as [number, [number, number, number]]);
        for (let i = 1; i < out.length; i++) if (!(out[i]![0] >= out[i - 1]![0])) throw new Error("GradientFog: stop distances must ascend");
        return out;
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        const stops = this.fogStops();
        const up = this.p.up.map(Number);
        const ul = Math.hypot(up[0]!, up[1]!, up[2]!) || 1;
        cb["gStopCount"] = stops.length;
        cb["gSkyMode"] = enumIndex("GradientFog", "skyMode", this.p.skyMode, kSkyModes);
        cb["gStops"] = Array.from({ length: kMaxStops }, (_, i) => {
            const [d, c] = stops[Math.min(i, stops.length - 1)]!;
            return [c[0], c[1], c[2], d];
        });
        cb["gCamPos"] = cameraData(this.scene!).pos;
        cb["gDensity"] = this.p.density;
        cb["gUp"] = up.map((x) => x / ul);
        cb["gStart"] = this.p.start;
        cb["gHeightFalloff"] = this.p.heightFalloff;
        cb["gHeightRef"] = this.p.heightRef;
        cb["gMaxOpacity"] = this.p.maxOpacity;
    }
}

registerRenderPass("GradientFog", (device, props) => new GradientFog(device, props), {
    desc: "Distance fog with a colour gradient over distance and optional exponential height falloff",
});
