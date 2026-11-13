/**
 * GradientMap: map image brightness onto a colour gradient (duotone / tritone / "gradient map" grading).
 * t = sRGB-encoded luma of src in [0, 1]; colour = piecewise-linear interpolation through 2-6 stops [[pos, colour], ...];
 * out = lerp(src, colour, strength). strength = 0 is the identity. Presets: duotone_navy, tritone_sunset, neon, sepia.
 *
 * Inputs : src. Params : preset {duotone_navy|tritone_sunset|neon|sepia|custom}, stops (custom), strength.
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, enumIndex, parseColour, type PortSpec } from "./StyleCommon.js";

type Stop = [number, string | number[]];
export const kGradientMaps: Record<string, Stop[]> = {
    duotone_navy: [[0, "#141a2e"], [1, "#f4ead2"]],
    tritone_sunset: [[0, "#2a1440"], [0.5, "#d0485a"], [1, "#ffd88a"]],
    neon: [[0, "#05010f"], [0.35, "#3a0ca3"], [0.7, "#f72585"], [1, "#b8f2ff"]],
    sepia: [[0, "#1e130a"], [0.5, "#8c6a46"], [1, "#f7ecd6"]],
};
const kNames = [...Object.keys(kGradientMaps), "custom"] as const;
const kMaxStops = 6;
const kDefaults = { preset: "duotone_navy" as string, stops: [] as unknown as number[], strength: 1 };

export class GradientMap extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/GradientMap.cs.slang";
    protected readonly outputDesc = "Gradient-mapped colour";
    protected readonly ports: readonly PortSpec[] = [["src", "Colour (linear)", false, "gSrc"]];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        this.mapStops();
    }

    mapStops(): [number, [number, number, number]][] {
        const name = kNames[enumIndex("GradientMap", "preset", this.p.preset, kNames)]!;
        const stops = name === "custom" ? (this.p.stops as unknown as Stop[]) : kGradientMaps[name]!;
        if (!Array.isArray(stops) || stops.length < 2 || stops.length > kMaxStops) throw new Error(`GradientMap: 2-${kMaxStops} stops`);
        const out = stops.map(([p, c]) => [Number(p), parseColour(c)] as [number, [number, number, number]]);
        for (let i = 1; i < out.length; i++) if (!(out[i]![0] >= out[i - 1]![0])) throw new Error("GradientMap: stop positions must ascend");
        return out;
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        const s = this.mapStops();
        cb["gStopCount"] = s.length;
        cb["gStrength"] = this.p.strength;
        cb["gStops"] = Array.from({ length: kMaxStops }, (_, i) => { const [p, c] = s[Math.min(i, s.length - 1)]!; return [c[0], c[1], c[2], p]; });
    }
}

registerRenderPass("GradientMap", (device, props) => new GradientMap(device, props), { desc: "Map image luma onto a 2-6 stop colour gradient" });
