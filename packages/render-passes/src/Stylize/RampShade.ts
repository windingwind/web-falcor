/**
 * RampShade: stylized diffuse shading through a colour ramp ("warped" diffuse).
 * x = 0.5 N.L + 0.5 (half-Lambert, mode "half_lambert") or saturate(N.L) ("lambert"), multiplied by an optional visibility
 * (e.g. DeferredLighting `visibility`), raised to `gamma`, then mapped through a piecewise-linear RGB ramp; `hardness`
 * sharpens every ramp segment towards a step (cel bands). shaded = albedo * (ramp(x) * overbright + ambient);
 * out = lerp(src, shaded, mix) on geometry pixels, background passes through. mix = 0 is the identity.
 * The half-Lambert + warp-ramp model follows "Illustrative Rendering in Team Fortress 2" (Mitchell, Francke & Eng, NPAR 2007).
 *
 * Inputs : src (colour, used for mix and background), albedo (GBuffer diffuseOpacity or StyleAovs flatAlbedo), normW,
 *          posW (background mask), visibility (optional, r channel in [0, 1]).
 * Params : mode {half_lambert|lambert}, ramp {tf2_warm|valorant|anime_hard|anime_soft|custom}, stops (custom ramp:
 *          [[pos, "#RRGGBB" or [r,g,b]], ...], 2-6 entries, ascending positions), gamma, hardness, overbright, ambient, mix,
 *          lightDir ([0,0,0] = first directional light of the scene).
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, enumIndex, keyLightDir, parseColour, type PortSpec } from "./StyleCommon.js";

const kModes = ["half_lambert", "lambert"] as const;
const kRampNames = ["tf2_warm", "valorant", "anime_hard", "anime_soft", "custom"] as const;
type Stop = [number, string | number[]];
/** Built-in ramps (sRGB hex; positions in [0, 1]). Written for this pass. */
export const kRamps: Record<string, Stop[]> = {
    tf2_warm: [[0, "#5a5e6e"], [0.4, "#7d5a55"], [0.55, "#c39c82"], [1, "#fff4e0"]],
    valorant: [[0, "#8a8aa2"], [0.5, "#c8c3cc"], [1, "#fffcf6"]],
    anime_hard: [[0, "#c4b8d4"], [0.5, "#c4b8d4"], [0.5001, "#ffffff"], [1, "#ffffff"]],
    anime_soft: [[0, "#bcb4cc"], [0.4, "#cfc6d8"], [0.6, "#f9f6f4"], [1, "#ffffff"]],
};
const kMaxStops = 6;

const kDefaults = {
    mode: "half_lambert" as string,
    ramp: "tf2_warm" as string,
    stops: [] as unknown as number[],
    gamma: 1,
    hardness: 0,
    overbright: 1,
    ambient: 0,
    mix: 1,
    lightDir: [0, 0, 0] as number[],
};

export class RampShade extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/RampShade.cs.slang";
    protected readonly outputDesc = "Ramp-shaded colour";
    protected readonly ports: readonly PortSpec[] = [
        ["src", "Colour (background and mix source)", false, "gSrc"],
        ["albedo", "Albedo", false, "gAlbedo"],
        ["normW", "World-space shading normal", false, "gNormW"],
        ["posW", "World-space position (w = 0: background)", false, "gPosW"],
        ["visibility", "Light visibility in r (optional)", true, "gVisibility"],
    ];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        enumIndex("RampShade", "mode", this.p.mode, kModes);
        this.rampStops();
    }

    /** The active ramp as [pos, linear rgb] pairs. */
    rampStops(): [number, [number, number, number]][] {
        const name = kRampNames[enumIndex("RampShade", "ramp", this.p.ramp, kRampNames)]!;
        const stops = (name === "custom" ? (this.p.stops as unknown as Stop[]) : kRamps[name]!);
        if (!Array.isArray(stops) || stops.length < 2 || stops.length > kMaxStops) throw new Error(`RampShade: ramp needs 2-${kMaxStops} stops`);
        const out = stops.map(([pos, c]) => [Number(pos), parseColour(c)] as [number, [number, number, number]]);
        for (let i = 1; i < out.length; i++) if (!(out[i]![0] >= out[i - 1]![0])) throw new Error("RampShade: stop positions must ascend");
        return out;
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        const stops = this.rampStops();
        cb["gMode"] = enumIndex("RampShade", "mode", this.p.mode, kModes);
        cb["gStopCount"] = stops.length;
        cb["gStops"] = Array.from({ length: kMaxStops }, (_, i) => {
            const [pos, c] = stops[Math.min(i, stops.length - 1)]!;
            return [c[0], c[1], c[2], pos];
        });
        cb["gLightDir"] = keyLightDir(this.scene, this.p.lightDir);
        cb["gGamma"] = this.p.gamma;
        cb["gHardness"] = Math.min(Math.max(this.p.hardness, 0), 1);
        cb["gOverbright"] = this.p.overbright;
        cb["gAmbient"] = this.p.ambient;
        cb["gMix"] = this.p.mix;
    }
}

registerRenderPass("RampShade", (device, props) => new RampShade(device, props), {
    desc: "Half-Lambert / Lambert diffuse through an RGB colour ramp (warped diffuse, cel bands)",
});
