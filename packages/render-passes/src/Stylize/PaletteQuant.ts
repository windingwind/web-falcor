/**
 * PaletteQuant: map every pixel to the nearest colour of a fixed palette (perceptual nearest colour in OKLab, or in linear
 * RGB), with an optional 4x4 ordered-dither offset; or quantise each sRGB-encoded channel to 2^bits levels (rgb555).
 * out = lerp(src, quantised, mix); mix = 0 is the identity. Input and output are linear RGB; palettes are given in sRGB hex.
 * OKLab: B. Ottosson, "A perceptual color space for image processing" (2020), matrices re-typed from the article.
 *
 * Palettes: 1bit_mac (black/white), obra_1bit (dark olive / pale cyan two-tone), gameboy4 (DMG greens), pico8_16
 * (PICO-8 16-colour palette), cga4 (CGA mode 4, palette 1 high intensity), ega16 (default EGA/VGA 16 colours),
 * sunset6, rgb555 (5 bits per channel, no palette), custom (`colours`, 2-16 "#RRGGBB" entries).
 *
 * Inputs : src. Params : palette, colours (custom), space {oklab|rgb}, dither (0-1), mix.
 */

import { Properties, registerRenderPass, type Device, type RenderData, type ShaderVar } from "@web-falcor/falcor";
import { StylePostPass, enumIndex, parseColour, type PortSpec } from "./StyleCommon.js";

export const kPalettes: Record<string, string[]> = {
    "1bit_mac": ["#000000", "#ffffff"],
    obra_1bit: ["#333319", "#e5ffff"],
    gameboy4: ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    pico8_16: ["#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f", "#c2c3c7", "#fff1e8",
        "#ff004d", "#ffa300", "#ffec27", "#00e436", "#29adff", "#83769c", "#ff77a8", "#ffccaa"],
    cga4: ["#000000", "#55ffff", "#ff55ff", "#ffffff"],
    ega16: ["#000000", "#0000aa", "#00aa00", "#00aaaa", "#aa0000", "#aa00aa", "#aa5500", "#aaaaaa",
        "#555555", "#5555ff", "#55ff55", "#55ffff", "#ff5555", "#ff55ff", "#ffff55", "#ffffff"],
    sunset6: ["#2b1b3d", "#5c2a5a", "#a23e5c", "#e0645a", "#f5a15d", "#fce3a6"],
};
const kPaletteNames = [...Object.keys(kPalettes), "rgb555", "custom"] as const;
const kSpaces = ["oklab", "rgb"] as const;
const kMaxColours = 16;

export function linearSrgbToOklab(c: readonly number[]): [number, number, number] {
    const l = 0.4122214708 * c[0]! + 0.5363325363 * c[1]! + 0.0514459929 * c[2]!;
    const m = 0.2119034982 * c[0]! + 0.6806995451 * c[1]! + 0.1073969566 * c[2]!;
    const s = 0.0883024619 * c[0]! + 0.2817188376 * c[1]! + 0.6299787005 * c[2]!;
    const l_ = Math.cbrt(l), m_ = Math.cbrt(m), s_ = Math.cbrt(s);
    return [0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_];
}

const kDefaults = {
    palette: "pico8_16" as string,
    colours: [] as string[],
    space: "oklab" as string,
    dither: 0,
    mix: 1,
};

export class PaletteQuant extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/PaletteQuant.cs.slang";
    protected readonly outputDesc = "Palette-quantised colour";
    protected readonly ports: readonly PortSpec[] = [["src", "Colour (linear)", false, "gSrc"]];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        this.paletteLinear();
        enumIndex("PaletteQuant", "space", this.p.space, kSpaces);
    }

    /** Active palette in linear RGB (empty for rgb555). */
    paletteLinear(): [number, number, number][] {
        const name = kPaletteNames[enumIndex("PaletteQuant", "palette", this.p.palette, kPaletteNames)]!;
        if (name === "rgb555") return [];
        const hex = name === "custom" ? this.p.colours : kPalettes[name]!;
        if (!Array.isArray(hex) || hex.length < 2 || hex.length > kMaxColours) throw new Error(`PaletteQuant: palette needs 2-${kMaxColours} colours`);
        return hex.map(parseColour);
    }

    protected setConstants(cb: ShaderVar, _W: number, _H: number, _rd: RenderData): void {
        const pal = this.paletteLinear();
        const space = enumIndex("PaletteQuant", "space", this.p.space, kSpaces);
        const cmp = pal.map((c) => (space === 0 ? linearSrgbToOklab(c) : c));
        let spacing = 1 / 31;
        if (pal.length) {
            // mean nearest-neighbour distance (comparison space) = the dither amplitude unit
            let sum = 0;
            for (let i = 0; i < cmp.length; i++) {
                let best = Infinity;
                for (let j = 0; j < cmp.length; j++) if (i !== j) best = Math.min(best, Math.hypot(cmp[i]![0] - cmp[j]![0], cmp[i]![1] - cmp[j]![1], cmp[i]![2] - cmp[j]![2]));
                sum += best;
            }
            spacing = sum / cmp.length;
        }
        const pad = <T>(a: T[], fill: T) => Array.from({ length: kMaxColours }, (_, i) => a[i] ?? fill);
        cb["gCount"] = pal.length;
        cb["gSpace"] = space;
        cb["gPalette"] = pad(pal.map((c) => [...c, 1]), [0, 0, 0, 1]);
        cb["gPaletteLab"] = pad(cmp.map((c) => [...c, 0]), [0, 0, 0, 0]);
        cb["gDither"] = this.p.dither;
        cb["gSpacing"] = spacing;
        cb["gMix"] = this.p.mix;
        cb["gBits"] = pal.length ? 0 : 5;
    }
}

registerRenderPass("PaletteQuant", (device, props) => new PaletteQuant(device, props), {
    desc: "Nearest-colour palette quantisation (OKLab / RGB, presets incl. 1-bit, Game Boy, PICO-8, CGA, EGA, RGB555) with optional ordered dither",
});
