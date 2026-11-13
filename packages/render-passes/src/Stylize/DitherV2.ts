/**
 * DitherV2: ordered dithering with selectable threshold pattern and pattern space.
 *  - pattern: bayer2 | bayer4 | bayer8 | blue_noise (64x64 void-and-cluster rank map generated at start-up, Ulichney 1993)
 *    | ps1 (the PlayStation GPU's 4x4 offset matrix on 8-bit values, then 5-bit truncation; `levels` is ignored)
 *    | ign (interleaved gradient noise, Jimenez 2014);
 *  - space: screen (pixels / scale) | view_sphere (yaw/pitch of the view ray, `scale` pixels per cell at the image
 *    centre: the pattern stays put on the world under camera rotation, as in Return of the Obra Dinn) | world (dominant-axis
 *    projection of posW, `worldScale` cells per unit);
 *  - quantisation of the sRGB-encoded colour to `levels` levels per channel; `strength` scales the threshold.
 * out = lerp(src, dithered, mix); mix = 0 is the identity.
 *
 * Inputs : src, posW (optional; needed by view_sphere/world), normW (optional; world axis choice).
 * Params : pattern, space, levels (2-256), scale (px), worldScale, strength (0-1), mix.
 */

import {
    Properties,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type Device,
    type RenderContext,
    type RenderData,
    type ShaderVar,
    type Texture,
} from "@web-falcor/falcor";
import { StylePostPass, cameraData, enumIndex, type PortSpec } from "./StyleCommon.js";

const kPatterns = ["bayer2", "bayer4", "bayer8", "blue_noise", "ps1", "ign"] as const;
const kSpaces = ["screen", "view_sphere", "world"] as const;
const kBlueSize = 64;

/**
 * Void-and-cluster rank map (Ulichney 1993), size n x n, toroidal Gaussian energy (sigma 1.5), deterministic.
 * Returns ranks / (n*n) in [0, 1). Written for this pass.
 */
export function voidAndCluster(n: number, sigma = 1.5, seed = 1): Float32Array {
    const N = n * n;
    const R = Math.ceil(4 * sigma);
    const kernel: [number, number, number][] = [];
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) kernel.push([dx, dy, Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))]);
    const energy = new Float64Array(N);
    const on = new Uint8Array(N);
    const add = (i: number, s: number) => {
        const x = i % n, y = (i / n) | 0;
        for (const [dx, dy, w] of kernel) energy[((y + dy + n) % n) * n + ((x + dx + n) % n)]! += s * w;
    };
    // initial binary pattern: ~10 % random points (LCG), then relaxed by swapping the tightest cluster into the largest void
    let st = seed >>> 0;
    const rnd = () => ((st = (Math.imul(st, 1664525) + 1013904223) >>> 0) / 4294967296);
    const initial = Math.max(1, Math.floor(N / 10));
    let placed = 0;
    while (placed < initial) { const i = Math.floor(rnd() * N); if (!on[i]) { on[i] = 1; add(i, 1); placed++; } }
    const argExt = (want: number, max: boolean) => {
        let best = -1, bv = max ? -Infinity : Infinity;
        for (let i = 0; i < N; i++) if (on[i] === want && (max ? energy[i]! > bv : energy[i]! < bv)) { bv = energy[i]!; best = i; }
        return best;
    };
    for (let it = 0; it < 4 * N; it++) {
        const c = argExt(1, true); on[c] = 0; add(c, -1);
        const v = argExt(0, false);
        if (v === c) { on[c] = 1; add(c, 1); break; }
        on[v] = 1; add(v, 1);
    }
    const rank = new Float32Array(N);
    // phase 1: remove points from the tightest clusters, ranks initial-1 .. 0
    const proto = on.slice(); const protoEnergy = energy.slice();
    for (let r = initial - 1; r >= 0; r--) { const c = argExt(1, true); on[c] = 0; add(c, -1); rank[c] = r; }
    on.set(proto); energy.set(protoEnergy);
    // phase 2/3: fill the largest voids, ranks initial .. N-1
    for (let r = initial; r < N; r++) { const v = argExt(0, false); on[v] = 1; add(v, 1); rank[v] = r; }
    for (let i = 0; i < N; i++) rank[i] = rank[i]! / N;
    return rank;
}

const kBlueCache = new WeakMap<Device, Texture>();

function blueNoiseTexture(device: Device): Texture {
    let t = kBlueCache.get(device);
    if (!t) {
        const r = voidAndCluster(kBlueSize);
        const data = new Float32Array(kBlueSize * kBlueSize * 4);
        for (let i = 0; i < r.length; i++) data.set([r[i]!, 0, 0, 1], i * 4);
        t = device.createTexture2D(kBlueSize, kBlueSize, ResourceFormat.RGBA32Float, 1, 1, data, ResourceBindFlags.ShaderResource);
        kBlueCache.set(device, t);
    }
    return t;
}

const kDefaults = {
    pattern: "bayer4" as string,
    space: "screen" as string,
    levels: 4,
    scale: 1,
    worldScale: 8,
    strength: 1,
    mix: 1,
};

export class DitherV2 extends StylePostPass<typeof kDefaults> {
    protected readonly shaderFile = "RenderPasses/Stylize/DitherV2.cs.slang";
    protected readonly outputDesc = "Dithered colour";
    protected readonly ports: readonly PortSpec[] = [
        ["src", "Colour (linear)", false, "gSrc"],
        ["posW", "World-space position (optional; view_sphere / world space)", true, "gPosW"],
        ["normW", "World-space normal (optional; world space axis)", true, "gNormW"],
    ];

    constructor(device: Device, props: Properties) {
        super(device, props, kDefaults);
        enumIndex("DitherV2", "pattern", this.p.pattern, kPatterns);
        enumIndex("DitherV2", "space", this.p.space, kSpaces);
    }

    protected setConstants(cb: ShaderVar, _W: number, H: number, _rd: RenderData): void {
        const pattern = enumIndex("DitherV2", "pattern", this.p.pattern, kPatterns);
        const space = enumIndex("DitherV2", "space", this.p.space, kSpaces);
        let angScale = 1, camPos: number[] = [0, 0, 0];
        if (this.scene) {
            const cam = cameraData(this.scene);
            camPos = cam.pos;
            const focalPx = H / (2 * Math.tan(0.5 * cam.fovY));
            angScale = focalPx / Math.max(this.p.scale, 1e-3);
        } else if (space !== 0) {
            throw new Error("DitherV2: view_sphere / world space need a scene");
        }
        cb["gPattern"] = pattern;
        cb["gSpace"] = space;
        cb["gLevels"] = Math.min(Math.max(Math.round(this.p.levels), 2), 256);
        cb["gScale"] = this.p.scale;
        cb["gAngScale"] = angScale;
        cb["gWorldScale"] = this.p.worldScale;
        cb["gCamPos"] = camPos;
        cb["gStrength"] = this.p.strength;
        cb["gMix"] = this.p.mix;
        cb["gBlueSize"] = kBlueSize;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        this.extraBindings = { gBlueNoise: blueNoiseTexture(this.device) };
        super.execute(ctx, rd);
    }
}

registerRenderPass("DitherV2", (device, props) => new DitherV2(device, props), {
    desc: "Ordered dither (Bayer 2/4/8, blue noise, PS1 matrix, IGN) in screen, view-sphere or world space",
});
