/**
 * WorldRevealMask: a world-space (or object-space) region mask that switches between two colour inputs.
 *
 * Every pixel's position is tested against a region and the pass outputs dst = lerp(a, b, mask): `b` inside the region,
 * `a` outside. With a time-driven sweep the region grows (sphere radius, box half extents) or moves (plane offset), so the
 * second look is "revealed" across the scene; a noisy, soft or glowing edge marks the front.
 *  - shape "sphere": signed distance |p - center| - radius;
 *  - shape "plane" : dot(p, normal) - offset (the side the normal points away from is inside);
 *  - shape "box"   : box signed distance around `center` with `halfExtents`.
 * sweep = speed * time is added to the radius / offset / half extents. `noiseAmp` > 0 adds 3D value noise (cell size
 * 1 / noiseScale, seeded) to the signed distance for an irregular front. mask = 1 - smoothstep(-w/2, w/2, d) with
 * w = edgeWidth (a hard step when w = 0); `invert` swaps inside and outside. The positions are posW (`space` "world")
 * or the object-space posObj input (`space` "object", e.g. StyleAovs.posObj, so the region follows each object).
 * Optional id filter: with an id input and `idMode` "only" the mask is limited to the listed ids, with "except" the listed
 * ids are never switched. Background pixels (posW.w = 0) get `background` (0 = a, 1 = b). The mask is scaled by
 * `strength`; edgeIntensity adds strength * edgeColour * 4 m (1 - m) along the soft front (m = mask before `strength`).
 * Identity: strength 0 (or an empty region, e.g. radius 0 with no noise or sweep) returns `a` unchanged, bit for bit.
 * The pass keeps no state: an animated sweep is driven by the `time` property (set it per frame, e.g. via
 * setProperties({ time })).
 *
 * Inputs : a, b (colour), posW (w = 0: background), posObj (optional; required for space "object"), ids (optional, the id
 *          is read from channel `idChannel`, e.g. StyleAovs.ids r = instance id or StyleAovs.id g).
 * Outputs: dst (colour), mask (optional; r = mask, g = signed distance incl. noise, b = edge term, a = 1).
 * Params : shape {sphere|plane|box}, space {world|object}, center [x,y,z], radius, normal [x,y,z], offset, halfExtents [x,y,z],
 *          speed, time, edgeWidth, noiseScale, noiseAmp, seed, invert, background, strength, edgeColour [r,g,b] (linear),
 *          edgeIntensity, idMode {none|only|except}, ids (up to 8 integer ids), idChannel (0..3).
 *
 * Idea after the three.js "webgpu_postprocessing_transition" example (two scene renders mixed by a thresholded transition
 * mask with a soft edge): https://github.com/mrdoob/three.js/blob/dev/examples/webgpu_postprocessing_transition.html
 * three.js, MIT License, Copyright (c) 2010-2026 three.js authors. Here the mask is a world-space signed-distance region
 * instead of a screen texture; the code is written for this pass. Hash: PCG (Jarzynski & Olano 2020).
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    RuntimeError,
    registerRenderPass,
    type CompileData,
    type Device,
    type PropertyValue,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture, enumIndex } from "./StyleCommon.js";

const kShaderFile = "RenderPasses/Stylize/WorldRevealMask.cs.slang";
/** Ordered option names; preserve this order for numeric property compatibility. */
const kShapes = ["sphere", "plane", "box"] as const;
/** Ordered option names; preserve this order for numeric property compatibility. */
const kSpaces = ["world", "object"] as const;
/** Ordered option names; preserve this order for numeric property compatibility. */
const kIdModes = ["none", "only", "except"] as const;
export const kWorldRevealMaxIds = 8;

/** Creation defaults for the pass parameters documented in this file header. */
const kDefaults = {
    shape: "sphere" as string,
    space: "world" as string,
    center: [0, 0, 0] as number[],
    radius: 1,
    normal: [0, 1, 0] as number[],
    offset: 0,
    halfExtents: [1, 1, 1] as number[],
    speed: 0,
    time: 0,
    edgeWidth: 0.1,
    noiseScale: 1,
    noiseAmp: 0,
    seed: 0,
    invert: false,
    background: 0,
    strength: 1,
    edgeColour: [1, 0.6, 0.2] as number[],
    edgeIntensity: 0,
    idMode: "none" as string,
    ids: [] as number[],
    idChannel: 0,
};
type Params = typeof kDefaults;

const vec3 = (pass: string, key: string, v: unknown): [number, number, number] => {
    if (!Array.isArray(v) || v.length !== 3 || v.some((x) => !Number.isFinite(Number(x))))
        throw new Error(`${pass}: ${key} must be [x, y, z] (got ${JSON.stringify(v)})`);
    return [Number(v[0]), Number(v[1]), Number(v[2])];
};

/** Reveals one image through another using a world-space or object-space region. */
export class WorldRevealMask extends RenderPass {
    private mParameters: Params = { ...kDefaults };
    private mPass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    /** Updates the given properties (others keep their values), e.g. setProperties(new Properties({ time })) per frame. */
    override setProperties(props: Properties): void {
        const next = { ...this.mParameters };
        for (const k of Object.keys(kDefaults) as (keyof Params)[]) (next as Record<string, unknown>)[k] = props.get(k, next[k] as never);
        enumIndex("WorldRevealMask", "shape", next.shape, kShapes);
        enumIndex("WorldRevealMask", "space", next.space, kSpaces);
        enumIndex("WorldRevealMask", "idMode", next.idMode, kIdModes);
        for (const k of ["center", "normal", "halfExtents", "edgeColour"] as const) vec3("WorldRevealMask", k, next[k]);
        if (
            !Array.isArray(next.ids) ||
            next.ids.length > kWorldRevealMaxIds ||
            next.ids.some((x) => !Number.isInteger(Number(x)) || Number(x) < 0)
        )
            throw new Error(`WorldRevealMask: ids must be up to ${kWorldRevealMaxIds} non-negative integers`);
        if (!(Number(next.idChannel) >= 0 && Number(next.idChannel) <= 3)) throw new Error("WorldRevealMask: idChannel must be 0..3");
        this.mParameters = next;
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.mParameters)) as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Radius", this.mParameters.radius, 0, 100, 0.01, (v) => (this.mParameters.radius = v));
        ui.slider("Offset", this.mParameters.offset, -100, 100, 0.01, (v) => (this.mParameters.offset = v));
        ui.slider("Time", this.mParameters.time, 0, 60, 0.01, (v) => (this.mParameters.time = v));
        ui.slider("Edge width", this.mParameters.edgeWidth, 0, 10, 0.01, (v) => (this.mParameters.edgeWidth = v));
        ui.slider("Noise amplitude", this.mParameters.noiseAmp, 0, 10, 0.01, (v) => (this.mParameters.noiseAmp = v));
        ui.slider("Strength", this.mParameters.strength, 0, 1, 0.01, (v) => (this.mParameters.strength = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "a", "Colour outside the region");
        addTexInput(r, "b", "Colour inside the region");
        addTexInput(r, "posW", "World-space position (w = 0: background)");
        addTexInput(r, "posObj", "Object-space position (StyleAovs.posObj; for space = object)", true);
        addTexInput(r, "ids", "Per-pixel ids (for the id filter; channel idChannel)", true);
        addColorOutput(r, "dst", "lerp(a, b, mask)", w, h);
        addColorOutput(r, "mask", "mask (r), signed distance (g), edge term (b)", w, h, true);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        const a = rd.getTexture("a");
        const b = rd.getTexture("b");
        const posW = rd.getTexture("posW");
        if (!dst || !a || !b || !posW) return;
        const posObj = rd.getTexture("posObj");
        const ids = rd.getTexture("ids");
        const mask = rd.getTexture("mask");
        const space = enumIndex("WorldRevealMask", "space", this.mParameters.space, kSpaces);
        if (space === 1 && !posObj) throw new RuntimeError("WorldRevealMask: space 'object' needs the posObj input");
        const idMode = enumIndex("WorldRevealMask", "idMode", this.mParameters.idMode, kIdModes);
        if (idMode !== 0 && !ids) throw new RuntimeError("WorldRevealMask: idMode 'only'/'except' needs the ids input");
        this.mPass ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.mPass.getRootVar();
        const cb = root["CB"]!;
        const n = vec3("WorldRevealMask", "normal", this.mParameters.normal);
        const nl = Math.hypot(...n) || 1;
        const idList = new Array<number>(kWorldRevealMaxIds).fill(0);
        this.mParameters.ids.forEach((x, i) => (idList[i] = Number(x) >>> 0));
        cb["gFrameDim"] = [dst.width, dst.height];
        cb["gShape"] = enumIndex("WorldRevealMask", "shape", this.mParameters.shape, kShapes);
        cb["gSpace"] = space;
        cb["gCenter"] = vec3("WorldRevealMask", "center", this.mParameters.center);
        cb["gRadius"] = this.mParameters.radius;
        cb["gNormal"] = n.map((x) => x / nl);
        cb["gOffset"] = this.mParameters.offset;
        cb["gHalfExtents"] = vec3("WorldRevealMask", "halfExtents", this.mParameters.halfExtents);
        cb["gSweep"] = this.mParameters.speed * this.mParameters.time;
        cb["gEdgeWidth"] = Math.max(0, this.mParameters.edgeWidth);
        cb["gNoiseScale"] = this.mParameters.noiseScale;
        cb["gNoiseAmp"] = this.mParameters.noiseAmp;
        cb["gSeed"] = Math.max(0, Math.round(this.mParameters.seed)) >>> 0;
        cb["gInvert"] = this.mParameters.invert ? 1 : 0;
        cb["gBackground"] = this.mParameters.background;
        cb["gStrength"] = this.mParameters.strength;
        cb["gEdgeColour"] = vec3("WorldRevealMask", "edgeColour", this.mParameters.edgeColour);
        cb["gEdgeIntensity"] = this.mParameters.edgeIntensity;
        cb["gIdMode"] = idMode;
        cb["gIdCount"] = this.mParameters.ids.length;
        cb["gIdChannel"] = Math.round(this.mParameters.idChannel);
        cb["gIdList0"] = idList.slice(0, 4);
        cb["gIdList1"] = idList.slice(4, 8);
        cb["gHasMask"] = mask ? 1 : 0;
        root["gA"] = a;
        root["gB"] = b;
        root["gPosW"] = posW;
        root["gPosObj"] = posObj ?? dummyTexture(this.device);
        root["gIds"] = ids ?? dummyTexture(this.device);
        root["gDst"] = dst;
        root["gMask"] = mask ?? dummyTexture(this.device, "out:mask");
        this.mPass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("WorldRevealMask", (device, props) => new WorldRevealMask(device, props), {
    desc: "World/object-space region mask (sphere/plane/box, time sweep, noisy edge) that switches between two colour inputs",
});
