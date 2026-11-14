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
const kShapes = ["sphere", "plane", "box"] as const;
const kSpaces = ["world", "object"] as const;
const kIdModes = ["none", "only", "except"] as const;
export const kWorldRevealMaxIds = 8;

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
    if (!Array.isArray(v) || v.length !== 3 || v.some((x) => !Number.isFinite(Number(x)))) throw new Error(`${pass}: ${key} must be [x, y, z] (got ${JSON.stringify(v)})`);
    return [Number(v[0]), Number(v[1]), Number(v[2])];
};

export class WorldRevealMask extends RenderPass {
    private p: Params = { ...kDefaults };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    /** Updates the given properties (others keep their values), e.g. setProperties(new Properties({ time })) per frame. */
    override setProperties(props: Properties): void {
        const next = { ...this.p };
        for (const k of Object.keys(kDefaults) as (keyof Params)[]) (next as Record<string, unknown>)[k] = props.get(k, next[k] as never);
        enumIndex("WorldRevealMask", "shape", next.shape, kShapes);
        enumIndex("WorldRevealMask", "space", next.space, kSpaces);
        enumIndex("WorldRevealMask", "idMode", next.idMode, kIdModes);
        for (const k of ["center", "normal", "halfExtents", "edgeColour"] as const) vec3("WorldRevealMask", k, next[k]);
        if (!Array.isArray(next.ids) || next.ids.length > kWorldRevealMaxIds || next.ids.some((x) => !Number.isInteger(Number(x)) || Number(x) < 0))
            throw new Error(`WorldRevealMask: ids must be up to ${kWorldRevealMaxIds} non-negative integers`);
        if (!(Number(next.idChannel) >= 0 && Number(next.idChannel) <= 3)) throw new Error("WorldRevealMask: idChannel must be 0..3");
        this.p = next;
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)) as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Radius", this.p.radius, 0, 100, 0.01, (v) => (this.p.radius = v));
        ui.slider("Offset", this.p.offset, -100, 100, 0.01, (v) => (this.p.offset = v));
        ui.slider("Time", this.p.time, 0, 60, 0.01, (v) => (this.p.time = v));
        ui.slider("Edge width", this.p.edgeWidth, 0, 10, 0.01, (v) => (this.p.edgeWidth = v));
        ui.slider("Noise amplitude", this.p.noiseAmp, 0, 10, 0.01, (v) => (this.p.noiseAmp = v));
        ui.slider("Strength", this.p.strength, 0, 1, 0.01, (v) => (this.p.strength = v));
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
        const a = rd.getTexture("a"), b = rd.getTexture("b"), posW = rd.getTexture("posW");
        if (!dst || !a || !b || !posW) return;
        const posObj = rd.getTexture("posObj"), ids = rd.getTexture("ids"), mask = rd.getTexture("mask");
        const space = enumIndex("WorldRevealMask", "space", this.p.space, kSpaces);
        if (space === 1 && !posObj) throw new RuntimeError("WorldRevealMask: space 'object' needs the posObj input");
        const idMode = enumIndex("WorldRevealMask", "idMode", this.p.idMode, kIdModes);
        if (idMode !== 0 && !ids) throw new RuntimeError("WorldRevealMask: idMode 'only'/'except' needs the ids input");
        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        const n = vec3("WorldRevealMask", "normal", this.p.normal);
        const nl = Math.hypot(...n) || 1;
        const idList = new Array<number>(kWorldRevealMaxIds).fill(0);
        this.p.ids.forEach((x, i) => (idList[i] = Number(x) >>> 0));
        cb["gFrameDim"] = [dst.width, dst.height];
        cb["gShape"] = enumIndex("WorldRevealMask", "shape", this.p.shape, kShapes);
        cb["gSpace"] = space;
        cb["gCenter"] = vec3("WorldRevealMask", "center", this.p.center);
        cb["gRadius"] = this.p.radius;
        cb["gNormal"] = n.map((x) => x / nl);
        cb["gOffset"] = this.p.offset;
        cb["gHalfExtents"] = vec3("WorldRevealMask", "halfExtents", this.p.halfExtents);
        cb["gSweep"] = this.p.speed * this.p.time;
        cb["gEdgeWidth"] = Math.max(0, this.p.edgeWidth);
        cb["gNoiseScale"] = this.p.noiseScale;
        cb["gNoiseAmp"] = this.p.noiseAmp;
        cb["gSeed"] = Math.max(0, Math.round(this.p.seed)) >>> 0;
        cb["gInvert"] = this.p.invert ? 1 : 0;
        cb["gBackground"] = this.p.background;
        cb["gStrength"] = this.p.strength;
        cb["gEdgeColour"] = vec3("WorldRevealMask", "edgeColour", this.p.edgeColour);
        cb["gEdgeIntensity"] = this.p.edgeIntensity;
        cb["gIdMode"] = idMode;
        cb["gIdCount"] = this.p.ids.length;
        cb["gIdChannel"] = Math.round(this.p.idChannel);
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
        this.pass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("WorldRevealMask", (device, props) => new WorldRevealMask(device, props), {
    desc: "World/object-space region mask (sphere/plane/box, time sweep, noisy edge) that switches between two colour inputs",
});
