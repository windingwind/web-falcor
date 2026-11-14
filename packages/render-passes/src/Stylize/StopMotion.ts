/**
 * StopMotion: the stop-motion / "animated on twos" pack.
 *  - per-object frame rate: `nodeHolds` maps scene-graph node indices to a hold in frames (2 = on twos, 3 = on threes) at
 *    `fps`; the pass applies it to its scene with Scene.setAnimationTimeQuantization when the scene is set or the property
 *    changes, so those nodes (and their descendants) hold each pose while the rest of the scene and the camera move every
 *    frame. An empty `nodeHolds` (the default) leaves the scene's quantisation untouched;
 *  - per-object jitter: on every key (key = floor(time * fps / jitterHold)) each object is shifted by its own integer
 *    screen offset of up to `jitterPx` pixels (seeded hash of id and key), like a puppet nudged between exposures. The
 *    output pixel p takes the source pixel q (|q - p| <= jitterPx) whose object offset maps it onto p; the nearest view
 *    depth wins (with posW; without it unshifted objects stay in front and the first match in scan order wins among
 *    shifted ones). Pixels left uncovered keep the source colour.
 *    Background pixels and objects excluded by `jitterIds` (if non-empty, only the listed ids jitter) do not move;
 *  - flicker: the colour is scaled per flicker key (floor(time * fps / flickerHold)) by 1 + flicker * (2 h - 1), h a seeded
 *    hash, like the exposure variation of a real stop-motion shoot.
 * The pass keeps no state: the clip time comes from the `time` property (seconds; set it per frame, e.g. with
 * setProperties({ time })). Identity: jitterPx 0 and flicker 0 return `src` bit for bit (and empty nodeHolds leave the
 * scene unchanged).
 *
 * Inputs : src (colour), id (optional; per-pixel object id in channel `idChannel`, e.g. StyleAovs.id g = raw instance id,
 *          alpha 0 = background), posW (optional; occlusion order of shifted objects, w = 0: background).
 * Params : fps, time, nodeHolds ({ node: frames }), jitterPx (0..8), jitterHold (frames), jitterIds (up to 8 ids; empty = all),
 *          idChannel (0..3), flicker, flickerHold (frames), seed.
 *
 * Idea for the per-object holds after StopMotion-Unity (https://github.com/OskarSigvardsson/StopMotion-Unity), MIT License,
 * Copyright (c) 2010-2017 Google, Inc. http://angularjs.org (as stated in that repository's LICENSE): per-object transform
 * holds. Here the holds are done by the scene's animation time quantisation; the code is written for this pass.
 * Hash: PCG (Jarzynski & Olano 2020).
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
    type Scene,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, cameraData, dummyTexture } from "./StyleCommon.js";

const kShaderFile = "RenderPasses/Stylize/StopMotion.cs.slang";
export const kStopMotionMaxIds = 8;
export const kStopMotionMaxJitterPx = 8;

const kDefaults = {
    fps: 24,
    time: 0,
    nodeHolds: {} as Record<string, number>,
    jitterPx: 0,
    jitterHold: 2,
    jitterIds: [] as number[],
    idChannel: 1,
    flicker: 0,
    flickerHold: 2,
    seed: 0,
};
type Params = typeof kDefaults;

const pcg = (v: number): number => {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
};

/** Key index of `time` for a hold of `hold` frames at `fps` (same rounding guard as the scene's time quantisation). */
export function stopMotionKey(time: number, fps: number, hold: number): number {
    return Math.max(0, Math.floor((time * fps) / Math.max(1, hold) + 1e-6));
}

/** Flicker gain of a key: 1 + flicker * (2 h - 1), h in [0, 1) from the seed and key. */
export function stopMotionGain(flicker: number, key: number, seed: number): number {
    if (flicker === 0) return 1;
    const h = (pcg((key >>> 0) + pcg((seed ^ 0x9e3779b9) >>> 0)) >>> 8) / 16777216;
    return Math.fround(1 + flicker * (2 * h - 1));
}

export class StopMotion extends RenderPass {
    private p: Params = { ...kDefaults, nodeHolds: {} };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    /** Updates the given properties (others keep their values), e.g. setProperties(new Properties({ time })) per frame. */
    override setProperties(props: Properties): void {
        const next: Params = { ...this.p };
        for (const k of Object.keys(kDefaults) as (keyof Params)[]) (next as Record<string, unknown>)[k] = props.get(k, next[k] as never);
        const holds = next.nodeHolds ?? {};
        if (typeof holds !== "object" || Array.isArray(holds)) throw new Error("StopMotion: nodeHolds must be an object { node: frames }");
        for (const [k, v] of Object.entries(holds))
            if (!Number.isInteger(Number(k)) || Number(k) < 0 || !(Number(v) >= 1)) throw new Error(`StopMotion: nodeHolds entry ${k}: ${String(v)} (node index >= 0, hold >= 1 frame)`);
        if (!(next.fps > 0)) throw new Error("StopMotion: fps must be > 0");
        if (!(next.jitterPx >= 0 && next.jitterPx <= kStopMotionMaxJitterPx)) throw new Error(`StopMotion: jitterPx must be in [0, ${kStopMotionMaxJitterPx}]`);
        if (!Array.isArray(next.jitterIds) || next.jitterIds.length > kStopMotionMaxIds || next.jitterIds.some((x) => !Number.isInteger(Number(x)) || Number(x) < 0))
            throw new Error(`StopMotion: jitterIds must be up to ${kStopMotionMaxIds} non-negative integers`);
        if (!(next.idChannel >= 0 && next.idChannel <= 3)) throw new Error("StopMotion: idChannel must be 0..3");
        const holdsChanged = JSON.stringify(holds) !== JSON.stringify(this.p.nodeHolds) || next.fps !== this.p.fps;
        next.nodeHolds = { ...holds };
        this.p = next;
        if (holdsChanged) this.applyHolds();
    }

    override setScene(scene: Scene | null): void {
        super.setScene(scene);
        this.applyHolds();
    }

    /** Applies `nodeHolds` to the scene (only when non-empty: an empty table leaves the scene as it is). */
    private applyHolds(): void {
        const entries = Object.entries(this.p.nodeHolds);
        if (!this.scene || entries.length === 0) return;
        this.scene.setAnimationTimeQuantization(Object.fromEntries(entries.map(([k, v]) => [Number(k), { hold: Number(v) / this.p.fps }])));
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)) as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Time", this.p.time, 0, 60, 1 / 24, (v) => (this.p.time = v));
        ui.slider("Jitter (px)", this.p.jitterPx, 0, kStopMotionMaxJitterPx, 1, (v) => (this.p.jitterPx = v));
        ui.slider("Jitter hold (frames)", this.p.jitterHold, 1, 8, 1, (v) => (this.p.jitterHold = v));
        ui.slider("Flicker", this.p.flicker, 0, 0.5, 0.01, (v) => (this.p.flicker = v));
        ui.slider("Flicker hold (frames)", this.p.flickerHold, 1, 8, 1, (v) => (this.p.flickerHold = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "Colour");
        addTexInput(r, "id", "Per-pixel object id (channel idChannel; alpha 0 = background)", true);
        addTexInput(r, "posW", "World-space position (occlusion order of shifted objects)", true);
        addColorOutput(r, "dst", "Colour with per-object jitter and flicker", w, h);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const id = rd.getTexture("id"), posW = rd.getTexture("posW");
        const jitter = Math.round(this.p.jitterPx);
        if (jitter > 0 && !id) throw new RuntimeError("StopMotion: jitterPx > 0 needs the id input");
        if (posW && !this.scene) throw new RuntimeError("StopMotion: posW ordering needs a scene (camera)");
        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        const ids = new Array<number>(kStopMotionMaxIds).fill(0);
        this.p.jitterIds.forEach((x, i) => (ids[i] = Number(x) >>> 0));
        const cam = this.scene ? cameraData(this.scene) : null;
        cb["gFrameDim"] = [dst.width, dst.height];
        cb["gJitterPx"] = jitter;
        cb["gJitterKey"] = stopMotionKey(this.p.time, this.p.fps, this.p.jitterHold) >>> 0;
        cb["gSeed"] = Math.max(0, Math.round(this.p.seed)) >>> 0;
        cb["gGain"] = stopMotionGain(this.p.flicker, stopMotionKey(this.p.time, this.p.fps, this.p.flickerHold), Math.max(0, Math.round(this.p.seed)));
        cb["gIdChannel"] = Math.round(this.p.idChannel);
        cb["gIdCount"] = this.p.jitterIds.length;
        cb["gIdList0"] = ids.slice(0, 4);
        cb["gIdList1"] = ids.slice(4, 8);
        cb["gHasPosW"] = posW ? 1 : 0;
        cb["gCamPos"] = cam?.pos ?? [0, 0, 0];
        cb["gCamFwd"] = cam?.fwd ?? [0, 0, -1];
        root["gSrc"] = src;
        root["gId"] = id ?? dummyTexture(this.device);
        root["gPosW"] = posW ?? dummyTexture(this.device, "in:posW");
        root["gDst"] = dst;
        this.pass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("StopMotion", (device, props) => new StopMotion(device, props), {
    desc: "Stop-motion pack: per-node animation holds (on twos/threes), per-object screen jitter per key, exposure flicker",
});
