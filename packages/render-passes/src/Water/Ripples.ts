/**
 * Ripples: interactive water ripples on a horizontal water plane, simulated as a world-anchored height field (2D wave
 * equation on a `resolution`^2 grid covering a `size` x `size` square around `centre`, plane height = centre along
 * `upAxis`) and applied to the scene as a normal perturbation, so screen-space reflections (SSR) and lighting passes
 * downstream show the waves.
 *
 * Drivers (all deterministic; the clock is the step count since the last reset x 1 / (fps x substeps)):
 *  - rain: `dropsPerStep` random impacts per step (radius `dropRadius`, depth `dropStrength`), seeded by `seed`;
 *  - emitters: [{ position: [x,y,z], radius, strength, period, phase }] oscillating sources (a bobbing float, a fountain);
 *  - movers: [{ from: [x,y,z], to: [x,y,z], start, end, radius, strength }] objects moving through the water between
 *    `start` and `end` seconds (wake);
 *  - contact: scene objects chosen by instance id (`contactIds`, raw geometry instance ids from StyleAovs.ids) whose
 *    surface lies within `contactDistance` of the plane bob with `contactStrength` x sin(2 pi t / contactPeriod) in the
 *    cells they touch (screen-space: only the visible contact counts).
 * Each frame runs `substeps` wave steps (h' = ((h_N + h_S + h_E + h_W) / 2 - h_prev) x damping + drivers; a sponge
 * border absorbs outgoing waves). Application: pixels on the plane (|height - level| <= planeTolerance, N.up >= upMin,
 * inside the region, faded over `edgeFade` x the region at its border, x mask[maskChannel] when `mask` is connected)
 * get normal = normalize(N - strength x w x grad h).
 *
 * State and history: the field is per-pass state in a ping-pong pair (Utils/History.ts). It is cleared and the clock
 * restarts on the first execute, resetHistory() / resetGraphHistory(), a scene or camera change, Camera.resetHistory(),
 * a primed camera cut and a change of resolution; a shot rendered after a reset is bit-identical to the same shot on a
 * fresh context.
 *
 * Identity: strength 0 copies normW bit for bit (as does every pixel off the water or with a flat field).
 *
 * Inputs : posW, normW, mask? (e.g. RainWetness.wetness, puddles in g), ids? (StyleAovs.ids, for contactIds)
 * Outputs: normW (perturbed), height? (h, weight, dh/dt1, dh/dt2)
 */

import {
    Buffer,
    ComputePass,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type Texture,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture } from "../Stylize/StyleCommon.js";
import { HistoryTracker, PingPongTextures } from "../Utils/History.js";

const kShaderFile = "RenderPasses/Water/Ripples.cs.slang";
/** Compute entry points declared by the companion Slang shader. */
const kEntries = ["clearContact", "contact", "step", "apply"] as const;
type Entry = (typeof kEntries)[number];
/** Matches the eight emitter, mover and contact entries in Ripples.cs.slang. */
const kMax = 8;

/** An oscillating source on the water plane; period and phase are in seconds and radians. */
export interface RippleEmitter {
    position: number[];
    radius?: number;
    strength?: number;
    period?: number;
    phase?: number;
}
/** A wake source moving from one world position to another between start and end times. */
export interface RippleMover {
    from: number[];
    to: number[];
    start?: number;
    end?: number;
    radius?: number;
    strength?: number;
}

/** Creation defaults for the pass parameters documented in this file header. */
const kDefaults = {
    centre: [0, 0, 0] as number[],
    size: 8,
    resolution: 256,
    upAxis: [0, 1, 0] as number[],
    fps: 30,
    substeps: 2,
    damping: 0.985,
    sponge: 6,
    seed: 0,
    dropsPerStep: 0.5,
    dropRadius: 0.06,
    dropStrength: 0.004,
    emitters: [] as RippleEmitter[],
    movers: [] as RippleMover[],
    contactIds: [] as number[],
    contactDistance: 0.05,
    contactStrength: 0.003,
    contactPeriod: 1,
    strength: 1,
    planeTolerance: 0.1,
    upMin: 0.9,
    edgeFade: 0.05,
    maskChannel: 1,
};
/** Creation properties for Ripples, including its documented defaults. */
export type RipplesParams = typeof kDefaults;

/** Unit up vector and an orthonormal planar basis (t1, t2) for it (shared with the tests). */
export function ripplesBasis(upAxis: ArrayLike<number>): { up: number[]; t1: number[]; t2: number[] } {
    let up = [Number(upAxis[0]), Number(upAxis[1]), Number(upAxis[2])];
    const l = Math.hypot(up[0]!, up[1]!, up[2]!);
    up = l > 0 ? up.map((x) => x / l) : [0, 1, 0];
    const a = Math.abs(up[0]!) < 0.9 ? [1, 0, 0] : [0, 0, 1];
    const cross = (x: number[], y: number[]) => [
        x[1]! * y[2]! - x[2]! * y[1]!,
        x[2]! * y[0]! - x[0]! * y[2]!,
        x[0]! * y[1]! - x[1]! * y[0]!,
    ];
    let t2 = cross(up, a);
    const l2 = Math.hypot(t2[0]!, t2[1]!, t2[2]!);
    t2 = t2.map((x) => x / l2);
    return { up, t1: cross(t2, up), t2 };
}

/**
 * Drops per step: a fractional rate r gives floor(r) drops on every step plus one more on the steps where
 * floor((s + 1) r) > floor(s r) (deterministic, rate-exact over time).
 */
export function dropsAtStep(rate: number, step: number): number {
    if (!(rate > 0)) return 0;
    return Math.min(Math.floor((step + 1) * rate) - Math.floor(step * rate), 64);
}

/** Simulates a world-anchored water height field and applies its normal perturbation. */
export class Ripples extends RenderPass {
    private mParameters: RipplesParams = JSON.parse(JSON.stringify(kDefaults));
    private mPasses: Map<Entry, ComputePass> = new Map<Entry, ComputePass>();
    private readonly mHistory: HistoryTracker = new HistoryTracker();
    private readonly mHistoryTextures: PingPongTextures;
    private mContactBuffer: Buffer | null = null;
    private mLatest: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof RipplesParams)[])
            (this.mParameters as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        this.mParameters.resolution = Math.max(8, Math.min(Math.round(Number(this.mParameters.resolution)), 2048));
        this.mParameters.substeps = Math.max(1, Math.min(Math.round(Number(this.mParameters.substeps)), 16));
        this.mHistoryTextures = new PingPongTextures(device, ResourceFormat.RGBA32Float, "Ripples::field");
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.mParameters)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Strength", this.mParameters.strength, 0, 4, 0.01, (v) => (this.mParameters.strength = v));
        ui.slider("Damping", this.mParameters.damping, 0.9, 1, 0.001, (v) => (this.mParameters.damping = v));
        ui.slider("Drops per step", this.mParameters.dropsPerStep, 0, 8, 0.01, (v) => (this.mParameters.dropsPerStep = v));
    }

    /** Clears the field and restarts the clock on the next frame. */
    resetHistory(): void {
        this.mHistory.reset();
    }

    /** Frames since the last reset. */
    get frameIndex(): number {
        return this.mHistory.frameIndex;
    }

    /** The field after the last frame (RGBA32Float, resolution^2: h, h_prev), for tests. */
    get fieldTexture(): Texture | null {
        return this.mLatest;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "posW", "World position");
        addTexInput(r, "normW", "World shading normal");
        addTexInput(r, "mask", "Optional water mask (channel maskChannel), e.g. RainWetness.wetness (puddles in g)", true);
        addTexInput(r, "ids", "Optional StyleAovs.ids (raw geometry instance id in r) for contactIds", true);
        addColorOutput(r, "normWOut", "Normal with the ripples", w, h);
        addColorOutput(r, "height", "Field height (r), water weight (g), slope (ba)", w, h, true);
        return r;
    }

    private kernel(e: Entry): ComputePass {
        let k = this.mPasses.get(e);
        if (!k) this.mPasses.set(e, (k = ComputePass.create(this.device, { path: kShaderFile, csEntry: e })));
        return k;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const out = rd.getTexture("normWOut");
        const posW = rd.getTexture("posW");
        const normW = rd.getTexture("normW");
        if (!out || !posW || !normW) return;
        const W = out.width;
        const H = out.height;
        const N = this.mParameters.resolution;
        const sub = this.mParameters.substeps;
        const frame = this.mHistory.begin(this.scene, `${N}`);
        this.mHistoryTextures.get(ctx, N, N, frame);
        const tex = this.mHistoryTextures.textures!;
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        if (!this.mContactBuffer || this.mContactBuffer.size !== N * N * 4)
            this.mContactBuffer = new Buffer(this.device, {
                size: N * N * 4,
                structSize: 4,
                bindFlags: flags,
                memoryType: MemoryType.DeviceLocal,
                name: "Ripples::contact",
            });

        const { up, t1, t2 } = ripplesBasis(this.mParameters.upAxis);
        const planar = (v: number[]) => [
            v[0]! * t1[0]! + v[1]! * t1[1]! + v[2]! * t1[2]!,
            v[0]! * t2[0]! + v[1]! * t2[1]! + v[2]! * t2[2]!,
        ];
        const c3 = this.mParameters.centre.map(Number);
        const em = (this.mParameters.emitters ?? []).slice(0, kMax);
        const mv = (this.mParameters.movers ?? []).slice(0, kMax);
        const pad = (a: number[][]) => Array.from({ length: kMax }, (_, i) => a[i] ?? [0, 0, 0, 0]);
        const ids = (this.mParameters.contactIds ?? []).slice(0, kMax).map((x) => Math.max(0, Math.round(Number(x))) >>> 0);
        const idv = Array.from({ length: kMax }, (_, i) => ids[i] ?? 0xffffffff);
        const idsTex = rd.getTexture("ids");
        const mask = rd.getTexture("mask");
        const useContact = !!idsTex && ids.length > 0;
        const consts: Record<string, unknown> = {
            gFrameDim: [W, H],
            gGrid: N,
            gStep: 0,
            gUp: up,
            gLevel: c3[0]! * up[0]! + c3[1]! * up[1]! + c3[2]! * up[2]!,
            gT1: t1,
            gSize: Math.max(Number(this.mParameters.size), 1e-3),
            gT2: t2,
            gDamping: Math.min(Math.max(Number(this.mParameters.damping), 0), 1),
            gCentre: planar(c3),
            gStepTime: 1 / (Math.max(Number(this.mParameters.fps), 1e-3) * sub),
            gSeed: Math.max(0, Math.round(Number(this.mParameters.seed))) >>> 0,
            gDrops: 0,
            gDropRadius: Math.max(Number(this.mParameters.dropRadius), 1e-4),
            gDropStrength: Number(this.mParameters.dropStrength),
            gEmitterCount: em.length,
            gEmitterA: pad(
                em.map((e) => [...planar(e.position.map(Number)), Math.max(Number(e.radius ?? 0.1), 1e-4), Number(e.strength ?? 0.003)]),
            ),
            gEmitterB: pad(em.map((e) => [Number(e.period ?? 1), Number(e.phase ?? 0), 0, 0])),
            gMoverCount: mv.length,
            gContactStrength: Number(this.mParameters.contactStrength),
            gContactPeriod: Number(this.mParameters.contactPeriod),
            gContactDistance: Number(this.mParameters.contactDistance),
            gMoverA: pad(mv.map((m) => [...planar(m.from.map(Number)), ...planar(m.to.map(Number))])),
            gMoverB: pad(
                mv.map((m) => [
                    Number(m.start ?? 0),
                    Number(m.end ?? 1),
                    Math.max(Number(m.radius ?? 0.1), 1e-4),
                    Number(m.strength ?? 0.05),
                ]),
            ),
            gContactIds0: idv.slice(0, 4),
            gContactIds1: idv.slice(4, 8),
            gContactCount: useContact ? ids.length : 0,
            gApplyContact: 0,
            gStrength: Math.max(Number(this.mParameters.strength), 0),
            gPlaneTolerance: Math.max(Number(this.mParameters.planeTolerance), 0),
            gUpMin: Number(this.mParameters.upMin),
            gEdgeFade: Math.max(Number(this.mParameters.edgeFade), 0),
            gHasMask: mask ? 1 : 0,
            gMaskChannel: Math.min(Math.max(Math.round(Number(this.mParameters.maskChannel)), 0), 3),
            gSponge: Math.max(Number(this.mParameters.sponge), 0),
        };
        const run = (e: Entry, x: number, y: number, extra: Record<string, unknown>, bind: Record<string, unknown>) => {
            const k = this.kernel(e);
            const root = k.getRootVar();
            const cb = root["CB"]!;
            for (const [name, v] of Object.entries({ ...consts, ...extra })) cb[name] = v as never;
            for (const [name, v] of Object.entries(bind)) root[name] = v as never;
            k.execute(ctx, x, y);
        };
        const heightOut = rd.getTexture("height") ?? dummyTexture(this.device, "out:height");
        // Every kernel gets every resource (placeholders where unused; a texture is never bound sampled and as storage at once).
        const binds = (field: Texture, write: Texture): Record<string, unknown> => ({
            gRead: field,
            gField: field,
            gWrite: write,
            gContact: this.mContactBuffer,
            gPosW: posW,
            gNormW: normW,
            gIds: idsTex ?? dummyTexture(this.device, "in3"),
            gMask: mask ?? dummyTexture(this.device, "in2"),
            gNormWOut: out,
            gHeightOut: heightOut,
        });
        const spare = dummyTexture(this.device, "out:field");
        if (useContact) {
            run("clearContact", N, N, {}, binds(tex[0]!, spare));
            run("contact", W, H, {}, binds(tex[0]!, spare));
        }
        const rate = Number(this.mParameters.dropsPerStep);
        for (let k = 0; k < sub; k++) {
            const s = frame.frameIndex * sub + k;
            run(
                "step",
                N,
                N,
                { gStep: s, gDrops: dropsAtStep(rate, s), gApplyContact: useContact && k === 0 ? 1 : 0 },
                binds(tex[s & 1]!, tex[1 - (s & 1)]!),
            );
        }
        this.mLatest = tex[((frame.frameIndex + 1) * sub) & 1]!;
        run("apply", W, H, {}, binds(this.mLatest, spare));
    }
}

registerRenderPass("Ripples", (device, props) => new Ripples(device, props), {
    desc: "Interactive water ripples: world-anchored wave-equation height field (rain drops, emitters, movers, object contact) applied as normals",
});
