/**
 * ParticleLayer: a GPU particle layer (dust, sparks, embers, fireflies, snow, bokeh motes) with screen-space depth-buffer
 * collision, rendered to a colour + depth layer for DepthComposite.
 *
 * Particles live in world space inside an emitter box (`boundsMin`/`boundsMax`; both zero = the scene bounds). Each frame
 * advances the simulation by a fixed `dt` (not wall-clock time, so a frame sequence is reproducible): gravity, a drag
 * towards `wind`, and a smooth per-particle sinusoidal turbulence field; a particle respawns at a random point of the box
 * when its age exceeds its lifetime (lifetime x [0.5, 1.5)) or when it leaves the box. With `collide` and `posW`
 * connected, a particle that would end up behind the visible surface (within `thickness` x the surface's view depth)
 * stays in front and bounces off the surface normal (`normW`, else the view axis) with `restitution`.
 *
 * Rendering: each particle is a soft disc (falloff (1 - r^2)^2) of world radius `size` (pixel radius clamped to
 * [0.75, maxRadiusPx]), hidden per pixel behind the scene surface (needs `posW`). Contributions are accumulated with
 * integer atomics (fixed point, 1/4096), so the result is independent of thread order.
 *   color: rgb = sum of colour x intensity x twinkle x falloff (emission), a = min(1, sum of falloff) (coverage);
 *          composite with DepthComposite (alphaMode additive for glowing particles, premultiplied for opaque flakes);
 *   depth: r = linear view depth of the nearest particle (DepthComposite's posW convention), 3.4e38 where none.
 *
 * State and history: the particle buffer is per-pass state. It is re-seeded (identical to a fresh context) on the first
 * frame, on resetHistory(), on a scene/camera change, on Camera.resetHistory() and on a primed camera cut (HistoryTracker
 * in Utils/History.ts); simulation time restarts at 0. Particle i always starts from the same seed-derived state, so
 * a shot renders identically whatever was rendered before it.
 *
 * Identity: intensity 0 (or count 0) gives an empty layer (rgb 0, alpha 0, far depth), which DepthComposite passes
 * through bit for bit.
 *
 * Inputs : posW? normW? (scene, for occlusion and collision)
 * Outputs: color, depth (RGBA32Float)
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
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture, parseColour } from "../Stylize/StyleCommon.js";
import { HistoryTracker } from "../Utils/History.js";

const kShaderFile = "RenderPasses/Layers/ParticleLayer.cs.slang";
const kEntries = ["init", "simulate", "clearAccum", "splat", "resolve"] as const;
type Entry = (typeof kEntries)[number];

const kDefaults = {
    count: 4096,
    seed: 0,
    boundsMin: [0, 0, 0] as number[],
    boundsMax: [0, 0, 0] as number[],
    dt: 1 / 30,
    lifetime: 4,
    gravity: [0, -0.05, 0] as number[],
    wind: [0, 0, 0] as number[],
    drag: 0.5,
    turbulence: 0.2,
    turbulenceScale: 2,
    size: 0.01,
    maxRadiusPx: 6,
    colour: [1, 0.85, 0.6] as number[] | string,
    colourJitter: 0.2,
    intensity: 1,
    twinkle: 0,
    twinkleRate: 1,
    collide: true,
    restitution: 0.3,
    thickness: 0.05,
};
export type ParticleLayerParams = typeof kDefaults;

export class ParticleLayer extends RenderPass {
    private p: ParticleLayerParams = JSON.parse(JSON.stringify(kDefaults));
    private passes = new Map<Entry, ComputePass>();
    private readonly history = new HistoryTracker();
    private state: Buffer | null = null;
    private accum: Buffer | null = null;
    private depthBits: Buffer | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof ParticleLayerParams)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        this.p.count = Math.max(0, Math.min(Math.round(this.p.count), 1 << 20));
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Intensity", this.p.intensity, 0, 10, 0.01, (v) => (this.p.intensity = v));
        ui.slider("Size", this.p.size, 0.001, 0.2, 0.001, (v) => (this.p.size = v));
    }

    /** Re-seeds the particles on the next frame. */
    resetHistory(): void {
        this.history.reset();
    }

    /** Simulation steps since the last re-seed. */
    get frameIndex(): number {
        return this.history.frameIndex;
    }

    /** The particle state buffer (2 x float4 per particle: position, age; velocity, generation), for tests. */
    get stateBuffer(): Buffer | null {
        return this.state;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "posW", "Scene world position (occlusion, collision)", true);
        addTexInput(r, "normW", "Scene normal (collision)", true);
        addColorOutput(r, "color", "Particle layer: emission (rgb), coverage (a)", w, h);
        addColorOutput(r, "depth", "Nearest particle view depth (r)", w, h, true);
        return r;
    }

    private kernel(e: Entry): ComputePass {
        let k = this.passes.get(e);
        if (!k) this.passes.set(e, (k = ComputePass.create(this.device, { path: kShaderFile, csEntry: e })));
        return k;
    }

    private ensureBuffers(W: number, H: number): void {
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const n = Math.max(this.p.count, 1);
        if (!this.state || this.state.size !== n * 32) this.state = new Buffer(this.device, { size: n * 32, structSize: 16, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name: "ParticleLayer::state" });
        if (!this.accum || this.accum.size !== W * H * 16) {
            this.accum = new Buffer(this.device, { size: W * H * 16, structSize: 4, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name: "ParticleLayer::accum" });
            this.depthBits = new Buffer(this.device, { size: W * H * 4, structSize: 4, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name: "ParticleLayer::depth" });
        }
    }

    private bounds(): [number[], number[]] {
        const zero = (v: number[]) => v.every((x) => Number(x) === 0);
        if (!(zero(this.p.boundsMin) && zero(this.p.boundsMax))) return [this.p.boundsMin.map(Number), this.p.boundsMax.map(Number)];
        const b = this.scene?.bounds;
        if (b && Number.isFinite(b.minPoint.x)) return [[b.minPoint.x, b.minPoint.y, b.minPoint.z], [b.maxPoint.x, b.maxPoint.y, b.maxPoint.z]];
        return [[-1, 0, -1], [1, 2, 1]];
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const color = rd.getTexture("color");
        if (!color) return;
        const W = color.width, H = color.height;
        const depthOut = rd.getTexture("depth"), posW = rd.getTexture("posW"), normW = rd.getTexture("normW");
        const frame = this.history.begin(this.scene, `${this.p.count}`);
        this.ensureBuffers(W, H);

        const [bmin, bmax] = this.bounds();
        const cam = this.scene?.camera;
        const d = cam?.getData();
        const vp = d?.viewProjMatNoJitter;
        const rows = [0, 1, 2, 3].map((r) => (vp ? [vp.get(r, 0), vp.get(r, 1), vp.get(r, 2), vp.get(r, 3)] : [0, 0, 0, 0]));
        const pos = cam ? cam.getPosition() : { x: 0, y: 0, z: 0 };
        const tgt = cam ? cam.getTarget() : { x: 0, y: 0, z: -1 };
        const f = [tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z];
        const fl = Math.hypot(f[0]!, f[1]!, f[2]!) || 1;
        const fovY = cam ? cam.getFovY() : Math.PI / 3;
        const consts: Record<string, unknown> = {
            gFrameDim: [W, H], gCount: this.p.count, gSeed: Math.max(0, Math.round(this.p.seed)) >>> 0,
            gBoundsMin: bmin, gBoundsMax: bmax, gDt: this.p.dt, gTime: frame.frameIndex * this.p.dt,
            gGravity: this.p.gravity, gLifetime: Math.max(this.p.lifetime, 1e-4), gWind: this.p.wind, gDrag: this.p.drag,
            gColour: parseColour(this.p.colour), gTurbulence: this.p.turbulence, gTurbulenceScale: this.p.turbulenceScale,
            gColourJitter: this.p.colourJitter, gIntensity: this.p.intensity, gTwinkle: this.p.twinkle, gTwinkleRate: this.p.twinkleRate,
            gSize: this.p.size, gMaxRadiusPx: Math.max(this.p.maxRadiusPx, 0.75), gRestitution: this.p.restitution, gThickness: this.p.thickness,
            gCollide: this.p.collide ? 1 : 0, gHasPosW: posW ? 1 : 0, gHasNormW: normW ? 1 : 0,
            gViewProj0: rows[0], gViewProj1: rows[1], gViewProj2: rows[2], gViewProj3: rows[3],
            gCamPos: [pos.x, pos.y, pos.z], gPixelsPerUnitAtUnitDepth: (0.5 * H) / Math.tan(0.5 * fovY),
            gCamFwd: [f[0]! / fl, f[1]! / fl, f[2]! / fl],
        };
        const run = (e: Entry, x: number, y: number) => {
            const k = this.kernel(e);
            const root = k.getRootVar();
            const cb = root["CB"]!;
            for (const [name, v] of Object.entries(consts)) cb[name] = v as never;
            root["gState"] = this.state!;
            root["gAccum"] = this.accum!;
            root["gDepthBits"] = this.depthBits!;
            root["gPosW"] = posW ?? dummyTexture(this.device);
            root["gNormW"] = normW ?? dummyTexture(this.device);
            root["gColor"] = color;
            root["gDepth"] = depthOut ?? dummyTexture(this.device, "out:depth");
            k.execute(ctx, x, y);
        };
        const n = this.p.count;
        if (frame.reset && n > 0) run("init", n, 1);
        if (n > 0) run("simulate", n, 1);
        run("clearAccum", W, H);
        if (n > 0 && this.p.intensity !== 0) run("splat", n, 1);
        run("resolve", W, H);
    }
}

registerRenderPass("ParticleLayer", (device, props) => new ParticleLayer(device, props), {
    desc: "GPU particle layer (dust, sparks, snow, bokeh motes) with depth-buffer collision; colour + depth for DepthComposite",
});
