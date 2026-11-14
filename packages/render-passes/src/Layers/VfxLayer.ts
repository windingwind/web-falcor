/**
 * VfxLayer: procedural visual-effect emitters (fire, electric arcs, plasma orbs) anchored in world space and rendered to a
 * colour + depth layer for DepthComposite, next to ParticleLayer (sparks, embers) in the same layer stack.
 *
 * Emitters (`emitters`, up to 16) are objects:
 *   { type: "fire" | "electric" | "plasma", position: [x, y, z], size, colour, intensity, seed, speed, ...type options }
 *   fire     : a flame standing on `position` along `up` (default [0, 1, 0]); size = flame height (world units);
 *              width = half width / height (0.35), turbulence [0, 1] (0.8). The flame shape is a tapered profile whose
 *              width is modulated by domain-warped value noise scrolling upwards at `speed`; heat maps to the colour
 *              (outer flame) and a white-hot core.
 *   electric : an arc from `position` to `target`; size = glow width (world units); amplitude = sideways excursion as a
 *              share of the arc length (0.12); strands 1..4 (2). Each strand is a noise curve pinned at both ends and
 *              re-drawn `speed` times per second (discrete jumps, like a real discharge); bright white core + coloured glow.
 *   plasma   : an orb at `position`; size = radius; filaments = ridge sharpness (8), corona = halo extent in radii (0.6).
 *              Animated filament ridges of value noise on the sphere sprite, a fresnel rim and a noisy outer corona.
 * All emitters are evaluated per pixel against the projected anchors (a perspective-correct sprite at the anchor's
 * depth; arcs interpolate depth along the arc), so their size follows the camera. A layer sample is hidden behind the
 * scene surface when `posW` is connected (or faded over `softness` depth units in front of it).
 *
 * Time: the effect is a pure function of `time` (seconds, a property; set it per frame for clips) and the emitter seeds.
 * The pass keeps no state between frames, so any frame renders identically whatever was rendered before.
 *
 *   color: rgb = emission (premultiplied, HDR; composite with DepthComposite alphaMode "additive"), a = coverage;
 *   depth: r = linear view depth of the nearest visible emitter sample (DepthComposite's posW convention), 3.4e38 where none.
 *
 * Identity: intensity 0, or no emitters, gives an empty layer (rgb 0, alpha 0, far depth), which DepthComposite passes
 * through bit for bit.
 *
 * The flame and arc looks follow well-known procedural techniques (scrolling-noise flame billboards as in the three.js
 * webgpu_tsl_vfx_flames example, MIT; thin bright ridges of animated noise as in the Godot Shaders "Procedural Electric
 * Current" shader, MIT); this code is written from scratch and copies neither.
 *
 * Inputs : posW? (scene, for occlusion)
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
    RuntimeError,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture, parseColour } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Layers/VfxLayer.cs.slang";
const kTypes = ["fire", "electric", "plasma"] as const;
export const kVfxMaxEmitters = 16;

export interface VfxEmitter {
    type: (typeof kTypes)[number];
    position: number[];
    /** electric: end point. */
    target?: number[];
    /** fire: up axis. */
    up?: number[];
    size?: number;
    colour?: number[] | string;
    intensity?: number;
    seed?: number;
    speed?: number;
    /** fire */
    width?: number;
    turbulence?: number;
    /** electric */
    amplitude?: number;
    strands?: number;
    /** plasma */
    filaments?: number;
    corona?: number;
}

const kEmitterDefaults = {
    fire: { size: 1, colour: [1, 0.45, 0.12], intensity: 4, speed: 1, width: 0.35, turbulence: 0.8 },
    electric: { size: 0.02, colour: [0.45, 0.6, 1], intensity: 4, speed: 12, amplitude: 0.12, strands: 2 },
    plasma: { size: 0.3, colour: [0.7, 0.35, 1], intensity: 3, speed: 1, filaments: 8, corona: 0.6 },
} as const;

const kDefaults = {
    emitters: [] as VfxEmitter[],
    time: 0,
    intensity: 1,
    softness: 0,
    maxRadiusPx: 64,
};
export type VfxLayerParams = typeof kDefaults;

const vec3 = (v: unknown, fb: number[]): number[] => (Array.isArray(v) && v.length === 3 ? v.map(Number) : fb);

/** Packs emitters into 16 floats each (see VfxLayer.cs.slang). Exported for tests. */
export function packVfxEmitters(emitters: readonly VfxEmitter[]): Float32Array {
    const n = Math.min(emitters.length, kVfxMaxEmitters);
    const out = new Float32Array(Math.max(n, 1) * 16);
    for (let i = 0; i < n; i++) {
        const e = emitters[i]!;
        const type = kTypes.indexOf(e.type);
        if (type < 0) throw new RuntimeError(`VfxLayer: unknown emitter type '${String(e.type)}' (${kTypes.join(", ")})`);
        const d = { ...kEmitterDefaults[e.type], ...e } as Required<VfxEmitter>;
        const pos = vec3(d.position, [0, 0, 0]);
        const axis = e.type === "electric" ? vec3(d.target, [pos[0]!, pos[1]! + 1, pos[2]!]) : e.type === "fire" ? vec3(d.up, [0, 1, 0]) : [0, 0, 0];
        if (e.type === "fire") {
            const l = Math.hypot(axis[0]!, axis[1]!, axis[2]!) || 1;
            for (let k = 0; k < 3; k++) axis[k] = axis[k]! / l;
        }
        const shape = e.type === "fire" ? d.width : e.type === "electric" ? d.amplitude : d.corona;
        const detail = e.type === "fire" ? d.turbulence : e.type === "electric" ? Math.round(d.strands) : d.filaments;
        out.set([...pos, type, ...axis, Math.max(Number(d.size), 0), ...parseColour(d.colour), Number(d.intensity),
            Math.max(0, Math.round(Number(d.seed ?? 0))) >>> 0, Number(d.speed), Number(shape), Number(detail)], i * 16);
    }
    return out;
}

export class VfxLayer extends RenderPass {
    private p: VfxLayerParams = JSON.parse(JSON.stringify(kDefaults));
    private pass: ComputePass | null = null;
    private emitterBuf: Buffer | null = null;
    private packed: Float32Array = new Float32Array(16);

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof VfxLayerParams)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        this.setEmitters(this.p.emitters);
    }

    private setEmitters(e: VfxEmitter[]): void {
        if (!Array.isArray(e)) throw new RuntimeError("VfxLayer: emitters must be a list");
        this.packed = packVfxEmitters(e);
        this.p.emitters = JSON.parse(JSON.stringify(e.slice(0, kVfxMaxEmitters)));
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)));
    }

    override setProperties(props: Properties): void {
        for (const k of Object.keys(kDefaults) as (keyof VfxLayerParams)[]) {
            if (!props.has(k)) continue;
            if (k === "emitters") this.setEmitters(props.get(k, [] as never));
            else (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        }
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Intensity", this.p.intensity, 0, 10, 0.01, (v) => (this.p.intensity = v));
        ui.slider("Time", this.p.time, 0, 60, 0.01, (v) => (this.p.time = v));
        ui.slider("Softness", this.p.softness, 0, 10, 0.01, (v) => (this.p.softness = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "posW", "Scene world position (occlusion)", true);
        addColorOutput(r, "color", "VFX layer: emission (rgb), coverage (a)", w, h);
        addColorOutput(r, "depth", "Nearest visible emitter view depth (r)", w, h, true);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const color = rd.getTexture("color");
        if (!color) return;
        const W = color.width, H = color.height;
        const depthOut = rd.getTexture("depth"), posW = rd.getTexture("posW");
        const n = Math.min(this.p.emitters.length, kVfxMaxEmitters);
        if (!this.emitterBuf || this.emitterBuf.size !== this.packed.byteLength) {
            this.emitterBuf = new Buffer(this.device, { size: this.packed.byteLength, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: "VfxLayer::emitters" });
        }
        this.emitterBuf.setBlob(this.packed);

        const cam = this.scene?.camera;
        const vp = cam?.getData().viewProjMatNoJitter;
        const rows = [0, 1, 2, 3].map((r) => (vp ? [vp.get(r, 0), vp.get(r, 1), vp.get(r, 2), vp.get(r, 3)] : [0, 0, 0, 0]));
        const pos = cam ? cam.getPosition() : { x: 0, y: 0, z: 0 };
        const tgt = cam ? cam.getTarget() : { x: 0, y: 0, z: -1 };
        const f = [tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z];
        const fl = Math.hypot(f[0]!, f[1]!, f[2]!) || 1;
        const fovY = cam ? cam.getFovY() : Math.PI / 3;

        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        const consts: Record<string, unknown> = {
            gFrameDim: [W, H], gCount: cam ? n : 0, gTime: Number(this.p.time), gIntensity: Number(this.p.intensity),
            gSoftness: Math.max(Number(this.p.softness), 0), gHasPosW: posW ? 1 : 0, gMaxRadiusPx: Math.max(Number(this.p.maxRadiusPx), 0.75),
            gViewProj0: rows[0], gViewProj1: rows[1], gViewProj2: rows[2], gViewProj3: rows[3],
            gCamPos: [pos.x, pos.y, pos.z], gPixelsPerUnitAtUnitDepth: (0.5 * H) / Math.tan(0.5 * fovY),
            gCamFwd: [f[0]! / fl, f[1]! / fl, f[2]! / fl],
        };
        for (const [name, v] of Object.entries(consts)) cb[name] = v as never;
        root["gEmitters"] = this.emitterBuf;
        root["gPosW"] = posW ?? dummyTexture(this.device);
        root["gColor"] = color;
        root["gDepth"] = depthOut ?? dummyTexture(this.device, "out:depth");
        this.pass.execute(ctx, W, H);
    }
}

registerRenderPass("VfxLayer", (device, props) => new VfxLayer(device, props), {
    desc: "Procedural fire / electric arc / plasma emitters anchored in world space; colour + depth layer for DepthComposite",
});
