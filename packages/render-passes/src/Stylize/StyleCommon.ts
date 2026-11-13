/**
 * Shared host-side helpers of the stylization passes in this directory: optional-input placeholders, camera data and the
 * key-light direction. Shader-side helpers live in each pass's .cs.slang file (the passes are independent).
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RuntimeError,
    ResourceBindFlags,
    ResourceFormat,
    RenderPassReflection,
    FieldFlags,
    LightType,
    type CompileData,
    type Device,
    type PropertyValue,
    type RenderContext,
    type Scene,
    type ShaderVar,
    type Texture,
    type UIWidgets,
} from "@web-falcor/falcor";

const kDummies = new WeakMap<Device, Map<string, Texture>>();

/**
 * A 1x1 RGBA32Float texture bound in place of an unconnected optional input or output. Each `slot` gets its own texture,
 * so a placeholder is never bound as a storage texture and a sampled texture in the same dispatch.
 */
export function dummyTexture(device: Device, slot = "in"): Texture {
    let m = kDummies.get(device);
    if (!m) kDummies.set(device, (m = new Map()));
    let t = m.get(slot);
    if (!t) {
        t = device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array([0, 0, 0, 1]),
            ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
        m.set(slot, t);
    }
    return t;
}

/** Adds a full-resolution RGBA32Float output usable as a UAV and as a shader resource. */
export function addColorOutput(r: RenderPassReflection, name: string, desc: string, w: number, h: number, optional = false): void {
    const f = r.addOutput(name, desc).texture2D(w, h).format(ResourceFormat.RGBA32Float)
        .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    if (optional) f.flags(FieldFlags.Optional);
}

/** Adds a shader-resource input; optional inputs may stay unconnected (a 1x1 placeholder is bound instead). */
export function addTexInput(r: RenderPassReflection, name: string, desc: string, optional = false): void {
    const f = r.addInput(name, desc).bindFlags(ResourceBindFlags.ShaderResource);
    if (optional) f.flags(FieldFlags.Optional);
}

export interface CameraData {
    pos: [number, number, number];
    fwd: [number, number, number];
    /** Rows of the view matrix (world -> view, right-handed, camera looks down -z). */
    viewRows: [number[], number[], number[]];
    /** Rows of the view-projection matrix (world -> clip). */
    viewProjRows: [number[], number[], number[], number[]];
    fovY: number;
}

export function cameraData(scene: Scene): CameraData {
    const cam = scene.camera;
    const p = cam.getPosition(), t = cam.getTarget();
    const f = [t.x - p.x, t.y - p.y, t.z - p.z];
    const l = Math.hypot(f[0]!, f[1]!, f[2]!) || 1;
    const v = cam.getViewMatrix(), vp = cam.getViewProjMatrix();
    const row = (m: typeof v, r: number) => [m.get(r, 0), m.get(r, 1), m.get(r, 2), m.get(r, 3)];
    return {
        pos: [p.x, p.y, p.z],
        fwd: [f[0]! / l, f[1]! / l, f[2]! / l],
        viewRows: [row(v, 0), row(v, 1), row(v, 2)],
        viewProjRows: [row(vp, 0), row(vp, 1), row(vp, 2), row(vp, 3)],
        fovY: cam.getFovY(),
    };
}

const norm3 = (v: number[]): [number, number, number] => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
    return [v[0]! / l, v[1]! / l, v[2]! / l];
};

/**
 * Unit vector towards the key light: `override` if it is a non-zero 3-vector, else the first active directional light of
 * the scene (-dirW), else straight up.
 */
export function keyLightDir(scene: Scene | null, override: unknown): [number, number, number] {
    if (Array.isArray(override) && override.length === 3 && override.some((x) => Number(x) !== 0)) return norm3(override.map(Number));
    const d = scene?.activeLights.find((l) => l.type === LightType.Directional && l.dirW);
    if (d?.dirW) return norm3([-d.dirW.x, -d.dirW.y, -d.dirW.z]);
    return [0, 1, 0];
}

/** Parses "#RRGGBB" / "RRGGBB" (sRGB) or an [r, g, b] triple (linear) into linear RGB. */
export function parseColour(c: unknown): [number, number, number] {
    if (Array.isArray(c)) return [Number(c[0]), Number(c[1]), Number(c[2])];
    const s = String(c).replace(/^#/, "");
    if (!/^[0-9a-fA-F]{6}$/.test(s)) throw new Error(`bad colour '${String(c)}' (expected #RRGGBB or [r, g, b])`);
    return [0, 2, 4].map((i) => srgbToLinear(parseInt(s.slice(i, i + 2), 16) / 255)) as [number, number, number];
}

export function srgbToLinear(x: number): number {
    return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(x: number): number {
    x = Math.min(Math.max(x, 0), 1);
    return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

/** Index of `v` in `names` (numbers are taken as indices); throws on an unknown name. */
export function enumIndex(pass: string, key: string, v: unknown, names: readonly string[]): number {
    const i = typeof v === "number" ? Math.round(v) : names.indexOf(String(v));
    if (i < 0 || i >= names.length) throw new Error(`${pass}: ${key} must be one of ${names.join("|")} (got ${String(v)})`);
    return i;
}

export type ParamValue = number | string | boolean | number[] | string[];

/** [port name, description, optional, shader texture name] */
export type PortSpec = readonly [string, string, boolean, string];

/**
 * Base of the single-dispatch image passes here: one `dst` output (RGBA32Float, same size as `src`), named inputs bound to
 * shader textures (unconnected optional inputs get a placeholder), parameters with defaults read from Properties.
 * Subclasses provide the shader path, the ports and `setConstants`.
 */
export abstract class StylePostPass<P extends Record<string, ParamValue>> extends RenderPass {
    protected p: P;
    /** Additional textures bound by shader name on the next execute (e.g. lookup tables). */
    protected extraBindings: Record<string, Texture> = {};
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties, defaults: P) {
        super(device);
        this.p = { ...defaults };
        for (const k of Object.keys(defaults) as (keyof P & string)[]) this.p[k] = props.get(k, defaults[k] as never) as P[typeof k];
    }

    protected abstract readonly shaderFile: string;
    protected abstract readonly ports: readonly PortSpec[];
    protected abstract readonly outputDesc: string;
    /** Fills the CB (gFrameDim and gHas<Port> flags are set by the base class). */
    protected abstract setConstants(cb: ShaderVar, W: number, H: number, rd: RenderData): void;
    /** False when the scene camera is needed. */
    protected readonly needsScene: boolean = false;

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)) as Record<string, PropertyValue>);
    }

    override renderUI(_ui: UIWidgets): void {}

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        for (const [name, desc, optional] of this.ports) addTexInput(r, name, desc, optional);
        addColorOutput(r, "dst", this.outputDesc, w, h);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        if (!dst) return;
        for (const [name, , optional] of this.ports) if (!optional && !rd.getTexture(name)) return;
        if (this.needsScene && !this.scene) throw new RuntimeError(`${this.constructor.name}: requires a scene (camera)`);
        this.pass ??= ComputePass.create(this.device, { path: this.shaderFile });
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        cb["gFrameDim"] = [dst.width, dst.height];
        for (const [name, , optional, tex] of this.ports) {
            const t = rd.getTexture(name);
            root[tex] = t ?? dummyTexture(this.device);
            if (optional) cb[`gHas${name[0]!.toUpperCase()}${name.slice(1)}`] = t ? 1 : 0;
        }
        this.setConstants(cb, dst.width, dst.height, rd);
        for (const [name, t] of Object.entries(this.extraBindings)) root[name] = t;
        root["gDst"] = dst;
        this.pass.execute(ctx, dst.width, dst.height);
    }
}
