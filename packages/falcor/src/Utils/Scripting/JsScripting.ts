/**
 * Native JS scene and render-graph scripts, the counterparts of .pyscene files and Mogwai .py scripts.
 * A script is an ES module whose default export receives the same names `from falcor import *` gives Python
 * (plus `sceneBuilder` or `m`), so every Python script has a line-for-line JS translation. Both languages drive
 * the same bridges (createSceneBridgeModule, createMogwaiRecorder), so they build the same scenes and graphs.
 */

import type { Device } from "../../Core/API/Device.js";
import { RuntimeError } from "../../Core/Error.js";
import { AssetResolver } from "../../Core/AssetResolver.js";
import { Properties } from "../Properties.js";
import { RenderGraph } from "../../RenderGraph/RenderGraph.js";
import { createPass } from "../../RenderGraph/RenderPass.js";
import type { Scene } from "../../Scene/Scene.js";
import { kSceneBuilderFlagsPython, type SceneBuilderBridge } from "../../Scene/SceneBuilder.js";
import { float2, float3, float4 } from "../Math/Vector.js";
import {
    buildSceneFromScript,
    createConsoleMogwai,
    createMogwaiRecorder,
    getPyodide,
    type ConsoleContext,
    createSceneBridgeModule,
    getGlobalSettings,
    kLegacyEnumNames,
    kLegacyStructNames,
    settingsBinding,
    type MogwaiCommand,
    type SceneScriptOptions,
} from "./Scripting.js";
import { PySystemExit, createPyRuntime, pyNumberRepr } from "./PyRuntime.js";
import { convertPython, kAstDumper, type ConvertOptions, type ConvertResult } from "./PyToJs.js";

/** File extensions loaded as JS scene/graph modules (".ts" too, where the server compiles it, e.g. Vite). */
export const kJsScriptExtensions = [".js", ".mjs", ".ts"];

/** True if `path` names a JS script module rather than a Python script or an asset. */
export function isJsScript(path: string): boolean {
    const p = path.split(/[?#]/)[0]!.toLowerCase();
    return kJsScriptExtensions.some((ext) => p.endsWith(ext));
}

/** True for a JS scene script by name (`*.scene.js`); other JS scripts are Mogwai/graph scripts, as .py next to .pyscene. */
export function isJsSceneScript(path: string): boolean {
    return /\.scene\.(js|mjs|ts)$/i.test(path.split(/[?#]/)[0]!);
}

// ---- Python-like construction: callable with or without `new`, vectors broadcast a scalar ----

type AnyCtor = new (...args: never[]) => object;

/** Wraps a class so `C(...)` and `new C(...)` both construct (Python has no `new`); statics and instanceof carry over. */
function pyCallable<T extends AnyCtor>(cls: T, construct: (...args: never[]) => object = (...args) => new cls(...args)): T {
    const f = function (this: unknown, ...args: never[]) {
        return construct(...args);
    } as unknown as T;
    Object.setPrototypeOf(f, cls);
    Object.defineProperty(f, "prototype", { value: cls.prototype });
    Object.defineProperty(f, "name", { value: cls.name });
    return f;
}

/** Python vector construction (ScriptBindings defineVecType): (), (s), ([..]), (vec) or all components. */
function vecArgs(name: string, comps: readonly string[], args: unknown[]): number[] {
    const n = comps.length;
    let a = args;
    if (a.length === 1 && Array.isArray(a[0])) a = a[0] as unknown[];
    else if (a.length === 1 && a[0] !== null && typeof a[0] === "object") a = comps.map((c) => (a[0] as Record<string, number>)[c]);
    if (a.length === 0) a = new Array(n).fill(0);
    else if (a.length === 1) a = new Array(n).fill(a[0]);
    if (a.length !== n || a.some((v) => typeof v !== "number" && typeof v !== "boolean")) throw new TypeError(`${name}: expected 0, 1 or ${n} components`);
    return a.map(Number);
}

const kComps = ["x", "y", "z", "w"] as const;
function floatVec<T extends AnyCtor>(base: T, n: number): T {
    const comps = kComps.slice(0, n);
    return pyCallable(base, (...args: never[]) => new (base as unknown as new (...c: number[]) => object)(...vecArgs(base.name, comps, args)));
}
/** int/uint/bool vectors: TS has none, so plain component classes (the bridges read .x/.y/.z/.w). */
function componentVec(name: string, n: number, cast: (v: number) => number | boolean): AnyCtor {
    const comps = kComps.slice(0, n);
    const cls = {
        [name]: class {
            constructor(...args: unknown[]) {
                vecArgs(name, comps, args).forEach((v, i) => ((this as Record<string, unknown>)[comps[i]!] = cast(v)));
            }
            toArray(): unknown[] {
                return comps.map((c) => (this as Record<string, unknown>)[c]);
            }
        },
    }[name]!;
    return pyCallable(cls);
}

/** The vector types both script kinds get (Python's float2..bool4). */
export function scriptVectorTypes(): Record<string, unknown> {
    const out: Record<string, unknown> = { float2: floatVec(float2, 2), float3: floatVec(float3, 3), float4: floatVec(float4, 4) };
    for (const n of [2, 3, 4]) {
        out[`int${n}`] = componentVec(`int${n}`, n, (v) => Math.trunc(v));
        out[`uint${n}`] = componentVec(`uint${n}`, n, (v) => Math.trunc(v) >>> 0);
        out[`bool${n}`] = componentVec(`bool${n}`, n, (v) => !!v);
    }
    return out;
}

// ---- Scene scripts ----

/** Python's bridge guard: writing a property the bridge object doesn't have throws (instead of being dropped). */
function guarded<T extends object>(obj: T, kind: string): T {
    return new Proxy(obj, {
        get: (target, key, receiver) => (key === "_o" ? target : Reflect.get(target, key, receiver)),
        set: (target, key, value, receiver) => {
            if (typeof key === "string" && !(key in target)) throw new TypeError(`${kind}: unsupported property '${key}'`);
            return Reflect.set(target, key, value, receiver);
        },
    });
}

/** A Python-style class over a bridge factory: `new Name(args)` or `Name(args)` gives the guarded bridge object. */
function bridgeClass(name: string, make: (...args: never[]) => object, statics: Record<string, unknown> = {}): unknown {
    const f = function (...args: never[]) {
        return guarded(make(...args), name);
    };
    Object.defineProperty(f, "name", { value: name });
    return Object.assign(f, statics);
}

/** Enum objects, as the Python scene prelude defines them. */
export const kSceneEnums = {
    AlphaMode: { Opaque: 0, Mask: 1 },
    MaterialType: { Unknown: 0, Standard: 1, Cloth: 2, Hair: 3, MERL: 4, MERLMix: 5, PBRTDiffuse: 6, PBRTDiffuseTransmission: 7, PBRTConductor: 8, PBRTDielectric: 9, PBRTCoatedConductor: 10, PBRTCoatedDiffuse: 11, RGL: 12 },
    ShadingModel: { MetalRough: 0, SpecGloss: 1 },
    TriangleMeshImportFlags: { Default: 0, GenSmoothNormals: 1, JoinIdenticalVertices: 2 },
    MaterialTextureSlot: { BaseColor: "BaseColor", Specular: "Specular", Emissive: "Emissive", Normal: "Normal", Transmission: "Transmission", Displacement: "Displacement", Index: "Index" },
    CompositionOrder: { Default: 1, SRT: 1, STR: 2, RST: 3, RTS: 4, TRS: 5, TSR: 6 },
    GridVolume_EmissionMode: { Direct: 0, Blackbody: 1 },
} as const;

/** Transform options: Python's Transform(**kwargs). position + target + up make a lookAt. */
export interface TransformOptions {
    translation?: unknown;
    rotationEuler?: unknown;
    rotationEulerDeg?: unknown;
    scaling?: unknown;
    position?: unknown;
    target?: unknown;
    up?: unknown;
    order?: number;
}

/**
 * The names a JS scene script receives: everything a .pyscene gets from the scene prelude, bound to `builder`.
 * Python keyword arguments become one options object (`Transform({ translation: float3(0, 1, 0) })`).
 */
export function createSceneScriptContext(builder: SceneBuilderBridge, extras: Record<string, unknown> = {}): Record<string, unknown> {
    const b = createSceneBridgeModule(builder);
    type Vec = { x: number; y: number; z: number };
    const flags = (smoothNormals: unknown, flagsArg: unknown) => {
        // createFromFile(path, flags) or (path, smoothNormals, flags), as the Python overloads.
        if (typeof smoothNormals === "number") [flagsArg, smoothNormals] = [smoothNormals, false];
        const f = Number(flagsArg ?? 0);
        return [!!smoothNormals || (f & kSceneEnums.TriangleMeshImportFlags.GenSmoothNormals) !== 0, (f & kSceneEnums.TriangleMeshImportFlags.JoinIdenticalVertices) !== 0] as const;
    };
    const TriangleMesh = Object.assign(
        function TriangleMesh() {
            return b._TriangleMesh.createEmpty();
        },
        {
            createQuad: (size?: { x: number; y: number }) => b._TriangleMesh.createQuad(size ?? null),
            createCube: (size?: Vec) => b._TriangleMesh.createCube(size ?? null),
            createSphere: (radius = 1, segmentsU = 32, segmentsV = 32) => b._TriangleMesh.createSphere(radius, segmentsU, segmentsV),
            createDisk: (radius = 1, segments = 32) => b._TriangleMesh.createDisk(radius, segments),
            createFromFile: (path: string, smoothNormals: unknown = false, flagsArg?: number) => b._TriangleMesh.createFromFile(path, ...flags(smoothNormals, flagsArg)),
        },
    );
    const Transform = function Transform(opts: TransformOptions = {}) {
        const t = b._Transform();
        if (opts.translation != null) t.translation = opts.translation as Vec;
        if (opts.scaling != null) t.scaling = opts.scaling as Vec;
        if (opts.rotationEuler != null) t.rotationEuler = opts.rotationEuler as Vec;
        if (opts.rotationEulerDeg != null) t.rotationEulerDeg = opts.rotationEulerDeg as Vec;
        if (opts.order != null) t.order = opts.order;
        if (opts.position != null && opts.target != null && opts.up != null) t.lookAt(opts.position as Vec, opts.target as Vec, opts.up as Vec);
        return guarded(t, "Transform");
    };
    // AABB(min, max) or AABB({ min, max }) (Python also takes p / min_point / max_point).
    const AABB = function AABB(minOrOpts?: unknown, max?: unknown) {
        const o = minOrOpts && typeof minOrOpts === "object" && !("x" in minOrOpts) ? (minOrOpts as Record<string, unknown>) : null;
        const lo = o ? (o.min ?? o.min_point ?? o.p) : minOrOpts;
        return b._makeAABB(lo as Vec, (o ? (o.max ?? o.max_point) : max) as Vec);
    };
    const EnvMap = Object.assign(
        function EnvMap(path: string) {
            return b._makeEnvMap(path);
        },
        { createFromFile: (path: string) => b._makeEnvMap(path) },
    );
    const StandardMaterial = bridgeClass("StandardMaterial", b.StandardMaterial as never);
    const GridVolume = bridgeClass("GridVolume", b._GridVolume as never, { GridSlot: { Density: "density", Emission: "emission" }, EmissionMode: kSceneEnums.GridVolume_EmissionMode });
    const ctx: Record<string, unknown> = {
        sceneBuilder: builder,
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
        ...scriptVectorTypes(),
        ...kSceneEnums,
        TriangleMesh,
        Transform,
        AABB,
        EnvMap,
        Camera: bridgeClass("Camera", b.Camera as never),
        PointLight: bridgeClass("PointLight", b.PointLight as never),
        DirectionalLight: bridgeClass("DirectionalLight", b.DirectionalLight as never),
        DistantLight: bridgeClass("DistantLight", b.DistantLight as never),
        RectLight: bridgeClass("RectLight", b.RectLight as never),
        DiscLight: bridgeClass("DiscLight", b.DiscLight as never),
        SphereLight: bridgeClass("SphereLight", b.SphereLight as never),
        StandardMaterial,
        Material: StandardMaterial, // Python's deprecated alias
        ClothMaterial: bridgeClass("ClothMaterial", b.ClothMaterial as never),
        HairMaterial: bridgeClass("HairMaterial", b.HairMaterial as never),
        PBRTDiffuseMaterial: bridgeClass("PBRTDiffuseMaterial", b.PBRTDiffuseMaterial as never),
        PBRTConductorMaterial: bridgeClass("PBRTConductorMaterial", b.PBRTConductorMaterial as never),
        PBRTDiffuseTransmissionMaterial: bridgeClass("PBRTDiffuseTransmissionMaterial", b.PBRTDiffuseTransmissionMaterial as never),
        PBRTDielectricMaterial: bridgeClass("PBRTDielectricMaterial", b.PBRTDielectricMaterial as never),
        PBRTCoatedConductorMaterial: bridgeClass("PBRTCoatedConductorMaterial", b.PBRTCoatedConductorMaterial as never),
        PBRTCoatedDiffuseMaterial: bridgeClass("PBRTCoatedDiffuseMaterial", b.PBRTCoatedDiffuseMaterial as never),
        MERLMaterial: bridgeClass("MERLMaterial", b._MERLMaterial as never),
        MERLMixMaterial: bridgeClass("MERLMixMaterial", b._MERLMixMaterial as never),
        RGLMaterial: bridgeClass("RGLMaterial", b._RGLMaterial as never),
        Animation: Object.assign(
            function Animation(name: string, nodeID: number, duration: number) {
                return b._Animation(name, nodeID, duration);
            },
            { Behavior: { Constant: 0, Linear: 1, Cycle: 2, Oscillate: 3 }, InterpolationMode: { Linear: 0, Hermite: 1 } },
        ),
        GridVolume,
        Volume: GridVolume, // legacy alias
        // blendRange defaults to 3 voxels, as the native binding.
        Grid: {
            createSphere: (radius: number, voxelSize: number, blendRange = 3) => b._Grid.createSphere(radius, voxelSize, blendRange),
            createBox: (width: number, height: number, depth: number, voxelSize: number, blendRange = 3) => b._Grid.createBox(width, height, depth, voxelSize, blendRange),
            createFromFile: (path: string, gridname: string) => b._Grid.createFromFile(path, gridname),
        },
        SDFGrid: {
            createNDGrid: (narrowBandThickness = 5) => b._SDFGridCreate("ndsdf", narrowBandThickness, 7),
            createSBS: (opts: { brickWidth?: number; compressed?: boolean; defaultGridWidth?: number } = {}) =>
                b._SDFGridCreate("sbs", 5, opts.brickWidth ?? 7, !!opts.compressed, opts.defaultGridWidth ?? 256),
            createSVS: () => b._SDFGridCreate("svs", 5, 7),
            createSVO: () => b._SDFGridCreate("svo", 5, 7),
        },
        ...extras,
    };
    return ctx;
}

/** Runs a script module's entry with `ctx`; exit() (SystemExit) ends it normally, as in Python. */
async function runEntry(url: string, names: string[], ctx: Record<string, unknown>): Promise<void> {
    const entry = await loadScriptEntry(url, names);
    try {
        await entry(ctx);
    } catch (e) {
        if (!(e instanceof PySystemExit)) throw e;
    }
}

/** The `py` runtime for a script context: exec(open(p).read()) runs another JS script with the same context. */
function pyRuntimeFor(ctx: () => Record<string, unknown>, names: string[]) {
    return Object.assign(createPyRuntime(), { exec: (_ctx: unknown, url: string) => runEntry(url, names, ctx()) });
}

/** A script module's entry: its default export (or a named `scene` / `graph` export). */
async function loadScriptEntry(url: string, names: string[]): Promise<(ctx: Record<string, unknown>) => unknown> {
    // A fresh module per run, so edited scripts reload without a page reload (their own imports stay cached).
    const abs = new URL(url, globalThis.location?.href ?? "http://localhost/").href;
    // blob:/data: URLs take no query (and are fresh per file anyway).
    const fresh = /^(blob|data):/.test(abs) ? abs : `${abs}${abs.includes("?") ? "&" : "?"}webfalcor=${++moduleRuns}`;
    const mod = (await import(/* @vite-ignore */ fresh)) as Record<string, unknown>;
    const entry = [mod.default, ...names.map((n) => mod[n])].find((f) => typeof f === "function");
    if (!entry) throw new RuntimeError(`${url}: a script module must export default function (${names.join(" or ")} works too)`);
    return entry as (ctx: Record<string, unknown>) => unknown;
}
let moduleRuns = 0;

/** Loads a JS scene script (`export default ({ sceneBuilder, StandardMaterial, ... }) => { ... }`); mirrors runSceneScript. */
export async function runSceneModule(device: Device, url: string, options: SceneScriptOptions & { baseUrl?: string } = {}): Promise<Scene> {
    const res = await fetch(url);
    if (!res.ok) throw new RuntimeError(`${url}: HTTP ${res.status}`);
    const source = await res.text();
    const baseUrl = options.baseUrl ?? url.slice(0, url.lastIndexOf("/"));
    return buildSceneFromScript(device, source, baseUrl, { ...options, path: options.path ?? url }, async (builder) => {
        let ctx: Record<string, unknown> = {};
        ctx = createSceneScriptContext(builder, { sceneDir: baseUrl, scriptUrl: url, py: pyRuntimeFor(() => ctx, ["scene"]) });
        await runEntry(url, ["scene"], ctx);
    });
}

// ---- Render-graph and Mogwai scripts ----

/** Adds the JS form of `with m.profiler.event(name):` to a profiler binding: event(name, fn). */
function withProfilerEvent(profiler: unknown): unknown {
    if (!profiler) return profiler;
    const p = profiler as { begin_event(n: string): void; end_event(n: string): void };
    return Object.assign(p, {
        event: <T>(name: string, fn: () => T): T => {
            p.begin_event(name);
            try {
                return fn();
            } finally {
                p.end_event(name);
            }
        },
    });
}

/** The names a JS Mogwai script receives: Python's `falcor` module, `m`, and the deprecated `t`/`fc`/`tc` globals. */
function mogwaiContext(falcor: Record<string, unknown>, m: Record<string, unknown>, extras: Record<string, unknown>): Record<string, unknown> {
    m.profiler = withProfilerEvent(m.profiler);
    // Old render scripts' enums and options structs, as the Python shim gives them (see kLegacyEnumNames).
    const legacy: Record<string, unknown> = { CullMode: { CullNone: "None", CullFront: "Front", CullBack: "Back" } };
    for (const n of kLegacyEnumNames) legacy[n] = new Proxy({}, { get: (_t, k) => (typeof k === "string" ? k : undefined) });
    for (const n of kLegacyStructNames) legacy[n] = (opts: Record<string, unknown> = {}) => ({ ...opts });
    return { ...legacy, ...falcor, ...scriptVectorTypes(), m, t: m.clock, fc: m.frameCapture, tc: m.timingCapture, ...extras };
}

/**
 * Records a JS Mogwai script (`export default ({ m, RenderGraph, createPass }) => { ... }`); the JS counterpart of
 * recordMogwaiScript, producing the same commands. `await m.script(url)` runs another JS script with the same m.
 */
export async function recordMogwaiModule(device: Device, url: string): Promise<MogwaiCommand[]> {
    let ctx: Record<string, unknown> = {};
    const run = async (scriptUrl: string) => {
        const target = new URL(scriptUrl, new URL(url, globalThis.location?.href ?? "http://localhost/")).href;
        ctx.scriptUrl = target;
        await runEntry(target, ["graph"], ctx);
    };
    const { commands, falcor, m } = createMogwaiRecorder(device, {
        propertiesToScript: (props) => props.toJSON(),
        profilerToScript: (v) => v,
        runScript: (path) => run(path),
    });
    ctx = mogwaiContext(falcor, m, { py: pyRuntimeFor(() => ctx, ["graph"]) });
    await run(url);
    return commands;
}

/**
 * Runs a JS render-graph script and returns the graphs it adds with m.addGraph (the active one first), without
 * recording: the counterpart of runGraphScript.
 */
export async function runGraphModule(device: Device, url: string, extras: Record<string, unknown> = {}): Promise<RenderGraph[]> {
    const graphs: RenderGraph[] = [];
    const settings = settingsBinding(getGlobalSettings(), () => {
        for (const g of graphs) for (const { pass } of g.getPasses()) pass.onOptionsChange(getGlobalSettings().getOptions());
    });
    const m: Record<string, unknown> = {
        addGraph: (graph: RenderGraph) => void graphs.push(graph),
        setActiveGraph: (graph: RenderGraph) => {
            if (graphs.includes(graph)) graphs.splice(graphs.indexOf(graph), 1);
            graphs.unshift(graph);
        },
        profiler: device.profilerHook?.pythonBindings() ?? null,
        settings,
        getSettings: () => settings,
        ...settings,
        ...extras,
    };
    const falcor: Record<string, unknown> = {
        RenderGraph: (name: string) => new RenderGraph(device, name),
        createPass: (type: string, props?: Record<string, unknown> | Properties) => createPass(device, type, props instanceof Properties ? props : new Properties((props ?? {}) as Record<string, never>)),
        TextureChannelFlags: { Red: 1, Green: 2, Blue: 4, Alpha: 8, RGB: 7, RGBA: 15 },
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
    };
    let ctx: Record<string, unknown> = {};
    ctx = mogwaiContext(falcor, m, { scriptUrl: url, py: pyRuntimeFor(() => ctx, ["graph"]) });
    await runEntry(url, ["graph"], ctx);
    if (graphs.length === 0) throw new RuntimeError(`${url} did not register a graph via m.addGraph()`);
    return graphs;
}

// ---- Console and conversion in the viewer ----

/** A console result as the console prints it (Python-like for numbers, JSON for plain data). */
function consoleRepr(v: unknown): string {
    if (v === undefined) return "";
    if (typeof v === "number") return pyNumberRepr(v);
    if (typeof v === "string") return v;
    if (typeof v === "function") return `<function ${v.name || "anonymous"}>`;
    if (v !== null && typeof v === "object") {
        const ctor = (v as object).constructor?.name;
        const o = v as Record<string, unknown>;
        const comps = ["x", "y", "z", "w"].filter((c) => typeof o[c] === "number");
        if (comps.length >= 2 && ctor && ctor !== "Object") return `${ctor}(${comps.map((c) => pyNumberRepr(o[c] as number)).join(", ")})`;
        if (ctor && ctor !== "Object" && ctor !== "Array") return `<${ctor}>`;
        try {
            return JSON.stringify(v);
        } catch {
            return String(v);
        }
    }
    return String(v);
}

/**
 * The JS console (the Python console's counterpart): runs `source` with the live `m`, the falcor names and the
 * vector types in scope; an expression's value is echoed. `await` works (the command runs as an async function).
 */
export async function runConsoleCommandJs(device: Device, source: string, context: ConsoleContext): Promise<string> {
    const m = createConsoleMogwai(device, context);
    const names: Record<string, unknown> = {
        ...scriptVectorTypes(),
        createPass: (type: string, props?: Record<string, unknown>) => createPass(device, type, new Properties((props ?? {}) as Record<string, never>)),
        RenderGraph: (name: string) => new RenderGraph(device, name),
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
        py: createPyRuntime(),
        m,
    };
    const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor as new (...args: string[]) => (...a: unknown[]) => Promise<unknown>;
    const params = Object.keys(names);
    // An expression first (its value is the result), else statements echoing their last expression, else plain statements.
    const tail = source.lastIndexOf(";", source.trimEnd().endsWith(";") ? source.trimEnd().length - 2 : undefined);
    const candidates = [`return (${source}\n);`, ...(tail > 0 ? [`${source.slice(0, tail + 1)}\nreturn (${source.slice(tail + 1).replace(/;\s*$/, "")}\n);`] : []), source];
    let fn: ((...a: unknown[]) => Promise<unknown>) | null = null;
    for (const body of candidates) {
        try {
            fn = new AsyncFunction(...params, body);
            break;
        } catch (e) {
            if (body === source) throw e;
        }
    }
    return consoleRepr(await fn!(...params.map((k) => names[k])));
}

let astDumperReady = false;
/** Converts Python source to a JS script module in the browser (Python's parser is Pyodide's; see PyToJs.ts). */
export function convertPythonSource(source: string, options: ConvertOptions): ConvertResult {
    const pyodide = getPyodide() as { runPython(s: string): unknown; globals: { get(k: string): (s: string) => string } } | null;
    if (!pyodide) throw new RuntimeError("Call initScripting() first: the converter parses with Python");
    if (!astDumperReady) {
        pyodide.runPython(kAstDumper);
        astDumperReady = true;
    }
    const dump = pyodide.globals.get("_py2js_dump");
    return convertPython(source, (src) => JSON.parse(dump(src)) as ReturnType<Parameters<typeof convertPython>[1]>, options);
}
