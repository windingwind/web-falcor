/**
 * Python scripting mirroring Falcor/Utils/Scripting (pybind11 -> Pyodide,
 * user decision docs §11.1: the .py path is primary).
 *
 * Executes unmodified upstream render-graph .py scripts: a JS `falcor` bridge
 * module provides RenderGraph/createPass, and a Mogwai-like `m` global
 * captures addGraph calls.
 */

import { AABB } from "../Math/AABB.js";
import type { Device } from "../../Core/API/Device.js";
import { RenderGraph } from "../../RenderGraph/RenderGraph.js";
import { Settings } from "../Settings.js";
import { buildSceneFromCache, encodeTextureSources, loadSceneCache, sceneCacheKey, snapshotCameras, snapshotGridVolumes, storeSceneCache } from "../../Scene/SceneCache.js";
import { createPass, type RenderPass } from "../../RenderGraph/RenderPass.js";
import { Properties } from "../Properties.js";
import { RuntimeError } from "../../Core/Error.js";
import { AssetResolver, withScriptSearchPath } from "../../Core/AssetResolver.js";
import { AnimationBridge, CameraBridge, GridVolumeBridge, LightBridge, MaterialBridge, SceneBuilderBridge, SceneBuilderFlags, SDFGridBridge, TransformBridge, TriangleMesh, kSceneBuilderFlagsPython, makeTransform } from "../../Scene/SceneBuilder.js";
import type { Scene } from "../../Scene/Scene.js";
import { LightType, type StaticVertex } from "../../Scene/SceneData.js";
import { MaterialType, ShadingModel } from "../../Scene/Material/MaterialData.js";
import { buildSphereGrid, buildBoxGrid, parsedGridStats, parsedGridValue, type ParsedFloatGrid } from "../../Scene/Volume/VDBLoader.js";

/** Python `Grid` from Grid.createSphere/createBox: native's read-only stats and getValue. */
class ProceduralGridHandle {
    private stats: ReturnType<typeof parsedGridStats> | null = null;
    constructor(readonly _proceduralGrid: ParsedFloatGrid) {}
    private get s() {
        return (this.stats ??= parsedGridStats(this._proceduralGrid));
    }
    get voxelCount(): number { return this.s.voxelCount; }
    /** Rounded down to an 8-brick, as Grid::getMinIndex. */
    get minIndex(): { x: number; y: number; z: number } { const m = this.s.minIndex; return { x: m[0] & ~7, y: m[1] & ~7, z: m[2] & ~7 }; }
    /** Rounded up to an 8-brick, as Grid::getMaxIndex. */
    get maxIndex(): { x: number; y: number; z: number } { const m = this.s.maxIndex; return { x: (m[0] + 7) & ~7, y: (m[1] + 7) & ~7, z: (m[2] + 7) & ~7 }; }
    get minValue(): number { return this.s.minValue; }
    get maxValue(): number { return this.s.maxValue; }
    getValue(ijk: { x: number; y: number; z: number }): number {
        return parsedGridValue(this._proceduralGrid, Number(ijk.x), Number(ijk.y), Number(ijk.z));
    }
}
import { float2, float3, float4 } from "../Math/Vector.js";

interface PyodideApi {
    registerJsModule(name: string, module: object): void;
    runPython(code: string, options?: { globals?: unknown }): unknown;
    globals: { set(name: string, value: unknown): void; get(name: string): unknown };
    toPy(obj: unknown): unknown;
}

let pyodide: PyodideApi | null = null;

/** Where scripts/setup-web.mjs puts the pinned Pyodide packages. */
export const kPyodidePackagesUrl = "/tools/pyodide-packages/";

/** Loads Pyodide (idempotent). indexURL points at the pyodide distribution. */
export async function initScripting(indexURL: string): Promise<void> {
    if (pyodide) return;
    const mod = (await import(/* @vite-ignore */ `${indexURL}/pyodide.mjs`)) as {
        loadPyodide(options: { indexURL: string; packageBaseUrl?: string }): Promise<PyodideApi>;
    };
    // Packages (numpy) come from tools/pyodide-packages/, provisioned by scripts/setup-web.mjs.
    pyodide = await mod.loadPyodide({ indexURL, packageBaseUrl: new URL(kPyodidePackagesUrl, globalThis.location?.href ?? "http://localhost/").href });
}

/** Shared Settings instance (native: SampleApp::getSettings()). */
const globalSettings = new Settings();

export function getGlobalSettings(): Settings {
    return globalSettings;
}

/** @internal The Pyodide instance (for the Testbed scripting layer). */
export function getPyodide(): unknown {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    return pyodide;
}

export function isScriptingInitialized(): boolean {
    return pyodide !== null;
}

/** SampleApp analog: `searchpath:media` settings feed the default asset resolver. */
function applyMediaSearchPaths(): void {
    for (const p of globalSettings.getSearchDirectories("media")) AssetResolver.getDefaultResolver().addSearchPath(p);
}

/** Converts a PyProxy (dict/list) or primitive into plain JS. */
function toJs(value: unknown): unknown {
    const proxy = value as { toJs?: (opts: object) => unknown };
    if (proxy && typeof proxy.toJs === "function") {
        return proxy.toJs({ dict_converter: Object.fromEntries, create_proxies: false });
    }
    return value;
}

/**
 * Mirrors Mogwai's scripting surface: executes a graph script and returns the
 * graphs registered via m.addGraph() (plus any RenderGraph left in globals).
 */
/** `extras` adds Mogwai extension objects to `m` (e.g. the viewer's frameCapture). */
export async function runGraphScript(device: Device, source: string, extras: Record<string, unknown> = {}): Promise<RenderGraph[]> {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    const graphs: RenderGraph[] = [];

    const falcorModule = {
        RenderGraph: (name: string) => new RenderGraph(device, name),
        createPass: (type: string, props?: unknown) =>
            createPass(device, type, new Properties((toJs(props) as Record<string, never>) ?? {})),
        // Output-channel marker flags (markOutput's optional second argument).
        TextureChannelFlags: { Red: 1, Green: 2, Blue: 4, Alpha: 8, RGB: 7, RGBA: 15 },
        // Vector factories for pass properties (e.g. SimplePostFX saturationCurve).
        // Factories, not classes: Pyodide can't `new` JS classes from Python.
        float2: (x = 0, y = 0) => new float2(x, y),
        float3: (x = 0, y = 0, z = 0) => new float3(x, y, z),
        float4: (x = 0, y = 0, z = 0, w = 0) => new float4(x, y, z, w),
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
    };
    pyodide.registerJsModule("falcor", falcorModule);

    // Mirrors the native Settings script binding (m.settings / m.getSettings()).
    const settings = {
        addOptions: (dict: unknown) => {
            globalSettings.addOptions(toJs(dict) as Record<string, never>);
            applyMediaSearchPaths();
            for (const g of graphs) for (const { pass } of g.getPasses()) pass.onOptionsChange(globalSettings.getOptions());
        },
        addFilteredAttributes: (dictOrList: unknown) => globalSettings.addFilteredAttributes(toJs(dictOrList) as Record<string, never>),
        clearOptions: () => globalSettings.clearOptions(),
        clearFilteredAttributes: () => globalSettings.clearFilteredAttributes(),
    };
    const mogwai = {
        addGraph: (graph: RenderGraph) => {
            graphs.push(graph);
        },
        // Mirrors the native Profiler script binding (m.profiler).
        profiler: device.profilerHook?.pythonBindings((v) => pyodide!.toPy(v)) ?? null,
        settings,
        getSettings: () => settings,
        ...extras,
    };
    pyodide.globals.set("m", mogwai);
    pyodide.runPython(kMogwaiShim);

    // Import falcor afresh: registerJsModule doesn't replace an already imported module.
    pyodide.runPython(`import sys\nsys.modules.pop("falcor", None)`);
    pyodide.runPython(source);

    if (graphs.length === 0) throw new RuntimeError("Graph script did not register a graph via m.addGraph()");
    return graphs;
}

/**
 * Interactive python console (mirrors Mogwai's console): runs a snippet with
 * `m` bound to the LIVE viewer state — `m.scene` is the real Scene (edits via
 * getLight/updateLights, getMaterial/updateMaterial, camera), `m.activeGraph`
 * the running graph (getPass/markOutput), `m.settings` the global Settings.
 * The last expression's repr and print() output are returned (native echoes
 * the same way). Vector factories come from `from falcor import *`.
 */
export function runConsoleCommand(
    device: Device,
    source: string,
    context: { scene: Scene | null; graph: RenderGraph | null; clock?: unknown; timingCapture?: unknown; frameCapture?: unknown; profiler?: import("../../Core/API/Profiler.js").Profiler | null },
): string {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    const lines: string[] = [];
    pyodide.registerJsModule("falcor", {
        createPass: (type: string, props?: unknown) => createPass(device, type, new Properties((toJs(props) as Record<string, never>) ?? {})),
        float2: (x = 0, y = 0) => new float2(x, y),
        float3: (x = 0, y = 0, z = 0) => new float3(x, y, z),
        float4: (x = 0, y = 0, z = 0, w = 0) => new float4(x, y, z, w),
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
    });
    const settings = {
        addOptions: (dict: unknown) => {
            globalSettings.addOptions(toJs(dict) as Record<string, never>);
            applyMediaSearchPaths();
        },
    };
    pyodide.globals.set("m", {
        scene: context.scene,
        activeGraph: context.graph,
        clock: context.clock,
        timingCapture: context.timingCapture,
        frameCapture: context.frameCapture,
        profiler: (context.profiler ?? device.profilerHook)?.pythonBindings((v) => pyodide!.toPy(v)) ?? null,
        settings,
        getSettings: () => settings,
    });
    const py = pyodide as unknown as { setStdout(opts: { batched: (s: string) => void }): void; runPython(src: string): unknown };
    py.setStdout({ batched: (s) => lines.push(s) });
    try {
        py.runPython(kMogwaiShim);
        const result = py.runPython('import sys\nsys.modules.pop("falcor", None)\nfrom falcor import *\n' + source);
        // Echo like python's repr for the scalars pyodide converts to JS.
        if (result !== undefined && result !== null) lines.push(typeof result === "boolean" ? (result ? "True" : "False") : String(result));
    } finally {
        py.setStdout({ batched: (s) => console.log(s) });
    }
    return lines.join("\n");
}

/**
 * Wraps the script's `m` so `m.profiler.event(name)` is native's ProfilerEvent context manager
 * (a JS object can't be used in `with`); everything else passes through to the JS object.
 */
const kMogwaiShim = `
class _PyProfilerEvent:
    def __init__(self, p, name): self._p, self._n = p, name
    def __enter__(self): self._p.begin_event(self._n); return self
    def __exit__(self, *args): self._p.end_event(self._n); return False
class _MogwaiProfiler:
    def __init__(self, p): object.__setattr__(self, '_p', p)
    def __getattr__(self, k): return getattr(object.__getattribute__(self, '_p'), k)
    def __setattr__(self, k, v): setattr(object.__getattribute__(self, '_p'), k, v)
    def event(self, name): return _PyProfilerEvent(object.__getattribute__(self, '_p'), name)
class _Mogwai:
    def __init__(self, o): object.__setattr__(self, '_o', o)
    def __getattr__(self, k):
        v = getattr(object.__getattribute__(self, '_o'), k)
        return _MogwaiProfiler(v) if k == 'profiler' and v is not None else v
    def __setattr__(self, k, v): setattr(object.__getattribute__(self, '_o'), k, v)
if not isinstance(m, _Mogwai): m = _Mogwai(m)
`;

/** Python prelude adapting pythonic pyscene API (kwargs, class-style ctors)
 *  to the JS SceneBuilder bridge. */
const kScenePrelude = `
import sys
sys.modules.pop('webfalcor_scene', None)  # registerJsModule per call; defeat import caching
from webfalcor_scene import (sceneBuilder, SceneBuilderFlags, _TriangleMesh,
    PointLight, DirectionalLight, DistantLight, RectLight, DiscLight, SphereLight,
    StandardMaterial, ClothMaterial, HairMaterial,
    PBRTDiffuseMaterial, PBRTConductorMaterial, PBRTDiffuseTransmissionMaterial, PBRTDielectricMaterial,
    PBRTCoatedConductorMaterial, PBRTCoatedDiffuseMaterial, _MERLMaterial, _MERLMixMaterial, _RGLMaterial,
    Camera, _makeTransform, _makeAABB, _makeEnvMap, _GridVolume, _Grid, _SDFGridCreate, _Transform, _Animation)

# Python-side vector types with arithmetic (upstream pyscenes do e.g. size / 2);
# the JS bridge reads .x/.y/.z/.w off any object.
class float2:
    def __init__(self, x=0.0, y=None):
        self.x = float(x); self.y = float(x if y is None else y)
class float3:
    def __init__(self, x=0.0, y=None, z=None):
        if y is None: y = z = x
        self.x = float(x); self.y = float(y); self.z = float(z)
    def _map(self, other, op):
        if isinstance(other, float3):
            return float3(op(self.x, other.x), op(self.y, other.y), op(self.z, other.z))
        return float3(op(self.x, other), op(self.y, other), op(self.z, other))
    def __add__(self, o): return self._map(o, lambda a, b: a + b)
    def __sub__(self, o): return self._map(o, lambda a, b: a - b)
    def __mul__(self, o): return self._map(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._map(o, lambda a, b: a * b)
    def __truediv__(self, o): return self._map(o, lambda a, b: a / b)
    def __neg__(self): return float3(-self.x, -self.y, -self.z)
class float4:
    def __init__(self, x=0.0, y=None, z=None, w=None):
        if y is None: y = z = w = x
        self.x = float(x); self.y = float(y); self.z = float(z); self.w = float(w)

# Integer and bool vectors (native int2..4, uint2..4, bool2..4).
def _vec_class(name, n, cast):
    comps = 'xyzw'[:n]
    def __init__(self, *args):
        vals = list(args) if len(args) == n else [args[0] if args else 0] * n
        for c, v in zip(comps, vals): setattr(self, c, cast(v))
    def __repr__(self): return f"{name}({', '.join(str(getattr(self, c)) for c in comps)})"
    def __eq__(self, o): return all(getattr(self, c) == getattr(o, c, None) for c in comps)
    return type(name, (), {'__init__': __init__, '__repr__': __repr__, '__eq__': __eq__})
for _n in (2, 3, 4):
    globals()[f'int{_n}'] = _vec_class(f'int{_n}', _n, int)
    globals()[f'uint{_n}'] = _vec_class(f'uint{_n}', _n, int)
    globals()[f'bool{_n}'] = _vec_class(f'bool{_n}', _n, bool)

class GridVolume_EmissionMode:
    Direct = 0
    Blackbody = 1

class TriangleMesh:
    def __new__(cls):
        return _TriangleMesh.createEmpty()
    @staticmethod
    def createQuad(size=None):
        return _TriangleMesh.createQuad(size)
    @staticmethod
    def createCube(size=None):
        return _TriangleMesh.createCube(size)
    @staticmethod
    def createSphere(radius=1.0, segmentsU=32, segmentsV=32):
        return _TriangleMesh.createSphere(radius, segmentsU, segmentsV)
    @staticmethod
    def createDisk(radius=1.0, segments=32):
        return _TriangleMesh.createDisk(radius, segments)
    @staticmethod
    def createFromFile(path, smoothNormals=False, flags=None):
        return _TriangleMesh.createFromFile(path, smoothNormals)

class CompositionOrder:
    Default = 1
    SRT = 1
    STR = 2
    RST = 3
    RTS = 4
    TRS = 5
    TSR = 6

def Transform(translation=None, rotationEuler=None, rotationEulerDeg=None, scaling=None, position=None, target=None, up=None, order=None):
    # Mirrors the Transform(**kwargs) binding: position/target/up together make a lookAt.
    t = _Transform()
    if translation is not None: t.translation = translation
    if scaling is not None: t.scaling = scaling
    if rotationEuler is not None: t.rotationEuler = rotationEuler
    if rotationEulerDeg is not None: t.rotationEulerDeg = rotationEulerDeg
    if order is not None: t.order = order
    if position is not None and target is not None and up is not None: t.lookAt(position, target, up)
    return t

def AABB(min=None, max=None):
    return _makeAABB(min, max)

class EnvMap:
    @staticmethod
    def createFromFile(path):
        return _makeEnvMap(path)
    def __new__(cls, path):
        return _makeEnvMap(path)

# Bridge reads return JS vectors: hand them back as the prelude's float3/float4 (with arithmetic).
def _pyvec(v):
    try:
        from pyodide.ffi import JsProxy
    except ImportError:
        return v
    if isinstance(v, JsProxy) and not callable(v) and all(hasattr(v, c) for c in 'xyz'):
        return float4(v.x, v.y, v.z, v.w) if hasattr(v, 'w') else float3(v.x, v.y, v.z)
    return v

# Guard: python setattr on JS proxies silently creates properties, so a typo'd
# or unimplemented bridge property would be DROPPED. Wrap the factories so
# unknown attribute writes raise instead (mirrors pybind11 strictness).
def _guarded(factory, known, kwnames=()):
    def make(*args, **kwargs):
        # Keyword arguments map onto the factory's positional parameters.
        args = list(args)
        for i, k in enumerate(kwnames):
            if k in kwargs:
                while len(args) < i: args.append(None)
                if len(args) == i: args.append(kwargs.pop(k))
                else: args[i] = kwargs.pop(k)
        if kwargs:
            raise TypeError(f'unexpected keyword arguments: {sorted(kwargs)} (web bridge)')
        obj = factory(*args)
        class Guard:
            __slots__ = ('_o',)
            def __init__(self, o): object.__setattr__(self, '_o', o)
            def __getattr__(self, k): return _pyvec(getattr(object.__getattribute__(self, '_o'), k))
            def __setattr__(self, k, v):
                if k not in known:
                    raise AttributeError(f'unsupported property: {k} (web bridge)')
                setattr(object.__getattribute__(self, '_o'), k, v)
        return Guard(obj)
    return make

_matProps = {'baseColor', 'specularParams', 'transmissionColor', 'emissiveColor',
             'emissiveFactor', 'doubleSided', 'roughness', 'metallic',
             'indexOfRefraction', 'specularTransmission', 'diffuseTransmission', 'thinSurface',
             'nestedPriority', 'volumeAbsorption', 'volumeScattering', 'volumeAnisotropy', 'alphaMode', 'alphaThreshold',
             'displacementScale', 'displacementOffset', 'lightProfileEnabled'}
_lightProps = {'position', 'intensity', 'direction', 'angle',
               'openingAngle', 'penumbraAngle', 'scaling', 'rotation'}
_camProps = {'position', 'target', 'up', 'focalLength', 'focalDistance', 'apertureRadius', 'shutterSpeed', 'ISOSpeed'}
StandardMaterial = _guarded(StandardMaterial, _matProps, ('name', 'model'))
Material = StandardMaterial  # PYTHONDEPRECATED alias (upstream SDF/legacy pyscenes)
ClothMaterial = _guarded(ClothMaterial, _matProps)
HairMaterial = _guarded(HairMaterial, _matProps)
PBRTDiffuseMaterial = _guarded(PBRTDiffuseMaterial, _matProps)

class MERLMaterial:
    """Measured MERL BRDF (Scene/Material/MERLMaterial): MERLMaterial(name, path)."""
    def __init__(self, name='', path=''):
        self._o = _MERLMaterial(name, path)
    def __getattr__(self, k): return getattr(object.__getattribute__(self, '_o'), k)

class MERLMixMaterial:
    """MERL BRDFs selected per texel (Scene/Material/MERLMixMaterial): MERLMixMaterial(name, paths).

    Load the selector with .loadTexture(MaterialTextureSlot.Index, path).
    """
    def __init__(self, name='', paths=()):
        self._o = _MERLMixMaterial(name, list(paths))
    def __getattr__(self, k): return getattr(object.__getattribute__(self, '_o'), k)

class RGLMaterial:
    """Measured RGL BSDF (Scene/Material/RGLMaterial): RGLMaterial(name, path) or .load(path)."""
    def __init__(self, name='', path=''):
        self._o = _RGLMaterial(name, path)
    def __getattr__(self, k): return getattr(object.__getattribute__(self, '_o'), k)
PBRTConductorMaterial = _guarded(PBRTConductorMaterial, _matProps)
PBRTDiffuseTransmissionMaterial = _guarded(PBRTDiffuseTransmissionMaterial, _matProps)
PBRTDielectricMaterial = _guarded(PBRTDielectricMaterial, _matProps)
PBRTCoatedConductorMaterial = _guarded(PBRTCoatedConductorMaterial, _matProps)
PBRTCoatedDiffuseMaterial = _guarded(PBRTCoatedDiffuseMaterial, _matProps)
PointLight = _guarded(PointLight, _lightProps)
DirectionalLight = _guarded(DirectionalLight, _lightProps)
DistantLight = _guarded(DistantLight, _lightProps)
RectLight = _guarded(RectLight, _lightProps)
DiscLight = _guarded(DiscLight, _lightProps)
SphereLight = _guarded(SphereLight, _lightProps)
Camera = _guarded(Camera, _camProps)

# Animation behavior enum (values mirror native Animation::Behavior; applied to
# imported clips via sceneBuilder.animations[i].pre/postInfinityBehavior).
class Animation:
    class Behavior:
        Constant = 0
        Linear = 1
        Cycle = 2
        Oscillate = 3
    class InterpolationMode:
        Linear = 0
        Hermite = 1
    # Animation(name, nodeID, duration): scripted keyframes (sceneBuilder.addAnimation).
    def __new__(cls, name, nodeID, duration):
        return _Animation(name, nodeID, duration)

class AlphaMode:
    Opaque = 0
    Mask = 1

class MaterialType:
    Unknown = 0
    Standard = 1
    Cloth = 2
    Hair = 3
    MERL = 4
    MERLMix = 5
    PBRTDiffuse = 6
    PBRTDiffuseTransmission = 7
    PBRTConductor = 8
    PBRTDielectric = 9
    PBRTCoatedConductor = 10
    PBRTCoatedDiffuse = 11
    RGL = 12

# Enums accepted for parity (values map to the web material/import defaults).
class ShadingModel:
    MetalRough = 0
    SpecGloss = 1
class TriangleMeshImportFlags:
    Default = 0
    GenSmoothNormals = 1
    JoinIdenticalVertices = 2
class MaterialTextureSlot:
    BaseColor = 'BaseColor'
    Specular = 'Specular'
    Emissive = 'Emissive'
    Normal = 'Normal'
    Transmission = 'Transmission'
    Displacement = 'Displacement'
    Index = 'Index'

# Grid volumes (smoke.pyscene etc.); GridSlot values are bridge slot strings.
class _GridSlot:
    Density = 'density'
    Emission = 'emission'
_gvProps = {'name', 'densityScale', 'emissionScale', 'albedo', 'anisotropy',
            'emissionMode', 'emissionTemperature', 'densityGrid', 'emissionGrid',
            'frameRate', 'startFrame', 'playbackEnabled'}
_GridVolumeGuarded = _guarded(_GridVolume, _gvProps)
class GridVolume:
    GridSlot = _GridSlot
    EmissionMode = GridVolume_EmissionMode
    def __new__(cls, name=''):
        return _GridVolumeGuarded(name)
Volume = GridVolume  # legacy pyscene alias (volume_test.pyscene)

# Procedural density grids (two_volumes.pyscene).
class Grid:
    # blendRange defaults to 3 voxels, as the native binding.
    @staticmethod
    def createSphere(radius, voxelSize, blendRange=3.0):
        return _Grid.createSphere(radius, voxelSize, blendRange)
    @staticmethod
    def createBox(width, height, depth, voxelSize, blendRange=3.0):
        return _Grid.createBox(width, height, depth, voxelSize, blendRange)
    @staticmethod
    def createFromFile(path, gridname):
        return _Grid.createFromFile(path, gridname)

# SDF grids: all four representations, built from the procedural generator,
# a .sdfg corner-value file or a .sdf primitive list.
class SDFGrid:
    @staticmethod
    def createNDGrid(narrowBandThickness=5.0):
        return _SDFGridCreate('ndsdf', narrowBandThickness, 7)
    @staticmethod
    def createSBS(brickWidth=7, compressed=False, defaultGridWidth=256):
        return _SDFGridCreate('sbs', 5.0, brickWidth, bool(compressed), defaultGridWidth)
    @staticmethod
    def createSVS(**kwargs):
        return _SDFGridCreate('svs', 5.0, 7)
    @staticmethod
    def createSVO(**kwargs):
        return _SDFGridCreate('svo', 5.0, 7)
`;

/**
 * Executes an unmodified .pyscene through the SceneBuilder bridge and
 * resolves the resulting scene (assets fetched relative to baseUrl).
 */
let sceneLoadedFromCache = false;

/** Whether the last runSceneScript call was served from the scene cache. */
export function wasSceneLoadedFromCache(): boolean {
    return sceneLoadedFromCache;
}

/** Scene load options: `cache` (or Flags::UseCache) enables the OPFS scene cache; `flags` mirrors SceneBuilder::Flags. */
export interface SceneScriptOptions {
    cache?: boolean;
    flags?: SceneBuilderFlags | number;
    /** The script's own file; heads Scene.importPaths as SceneBuilder(path) does natively. */
    path?: string;
}

export async function runSceneScript(device: Device, source: string, baseUrl: string, options?: SceneScriptOptions): Promise<Scene> {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    sceneLoadedFromCache = false;
    const scene = await withScriptSearchPath(baseUrl, () => runSceneScriptInternal(device, source, baseUrl, options));
    if (options?.path) scene.importPaths.unshift(options.path);
    return scene;
}

async function runSceneScriptInternal(device: Device, source: string, baseUrl: string, options?: SceneScriptOptions): Promise<Scene> {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    const flags = (options?.flags ?? SceneBuilderFlags.Default) as number;
    const useCache = options?.cache || (flags & SceneBuilderFlags.UseCache) !== 0 || (flags & SceneBuilderFlags.RebuildCache) !== 0;
    const rebuildCache = (flags & SceneBuilderFlags.RebuildCache) !== 0;
    let cacheKey: string | null = null;
    if (useCache) {
        // Native keys the cache on the build flags too (cache/rebuild bits excluded).
        cacheKey = await sceneCacheKey(`${source}\n#flags=${flags & ~(SceneBuilderFlags.UseCache | SceneBuilderFlags.RebuildCache)}`);
        const cached = rebuildCache ? null : await loadSceneCache(cacheKey);
        if (cached) {
            sceneLoadedFromCache = true;
            return await buildSceneFromCache(device, cached);
        }
    }
    const builder = new SceneBuilderBridge(flags);

    type VecLike = { x: number; y: number; z: number };
    const sceneModule = {
        sceneBuilder: builder,
        // Pyodide calls JS classes without `new`; vectors live python-side (prelude).
        _TriangleMesh: {
            createQuad: (size?: { x: number; y: number } | null) => TriangleMesh.createQuad(size ? new float2(size.x, size.y) : undefined),
            createCube: (size?: VecLike | null) => TriangleMesh.createCube(size ? new float3(size.x, size.y, size.z) : undefined),
            createSphere: (radius?: number, segmentsU?: number, segmentsV?: number) => TriangleMesh.createSphere(radius, segmentsU, segmentsV),
            createDisk: (radius?: number, segments?: number) => TriangleMesh.createDisk(radius, segments),
            createFromFile: (path: string, smoothNormals?: boolean) => TriangleMesh.createFromFile(String(path), !!smoothNormals),
            // Mutable builder: TriangleMesh() then addVertex()/addTriangle() (tutorial.pyscene).
            createEmpty: () => {
                const vertices: StaticVertex[] = [];
                const indexList: number[] = [];
                return {
                    vertices,
                    get indices() {
                        return new Uint32Array(indexList);
                    },
                    addVertex(position: VecLike, normal: VecLike, texCrd?: { x: number; y: number } | null) {
                        vertices.push({
                            position: new float3(position.x, position.y, position.z),
                            normal: new float3(normal.x, normal.y, normal.z),
                            tangent: new float4(0, 0, 0, 0),
                            texCrd: texCrd ? new float2(texCrd.x, texCrd.y) : new float2(0, 0),
                        });
                        return vertices.length - 1;
                    },
                    addTriangle(i0: number, i1: number, i2: number) {
                        indexList.push(Number(i0), Number(i1), Number(i2));
                    },
                };
            },
        },
        Camera: (name = "") => new CameraBridge(String(name)),
        PointLight: (name = "") => new LightBridge(LightType.Point, name),
        DirectionalLight: (name = "") => new LightBridge(LightType.Directional, name),
        DistantLight: (name = "") => new LightBridge(LightType.Distant, name),
        RectLight: (name = "") => new LightBridge(LightType.Rect, name),
        DiscLight: (name = "") => new LightBridge(LightType.Disc, name),
        SphereLight: (name = "") => new LightBridge(LightType.Sphere, name),
        // Mirrors StandardMaterial(name, model = ShadingModel.MetalRough).
        StandardMaterial: (name = "", model = 0) => new MaterialBridge(MaterialType.Standard, name, Number(model) as ShadingModel),
        ClothMaterial: (name = "") => new MaterialBridge(MaterialType.Cloth, name),
        HairMaterial: (name = "") => new MaterialBridge(MaterialType.Hair, name),
        PBRTDiffuseMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTDiffuse, name),
        PBRTConductorMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTConductor, name),
        PBRTDiffuseTransmissionMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTDiffuseTransmission, name),
        PBRTDielectricMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTDielectric, name),
        PBRTCoatedConductorMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTCoatedConductor, name),
        PBRTCoatedDiffuseMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTCoatedDiffuse, name),
        _MERLMaterial: (name = "", path = "") => {
            const m = new MaterialBridge(MaterialType.MERL, name);
            if (path) m.load(path);
            return m;
        },
        _MERLMixMaterial: (name = "", paths: unknown = []) => {
            const m = new MaterialBridge(MaterialType.MERLMix, name);
            // Python lists arrive as proxies, so iterate rather than Array.isArray.
            const list = typeof paths === "string" ? [paths] : (paths as Iterable<unknown>);
            for (const path of list ?? []) m.load(path);
            return m;
        },
        _RGLMaterial: (name = "", path = "") => {
            const m = new MaterialBridge(MaterialType.RGL, name);
            if (path) m.load(path);
            return m;
        },
        _makeTransform: makeTransform,
        _Transform: () => new TransformBridge(),
        _Animation: (name: string, nodeID: number, duration: number) => new AnimationBridge(String(name), Number(nodeID), Number(duration)),
        _makeAABB: (min?: VecLike, max?: VecLike) => new AABB(min, max),
        _makeEnvMap: (path: string) => ({ path, intensity: 1 }),
        _GridVolume: (name = "") => new GridVolumeBridge(name),
        _Grid: {
            createSphere: (radius: number, voxelSize: number, blendRange = 3) => new ProceduralGridHandle(buildSphereGrid(Number(radius), Number(voxelSize), Number(blendRange))),
            createBox: (width: number, height: number, depth: number, voxelSize: number, blendRange = 3) =>
                new ProceduralGridHandle(buildBoxGrid(Number(width), Number(height), Number(depth), Number(voxelSize), Number(blendRange))),
            // Loaded with the scene (fetches are asynchronous); stats aren't available in the script.
            createFromFile: (path: string, gridname: string) => ({ _file: { path: String(path), gridname: String(gridname) } }),
        },
        _SDFGridCreate: (type: string, narrowBandThickness = 5.0, brickWidth = 7, compressed = false, defaultGridWidth = 256) =>
            new SDFGridBridge(type as "ndsdf" | "sbs", Number(narrowBandThickness), Number(brickWidth), !!compressed, Number(defaultGridWidth)),
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
    };
    pyodide.registerJsModule("webfalcor_scene", sceneModule);

    // Native runs scene scripts as files: define __file__ and provide their local imports.
    const sceneDir = `/mogwai${new URL(`${baseUrl.startsWith("http") ? baseUrl : location.origin + (baseUrl.startsWith("/") ? "" : "/") + baseUrl}/`).pathname.replace(/\/$/, "")}`;
    if (/^\s*(from|import)\s/m.test(source)) writePythonFiles(await fetchLocalPythonModules(baseUrl, source, "/mogwai"));
    pyodide.globals.set("__file__", `${sceneDir}/scene.pyscene`);
    // Modules the scene imports do `from falcor import *` and expect the scene API, as natively.
    const exposeSceneApi = `
import types as _types
_prev_falcor = sys.modules.get("falcor")
_scene_falcor = _types.ModuleType("falcor")
if _prev_falcor is not None:
    for _k in dir(_prev_falcor):
        if not _k.startswith("__"): setattr(_scene_falcor, _k, getattr(_prev_falcor, _k))
for _k, _v in list(globals().items()):
    if not _k.startswith("_") and _k != "sys": setattr(_scene_falcor, _k, _v)
sys.modules["falcor"] = _scene_falcor
`;
    try {
        pyodide.runPython(kScenePrelude + "\n" + exposeSceneApi + "\n" + source);
    } finally {
        pyodide.runPython(`import sys\nif globals().get("_prev_falcor") is not None: sys.modules["falcor"] = _prev_falcor\nelse: sys.modules.pop("falcor", None)`);
    }

    const scene = await builder.resolve(device, baseUrl);
    const env = scene.getEnvMap();
    // Programmatic env maps without retained source bytes can't be restored.
    if (cacheKey && builder.lastSceneArgs?.cacheable && (!env || env.sourceBytes)) {
        const { meshes, materials, lights, nodes, cameraNodeID, textureManager, curves, animations, weightTracks, sdfGrids } = builder.lastSceneArgs;
        const textures = await encodeTextureSources(textureManager);
        await storeSceneCache(cacheKey, {
            meshes,
            materials,
            lights,
            nodes,
            cameraNodeID,
            ...snapshotCameras(scene),
            textures,
            curves,
            envMap: env?.sourceBytes
                ? { bytes: env.sourceBytes, isExr: env.sourceIsExr, intensity: env.intensity, tint: env.tint, rotationDeg: env.rotationDeg, equalAreaOctahedral: env.sourceEqualAreaOctahedral }
                : undefined,
            animations,
            weightTracks,
            sdfGrids,
            gridVolumes: snapshotGridVolumes(scene),
            customPrimitives: Array.from({ length: scene.getCustomPrimitiveCount() }, (_v, i) => ({
                userID: scene.getCustomPrimitive(i).userID,
                aabb: scene.getCustomPrimitiveAABB(i),
            })),
        });
    }
    return scene;
}

/**
 * Fetches the Python modules a script imports from its own directory (and
 * sys.path.append'ed ones): `from X.Y import` / `import X` become files under
 * `root` + their URL path, with package __init__.py files. Pyodide can only
 * import what is in its file system.
 */
export async function fetchLocalPythonModules(dirUrl: string, source: string, root: string): Promise<Record<string, string>> {
    const files: Record<string, string> = {};
    const base = dirUrl.startsWith("http") ? dirUrl : `${location.origin}${dirUrl.startsWith("/") ? "" : "/"}${dirUrl}`;
    const dir = new URL(`${base}/`).pathname.replace(/\/$/, "");
    const searchDirs = [dir];
    for (const m of source.matchAll(/sys\.path\.append\(\s*['"]([^'"]+)['"]\s*\)/g)) searchDirs.push(new URL(m[1]!, `${location.origin}${dir}/`).pathname.replace(/\/$/, ""));
    const pending: string[] = [source];
    const seen = new Set<string>();
    while (pending.length > 0) {
        const text = pending.pop()!;
        for (const m of text.matchAll(/^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))/gm)) {
            const mod = m[1] ?? m[2]!;
            if (seen.has(mod) || ["falcor", "sys", "os", "math", "random", "json"].includes(mod)) continue;
            seen.add(mod);
            for (const d of searchDirs) {
                const url = `${d}/${mod.replaceAll(".", "/")}.py`;
                if (files[`${root}${url}`] !== undefined) break;
                const res = await fetch(url);
                if (!res.ok || res.headers.get("content-type")?.includes("html")) continue;
                const body = await res.text();
                files[`${root}${url}`] = body;
                const parts = mod.split(".");
                for (let i = 1; i < parts.length; i++) files[`${root}${d}/${parts.slice(0, i).join("/")}/__init__.py`] ??= "";
                pending.push(body);
                break;
            }
        }
    }
    // Files the scripts read themselves, e.g. exec(open('../../../scripts/X.py').read()).
    for (const text of [source, ...Object.values(files)]) {
        for (const m of text.matchAll(/open\(\s*['"]([^'"]+\.py)['"]/g)) {
            const url = new URL(m[1]!, `${location.origin}${dir}/`).pathname;
            if (files[`${root}${url}`] !== undefined) continue;
            const res = await fetch(url);
            if (res.ok && !res.headers.get("content-type")?.includes("html")) files[`${root}${url}`] = await res.text();
        }
    }
    return files;
}

/** Writes path -> source files into Pyodide's file system. */
function writePythonFiles(files: Record<string, string>): void {
    const fs = (pyodide as unknown as { FS: { mkdirTree(p: string): void; writeFile(p: string, d: string): void } }).FS;
    for (const [path, text] of Object.entries(files)) {
        fs.mkdirTree(path.slice(0, path.lastIndexOf("/")) || "/");
        fs.writeFile(path, text);
    }
}

/** A value the script read from Mogwai state while recording (resolved at replay). */
export interface MogwaiRef {
    mogwaiRef: "clock" | "frameCapture" | "scene";
    path: string;
}

/** One recorded Mogwai script call (see recordMogwaiScript). */
export type MogwaiCommand =
    | { op: "addGraph"; graph: RenderGraph }
    | { op: "removeGraph"; graph: RenderGraph }
    | { op: "loadScene"; path: string; flags: number }
    | { op: "unloadScene" }
    | { op: "resizeFrameBuffer"; width: number; height: number }
    | { op: "renderFrame" }
    | { op: "set"; target: "clock" | "frameCapture" | "scene"; key: string; value: unknown }
    | { op: "call"; target: "clock" | "frameCapture" | "scene" | RenderGraph | RenderPass; method: string; args: unknown[] }
    | { op: "setPass"; target: RenderPass; key: string; value: unknown };

/**
 * Runs a Mogwai script (e.g. an unmodified `tests/image_tests` test) and records
 * what it asks Mogwai to do, for asynchronous replay: native scripts call
 * m.loadScene / m.renderFrame / m.frameCapture.capture synchronously, which
 * the web can only perform asynchronously. Graph calls after m.addGraph(g)
 * (e.g. g.updatePass between captures) are recorded in order too. `files`
 * (path -> source) are written to Pyodide's file system under /mogwai, `cwd`
 * becomes the working directory and first sys.path entry.
 */
export function recordMogwaiScript(device: Device, source: string, files: Record<string, string>, cwd: string): MogwaiCommand[] {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    const commands: MogwaiCommand[] = [];
    const added = new WeakSet<RenderGraph>();
    const targets = new WeakMap<object, RenderGraph>();
    const conv = (v: unknown) => toJs(v);

    // Passes fetched from a graph: property reads run now, set_properties is recorded.
    const makePass = (pass: RenderPass) =>
        new Proxy(pass, {
            get(target, key, receiver) {
                if (key === "getDictionary") return () => pyodide!.toPy(target.getProperties().toJSON());
                if (key === "properties") return pyodide!.toPy(target.getProperties().toJSON());
                if (key === "set_properties")
                    return (dict: unknown) => void commands.push({ op: "call", target, method: "setProperties", args: [new Properties(conv(dict) as Record<string, never>)] });
                const value = Reflect.get(target, key, receiver);
                if (typeof value !== "function") return value;
                return (...args: unknown[]) => void commands.push({ op: "call", target, method: String(key), args: args.map(conv) });
            },
            // Python property sets (e.g. SceneDebugger.mode) apply in script order at replay.
            set: (target, key, value) => (commands.push({ op: "setPass", target, key: String(key), value: conv(value) }), true),
        });
    const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c: string) => c.toUpperCase());
    // Graphs record their calls once added (building them at import time runs directly).
    const makeGraph = (name: string) => {
        const graph = new RenderGraph(device, name);
        const proxy = new Proxy(graph, {
            get(target, key, receiver) {
                const name = typeof key === "string" && !(key in target) ? camel(key) : key;
                if (name === "getPass" || name === "get") return (passName: string) => makePass(target.getPass(String(passName))!);
                const value = Reflect.get(target, name, receiver);
                if (typeof value !== "function") return value;
                return (...args: unknown[]) => {
                    if (!added.has(target)) return (value as (...a: unknown[]) => unknown).apply(target, args.map(conv));
                    commands.push({ op: "call", target, method: String(name), args: args.map(conv) });
                    return undefined;
                };
            },
        });
        targets.set(proxy, graph);
        return proxy;
    };
    // Records property sets and calls on m.clock / m.frameCapture / m.scene (nested paths for the scene).
    // Values read at record time (e.g. m.scene.cameras[1]) become references resolved at replay.
    const kRef = Symbol("mogwaiRef");
    const deref = (v: unknown): unknown => {
        const ref = (v as { [kRef]?: MogwaiRef } | null | undefined)?.[kRef];
        return ref ?? conv(v);
    };
    const recorder = (target: "clock" | "frameCapture" | "scene", path = ""): Record<string, unknown> =>
        new Proxy((() => {}) as unknown as Record<string, unknown>, {
            get: (_t, key) => {
                if (key === kRef) return { mogwaiRef: target, path } satisfies MogwaiRef;
                if (key === "then") return undefined;
                // Subscripts arrive as .get(index) on the JS side.
                if (key === "get") return (index: unknown) => recorder(target, `${path}.${String(index)}`);
                return recorder(target, path ? `${path}.${String(key)}` : String(key));
            },
            set: (_t, key, value) => (commands.push({ op: "set", target, key: path ? `${path}.${String(key)}` : String(key), value: deref(value) }), true),
            apply: (_t, _this, args: unknown[]) => void commands.push({ op: "call", target, method: path, args: args.map(deref) }),
        });

    const recordedSettings = { addOptions: (dict: unknown) => globalSettings.addOptions(toJs(dict) as Record<string, never>) };
    pyodide.registerJsModule("_falcor_js", {
        RenderGraph: makeGraph,
        createPass: (type: string, props?: unknown) => createPass(device, type, new Properties((toJs(props) as Record<string, never>) ?? {})),
        TextureChannelFlags: { Red: 1, Green: 2, Blue: 4, Alpha: 8, RGB: 7, RGBA: 15 },
        float2: (x = 0, y = 0) => new float2(x, y),
        float3: (x = 0, y = 0, z = 0) => new float3(x, y, z),
        float4: (x = 0, y = 0, z = 0, w = 0) => new float4(x, y, z, w),
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
    });
    // A real module object: scripts probe it (e.g. `"IMAGE_TEST_RUN_ONLY" in falcor.__dict__`).
    pyodide.runPython(
        // registerJsModule doesn't replace an imported module: drop the previous run's first.
        `import sys, types\nsys.modules.pop("_falcor_js", None)\nimport _falcor_js\n_falcor = types.ModuleType("falcor")\nfor _k in dir(_falcor_js):\n    if not _k.startswith("__"): setattr(_falcor, _k, getattr(_falcor_js, _k))\nsys.modules["falcor"] = _falcor`,
    );
    // Graphs in m (Renderer::mGraphs order), for getGraph / removeGraph(name) / activeGraph.
    const graphList: RenderGraph[] = [];
    const byName = (g: RenderGraph | string) => (typeof g === "string" ? graphList.find((x) => x.name === g) : (targets.get(g) ?? g));
    pyodide.globals.set("m", {
        addGraph: (g: RenderGraph) => {
            const graph = targets.get(g) ?? g;
            added.add(graph);
            graphList.push(graph);
            commands.push({ op: "addGraph", graph });
        },
        /** Mirrors Renderer::getGraph (None when no graph has that name: JS undefined, not null). */
        getGraph: (name: string) => graphList.find((x) => x.name === String(name)),
        get activeGraph() {
            return graphList.at(-1);
        },
        unloadScene: () => void commands.push({ op: "unloadScene" }),
        loadScene: (path: string, flags?: number) => void commands.push({ op: "loadScene", path: String(path), flags: Number(flags ?? 0) }),
        // Mirrors Renderer::removeGraph: a graph or its name.
        removeGraph: (g: RenderGraph | string) => {
            const graph = byName(g);
            if (!graph) return;
            graphList.splice(graphList.indexOf(graph), 1);
            commands.push({ op: "removeGraph", graph });
        },
        resizeFrameBuffer: (width: number, height: number) => void commands.push({ op: "resizeFrameBuffer", width: Number(width), height: Number(height) }),
        renderFrame: () => void commands.push({ op: "renderFrame" }),
        clock: recorder("clock"),
        frameCapture: recorder("frameCapture"),
        scene: recorder("scene"),
        ui: false,
        // Profiler reads/events take effect at record time (the profiler isn't replayed).
        profiler: device.profilerHook?.pythonBindings((v) => pyodide!.toPy(v)) ?? null,
        settings: recordedSettings,
        getSettings: () => recordedSettings,
    });

    writePythonFiles(files);
    (pyodide as unknown as { FS: { mkdirTree(p: string): void } }).FS.mkdirTree(cwd);
    // Fresh imports per run: the scripts' modules (helpers, graphs.*) are cached across runs otherwise.
    pyodide.runPython(
        `import os, sys\nos.chdir(${JSON.stringify(cwd)})\nsys.path.insert(0, ${JSON.stringify(cwd)})\n` +
            `for _k in [k for k, v in list(sys.modules.items()) if getattr(v, "__file__", None) and str(getattr(v, "__file__")).startswith("/mogwai")]: del sys.modules[_k]`,
    );
    try {
        pyodide.globals.set("__mogwai_script", source);
        pyodide.runPython(kMogwaiShim);
        pyodide.runPython(`try:\n    exec(compile(__mogwai_script, "script", "exec"), globals())\nexcept SystemExit:\n    pass\n`);
    } finally {
        // The recording falcor module must not leak into later graph/console scripts.
        pyodide.runPython(`sys.path.remove(${JSON.stringify(cwd)})\nsys.modules.pop("falcor", None)`);
    }
    return commands;
}
