/**
 * Python scripting mirroring Falcor/Utils/Scripting (pybind11 -> Pyodide,
 * user decision docs §11.1: the .py path is primary).
 *
 * Executes unmodified upstream render-graph .py scripts: a JS `falcor` bridge
 * module provides RenderGraph/createPass, and a Mogwai-like `m` global
 * captures addGraph calls.
 */

import type { Device } from "../../Core/API/Device.js";
import { RenderGraph } from "../../RenderGraph/RenderGraph.js";
import { Settings } from "../Settings.js";
import { buildSceneFromCache, encodeTextureSources, loadSceneCache, sceneCacheKey, snapshotCameraPose, snapshotGridVolumes, storeSceneCache } from "../../Scene/SceneCache.js";
import { createPass } from "../../RenderGraph/RenderPass.js";
import { Properties } from "../Properties.js";
import { RuntimeError } from "../../Core/Error.js";
import { AssetResolver, withScriptSearchPath } from "../../Core/AssetResolver.js";
import { CameraBridge, GridVolumeBridge, LightBridge, MaterialBridge, SceneBuilderBridge, SceneBuilderFlags, SDFGridBridge, TriangleMesh, kSceneBuilderFlagsPython, makeTransform } from "../../Scene/SceneBuilder.js";
import type { Scene } from "../../Scene/Scene.js";
import { LightType, type StaticVertex } from "../../Scene/SceneData.js";
import { MaterialType, ShadingModel } from "../../Scene/Material/MaterialData.js";
import { buildSphereGrid, buildBoxGrid } from "../../Scene/Volume/VDBLoader.js";
import { float2, float3, float4 } from "../Math/Vector.js";

interface PyodideApi {
    registerJsModule(name: string, module: object): void;
    runPython(code: string, options?: { globals?: unknown }): unknown;
    globals: { set(name: string, value: unknown): void; get(name: string): unknown };
    toPy(obj: unknown): unknown;
}

let pyodide: PyodideApi | null = null;

/** Loads Pyodide (idempotent). indexURL points at the pyodide distribution. */
export async function initScripting(indexURL: string): Promise<void> {
    if (pyodide) return;
    const mod = (await import(/* @vite-ignore */ `${indexURL}/pyodide.mjs`)) as {
        loadPyodide(options: { indexURL: string }): Promise<PyodideApi>;
    };
    pyodide = await mod.loadPyodide({ indexURL });
}

/** Shared Settings instance (native: SampleApp::getSettings()). */
const globalSettings = new Settings();

export function getGlobalSettings(): Settings {
    return globalSettings;
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

    const mogwai = {
        addGraph: (graph: RenderGraph) => {
            graphs.push(graph);
        },
        // Mirrors the native Profiler script binding (m.profiler).
        profiler: device.profilerHook?.pythonBindings((v) => pyodide!.toPy(v)) ?? null,
        // Mirrors the native Settings script binding (Mogwai getSettings()).
        settings: {
            addOptions: (dict: unknown) => {
                globalSettings.addOptions(toJs(dict) as Record<string, never>);
                applyMediaSearchPaths();
                for (const g of graphs) for (const { pass } of g.getPasses()) pass.onOptionsChange(globalSettings.getOptions());
            },
            addFilteredAttributes: (dictOrList: unknown) => globalSettings.addFilteredAttributes(toJs(dictOrList) as Record<string, never>),
            clearOptions: () => globalSettings.clearOptions(),
            clearFilteredAttributes: () => globalSettings.clearFilteredAttributes(),
        },
        ...extras,
    };
    pyodide.globals.set("m", mogwai);

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
    pyodide.globals.set("m", {
        scene: context.scene,
        activeGraph: context.graph,
        clock: context.clock,
        timingCapture: context.timingCapture,
        frameCapture: context.frameCapture,
        profiler: (context.profiler ?? device.profilerHook)?.pythonBindings((v) => pyodide!.toPy(v)) ?? null,
        settings: {
            addOptions: (dict: unknown) => {
                globalSettings.addOptions(toJs(dict) as Record<string, never>);
                applyMediaSearchPaths();
            },
        },
    });
    const py = pyodide as unknown as { setStdout(opts: { batched: (s: string) => void }): void; runPython(src: string): unknown };
    py.setStdout({ batched: (s) => lines.push(s) });
    try {
        const result = py.runPython("from falcor import *\n" + source);
        if (result !== undefined && result !== null) lines.push(String(result));
    } finally {
        py.setStdout({ batched: (s) => console.log(s) });
    }
    return lines.join("\n");
}

/** Python prelude adapting pythonic pyscene API (kwargs, class-style ctors)
 *  to the JS SceneBuilder bridge. */
const kScenePrelude = `
import sys
sys.modules.pop('webfalcor_scene', None)  # registerJsModule per call; defeat import caching
from webfalcor_scene import (sceneBuilder, SceneBuilderFlags, _TriangleMesh,
    PointLight, DirectionalLight, DistantLight, RectLight, DiscLight, SphereLight,
    StandardMaterial, ClothMaterial, HairMaterial,
    PBRTDiffuseMaterial, PBRTConductorMaterial, _MERLMaterial, _MERLMixMaterial, _RGLMaterial,
    Camera, _makeTransform, _makeAABB, _makeEnvMap, _GridVolume, _Grid, _SDFGridCreate)

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

def Transform(translation=None, rotationEuler=None, rotationEulerDeg=None, scaling=None):
    return _makeTransform(translation, rotationEuler, rotationEulerDeg, scaling)

def AABB(min, max):
    return _makeAABB(min, max)

class EnvMap:
    @staticmethod
    def createFromFile(path):
        return _makeEnvMap(path)
    def __new__(cls, path):
        return _makeEnvMap(path)

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
            def __getattr__(self, k): return getattr(object.__getattribute__(self, '_o'), k)
            def __setattr__(self, k, v):
                if k not in known:
                    raise AttributeError(f'unsupported property: {k} (web bridge)')
                setattr(object.__getattribute__(self, '_o'), k, v)
        return Guard(obj)
    return make

_matProps = {'baseColor', 'specularParams', 'transmissionColor', 'emissiveColor',
             'emissiveFactor', 'doubleSided', 'roughness', 'metallic',
             'indexOfRefraction', 'specularTransmission', 'diffuseTransmission', 'thinSurface',
             'nestedPriority', 'volumeAbsorption', 'volumeScattering',
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
            'emissionMode', 'emissionTemperature', 'densityGrid',
            'frameRate', 'startFrame', 'playbackEnabled'}
_GridVolumeGuarded = _guarded(_GridVolume, _gvProps)
class GridVolume:
    GridSlot = _GridSlot
    def __new__(cls, name=''):
        return _GridVolumeGuarded(name)
Volume = GridVolume  # legacy pyscene alias (volume_test.pyscene)

# Procedural density grids (two_volumes.pyscene).
class Grid:
    @staticmethod
    def createSphere(radius, voxelSize):
        return _Grid.createSphere(radius, voxelSize)
    @staticmethod
    def createBox(width, height, depth, voxelSize):
        return _Grid.createBox(width, height, depth, voxelSize)

# SDF grids: all four representations, built from the procedural generator,
# a .sdfg corner-value file or a .sdf primitive list.
class SDFGrid:
    @staticmethod
    def createNDGrid(narrowBandThickness=5.0):
        return _SDFGridCreate('ndsdf', narrowBandThickness, 7)
    @staticmethod
    def createSBS(brickWidth=7, compressed=False, defaultGridWidth=256):
        return _SDFGridCreate('sbs', 5.0, brickWidth)
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
}

export async function runSceneScript(device: Device, source: string, baseUrl: string, options?: SceneScriptOptions): Promise<Scene> {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    sceneLoadedFromCache = false;
    return withScriptSearchPath(baseUrl, () => runSceneScriptInternal(device, source, baseUrl, options));
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
        Camera: (_name = "") => new CameraBridge(),
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
        _makeAABB: (min: VecLike, max: VecLike) => ({ min: { x: min.x, y: min.y, z: min.z }, max: { x: max.x, y: max.y, z: max.z } }),
        _makeEnvMap: (path: string) => ({ path, intensity: 1 }),
        _GridVolume: (name = "") => new GridVolumeBridge(name),
        _Grid: {
            createSphere: (radius: number, voxelSize: number) => ({ _proceduralGrid: buildSphereGrid(radius, voxelSize) }),
            createBox: (width: number, height: number, depth: number, voxelSize: number) => ({ _proceduralGrid: buildBoxGrid(width, height, depth, voxelSize) }),
        },
        _SDFGridCreate: (type: string, narrowBandThickness = 5.0, brickWidth = 7) => new SDFGridBridge(type as "ndsdf" | "sbs", narrowBandThickness, brickWidth),
        SceneBuilderFlags: kSceneBuilderFlagsPython,
        ...AssetResolver.pythonBindings,
    };
    pyodide.registerJsModule("webfalcor_scene", sceneModule);

    pyodide.runPython(kScenePrelude + "\n" + source);

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
            camera: snapshotCameraPose(scene),
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
