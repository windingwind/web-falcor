/**
 * Python scripting mirroring Falcor/Utils/Scripting (pybind11 -> Pyodide,
 * user decision DESIGN.md §11.1: the .py path is primary).
 *
 * Executes unmodified upstream render-graph .py scripts: a JS `falcor` bridge
 * module provides RenderGraph/createPass, and a Mogwai-like `m` global
 * captures addGraph calls.
 */

import type { Device } from "../../Core/API/Device.js";
import { RenderGraph } from "../../RenderGraph/RenderGraph.js";
import { createPass } from "../../RenderGraph/RenderPass.js";
import { Properties } from "../Properties.js";
import { RuntimeError } from "../../Core/Error.js";
import { CameraBridge, LightBridge, MaterialBridge, SceneBuilderBridge, TriangleMesh, makeTransform } from "../../Scene/SceneBuilder.js";
import type { Scene } from "../../Scene/Scene.js";
import { LightType } from "../../Scene/SceneData.js";
import { MaterialType } from "../../Scene/Material/MaterialData.js";
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

export function isScriptingInitialized(): boolean {
    return pyodide !== null;
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
export async function runGraphScript(device: Device, source: string): Promise<RenderGraph[]> {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    const graphs: RenderGraph[] = [];

    const falcorModule = {
        RenderGraph: (name: string) => new RenderGraph(device, name),
        createPass: (type: string, props?: unknown) =>
            createPass(device, type, new Properties((toJs(props) as Record<string, never>) ?? {})),
    };
    pyodide.registerJsModule("falcor", falcorModule);

    const mogwai = {
        addGraph: (graph: RenderGraph) => {
            graphs.push(graph);
        },
    };
    pyodide.globals.set("m", mogwai);

    pyodide.runPython(source);

    if (graphs.length === 0) throw new RuntimeError("Graph script did not register a graph via m.addGraph()");
    return graphs;
}

/** Python prelude adapting pythonic pyscene API (kwargs, class-style ctors)
 *  to the JS SceneBuilder bridge. */
const kScenePrelude = `
import sys
sys.modules.pop('webfalcor_scene', None)  # registerJsModule per call; defeat import caching
from webfalcor_scene import (sceneBuilder, _TriangleMesh,
    PointLight, DirectionalLight, DistantLight, StandardMaterial, ClothMaterial, HairMaterial,
    PBRTDiffuseMaterial, PBRTConductorMaterial, Camera, _makeTransform, _makeEnvMap)

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
    @staticmethod
    def createQuad(size=None):
        return _TriangleMesh.createQuad(size)
    @staticmethod
    def createCube(size=None):
        return _TriangleMesh.createCube(size)
    @staticmethod
    def createSphere(radius=1.0, segmentsU=32, segmentsV=32):
        return _TriangleMesh.createSphere(radius, segmentsU, segmentsV)

def Transform(translation=None, rotationEuler=None, rotationEulerDeg=None, scaling=None):
    return _makeTransform(translation, rotationEuler, rotationEulerDeg, scaling)

class EnvMap:
    @staticmethod
    def createFromFile(path):
        return _makeEnvMap(path)
    def __new__(cls, path):
        return _makeEnvMap(path)

# Guard: python setattr on JS proxies silently creates properties, so a typo'd
# or unimplemented bridge property would be DROPPED. Wrap the factories so
# unknown attribute writes raise instead (mirrors pybind11 strictness).
def _guarded(factory, known):
    def make(*args):
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
             'indexOfRefraction', 'specularTransmission', 'diffuseTransmission', 'thinSurface'}
_lightProps = {'position', 'intensity', 'direction', 'angle'}
_camProps = {'position', 'target', 'up', 'focalLength'}
StandardMaterial = _guarded(StandardMaterial, _matProps)
ClothMaterial = _guarded(ClothMaterial, _matProps)
HairMaterial = _guarded(HairMaterial, _matProps)
PBRTDiffuseMaterial = _guarded(PBRTDiffuseMaterial, _matProps)
PBRTConductorMaterial = _guarded(PBRTConductorMaterial, _matProps)
PointLight = _guarded(PointLight, _lightProps)
DirectionalLight = _guarded(DirectionalLight, _lightProps)
DistantLight = _guarded(DistantLight, _lightProps)
Camera = _guarded(Camera, _camProps)
`;

/**
 * Executes an unmodified .pyscene through the SceneBuilder bridge and
 * resolves the resulting scene (assets fetched relative to baseUrl).
 */
export async function runSceneScript(device: Device, source: string, baseUrl: string): Promise<Scene> {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    const builder = new SceneBuilderBridge();

    type VecLike = { x: number; y: number; z: number };
    const sceneModule = {
        sceneBuilder: builder,
        // Pyodide calls JS classes without `new`; vectors live python-side (prelude).
        _TriangleMesh: {
            createQuad: (size?: { x: number; y: number } | null) => TriangleMesh.createQuad(size ? new float2(size.x, size.y) : undefined),
            createCube: (size?: VecLike | null) => TriangleMesh.createCube(size ? new float3(size.x, size.y, size.z) : undefined),
            createSphere: (radius?: number, segmentsU?: number, segmentsV?: number) => TriangleMesh.createSphere(radius, segmentsU, segmentsV),
        },
        Camera: (_name = "") => new CameraBridge(),
        PointLight: (name = "") => new LightBridge(LightType.Point, name),
        DirectionalLight: (name = "") => new LightBridge(LightType.Directional, name),
        DistantLight: (name = "") => new LightBridge(LightType.Distant, name),
        StandardMaterial: (name = "") => new MaterialBridge(MaterialType.Standard, name),
        ClothMaterial: (name = "") => new MaterialBridge(MaterialType.Cloth, name),
        HairMaterial: (name = "") => new MaterialBridge(MaterialType.Hair, name),
        PBRTDiffuseMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTDiffuse, name),
        PBRTConductorMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTConductor, name),
        _makeTransform: makeTransform,
        _makeEnvMap: (path: string) => ({ path, intensity: 1 }),
    };
    pyodide.registerJsModule("webfalcor_scene", sceneModule);

    pyodide.runPython(kScenePrelude + "\n" + source);

    return builder.resolve(device, baseUrl);
}
