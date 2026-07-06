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
import { LightBridge, MaterialBridge, SceneBuilderBridge, TriangleMesh, makeTransform } from "../../Scene/SceneBuilder.js";
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
from webfalcor_scene import (sceneBuilder, TriangleMesh, float2, float3, float4,
    PointLight, DirectionalLight, StandardMaterial, ClothMaterial, HairMaterial,
    PBRTDiffuseMaterial, PBRTConductorMaterial, _makeTransform, _makeEnvMap)

def Transform(translation=None, rotationEuler=None, rotationEulerDeg=None, scaling=None):
    return _makeTransform(translation, rotationEuler, rotationEulerDeg, scaling)

class EnvMap:
    @staticmethod
    def createFromFile(path):
        return _makeEnvMap(path)
    def __new__(cls, path):
        return _makeEnvMap(path)
`;

/**
 * Executes an unmodified .pyscene through the SceneBuilder bridge and
 * resolves the resulting scene (assets fetched relative to baseUrl).
 */
export async function runSceneScript(device: Device, source: string, baseUrl: string): Promise<Scene> {
    if (!pyodide) throw new RuntimeError("Call initScripting() first");
    const builder = new SceneBuilderBridge();

    const sceneModule = {
        sceneBuilder: builder,
        // Pyodide calls JS classes without `new` — expose factories.
        TriangleMesh: {
            createQuad: (size?: float2) => TriangleMesh.createQuad(size),
        },
        float2: (x = 0, y = 0) => new float2(x, y),
        float3: (x = 0, y = 0, z = 0) => new float3(x, y, z),
        float4: (x = 0, y = 0, z = 0, w = 0) => new float4(x, y, z, w),
        PointLight: (name = "") => new LightBridge(LightType.Point, name),
        DirectionalLight: (name = "") => new LightBridge(LightType.Directional, name),
        StandardMaterial: (name = "") => new MaterialBridge(MaterialType.Standard, name),
        ClothMaterial: (name = "") => new MaterialBridge(MaterialType.Cloth, name),
        HairMaterial: (name = "") => new MaterialBridge(MaterialType.Hair, name),
        PBRTDiffuseMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTDiffuse, name),
        PBRTConductorMaterial: (name = "") => new MaterialBridge(MaterialType.PBRTConductor, name),
        _makeTransform: makeTransform,
        _makeEnvMap: (path: string) => ({ path }),
    };
    pyodide.registerJsModule("webfalcor_scene", sceneModule);

    pyodide.runPython(kScenePrelude + "\n" + source);

    return builder.resolve(device, baseUrl);
}
