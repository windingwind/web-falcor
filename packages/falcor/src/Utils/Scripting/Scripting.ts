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
