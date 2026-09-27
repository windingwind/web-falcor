/**
 * JS scripts replace Python ones: every name a .pyscene / Mogwai script gets has a JS counterpart, and the
 * converted corpus (npm run py2js:corpus -> tests/gpu/generated/py2js) builds the same scenes, graphs and
 * recorded Mogwai commands as the Python originals.
 */

import {
    RenderGraph,
    SceneBuilderBridge,
    createPass,
    createSceneScriptContext,
    getPyodide,
    initScripting,
    isJsScript,
    kMogwaiShim,
    recordMogwaiModule,
    runGraphModule,
    runGraphScript,
    runSceneModule,
    runSceneScript,
    type MogwaiCommand,
    type Scene,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { recordMogwaiSource } from "../../../packages/mogwai/src/ScriptRunner.js";
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

const kGenerated = "/tests/gpu/generated/py2js";

type Py = { runPython(src: string): unknown; globals: { set(k: string, v: unknown): void } };
const py = () => getPyodide() as Py;

async function generated(path: string): Promise<string> {
    const url = `${kGenerated}/${path}`;
    const res = await fetch(url, { method: "HEAD" });
    if (!res.ok || !(res.headers.get("content-type") ?? "").includes("javascript")) throw new SkipError(`${url} missing: run npm run py2js:corpus`);
    return url;
}

// ---- API parity ----

gpuTest("Py2Js.sceneApiCoversPython", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    // The names a .pyscene sees (the scene prelude's globals), minus modules and private helpers.
    let pyNames: string[] = [];
    (globalThis as { __py2jsNames?: (s: string) => void }).__py2jsNames = (s) => (pyNames = JSON.parse(s) as string[]);
    await runSceneScript(
        device,
        "import js as _js, json as _json, types as _types\n" +
            "_js.__py2jsNames(_json.dumps(sorted(k for k, v in globals().items() if not k.startswith('_') and not isinstance(v, _types.ModuleType))))\n" +
            "sceneBuilder.addMeshInstance(sceneBuilder.addNode('q', Transform()), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(), StandardMaterial('m')))\n",
        "/tests/oracle/assets",
    );
    const js = new Set(Object.keys(createSceneScriptContext(new SceneBuilderBridge())));
    const missing = pyNames.filter((n) => !js.has(n));
    console.error(`# py2js scene API: ${pyNames.length} Python names, ${js.size} JS names`);
    expectEq(missing.join(", "), "", "Python scene names without a JS counterpart");
});

gpuTest("Py2Js.mogwaiApiCoversPython", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    // `from falcor import *` in a Mogwai script, plus the shim's globals (m, t, fc, tc, legacy enums).
    let pyNames: string[] = [];
    let mNames: string[] = [];
    Object.assign(globalThis, { __py2jsNames: (s: string) => (pyNames = JSON.parse(s) as string[]), __py2jsM: (s: string) => (mNames = JSON.parse(s) as string[]) });
    py().globals.set("__py2js_shim", kMogwaiShim);
    await recordMogwaiSource(
        device,
        "import js as _js, json as _json, types as _types, falcor as _falcor\n" +
            "_d = {'m': m}\nexec(__py2js_shim, _d)\n" +
            "_n = {k for k, v in _d.items() if not k.startswith('_') and not isinstance(v, (_types.ModuleType, type)) or k == 'CullMode' or k[0].isupper() and not k.startswith('_')}\n" +
            "_n |= {k for k in dir(_falcor) if not k.startswith('_')}\n" +
            // A JS proxy's own Python attributes (to_py, as_object_map, …) aren't script names.
            "_proxy = set(dir(_js.Object.new()))\n_n -= _proxy\n" +
            "_js.__py2jsNames(_json.dumps(sorted(_n)))\n" +
            "_js.__py2jsM(_json.dumps(sorted(k for k in dir(object.__getattribute__(m, '_o')) if not k.startswith('_') and k not in _proxy)))\n",
        "/tests/oracle/assets",
    );
    // The JS context of a Mogwai script (built by recording a script that reports its context).
    // Names, including m's accessor properties (sceneUpdateCallback, keyCallback aren't enumerable).
    const url = URL.createObjectURL(new Blob(["export default (ctx) => { globalThis.__py2jsCtx = Object.keys(ctx); globalThis.__py2jsCtxM = Object.getOwnPropertyNames(ctx.m); };"], { type: "text/javascript" }));
    await recordMogwaiModule(device, url);
    const g = globalThis as unknown as { __py2jsCtx: string[]; __py2jsCtxM: string[] };
    const [js, jsM] = [new Set(g.__py2jsCtx), new Set(g.__py2jsCtxM)];
    console.error(`# py2js Mogwai API: ${pyNames.length} Python names, ${js.size} JS names; m: ${mNames.length} / ${jsM.size}`);
    expectEq(pyNames.filter((n) => !js.has(n)).join(", "), "", "Python Mogwai-script names without a JS counterpart");
    expectEq(mNames.filter((n) => !jsM.has(n)).join(", "), "", "Python m attributes without a JS counterpart");
});

// ---- Render-graph scripts ----

/** A graph's structure: passes with their types and properties, edges, outputs. */
function graphShape(g: RenderGraph): unknown {
    return {
        name: g.name,
        passes: g.getPasses().map(({ name, pass }) => ({ name, type: pass.type, props: pass.getProperties().toJSON() })),
        edges: g.getEdges(),
        outputs: g.getOutputNames(),
    };
}

gpuTest("Py2Js.graphScriptsMatch", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    // Falcor's graph scripts (some also load a scene): recorded as Mogwai scripts, the graphs they add compared.
    // PathTracerNRD needs DLSSPass (NVIDIA-only), in Python too.
    for (const name of ["BSDFViewer", "MinimalPathTracer", "PathTracer", "RTXDI", "SceneDebugger", "WARDiffPathTracer"]) {
        const source = await (await fetch(`/Falcor/scripts/${name}.py`)).text();
        const py = await recordMogwaiSource(device, source, "/Falcor/scripts", `${name}.py`);
        const js = await recordMogwaiModule(device, await generated(`Falcor/scripts/${name}.js`));
        expectEq(JSON.stringify(js.map(commandShape)), JSON.stringify(py.map(commandShape)), `${name}: recorded commands`);
    }
    // A graph-only script through the non-recording runners too.
    const pyGraphs = await runGraphScript(device, await (await fetch("/Falcor/scripts/MinimalPathTracer.py")).text());
    const jsGraphs = await runGraphModule(device, await generated("Falcor/scripts/MinimalPathTracer.js"));
    expectEq(JSON.stringify(jsGraphs.map(graphShape)), JSON.stringify(pyGraphs.map(graphShape)), "MinimalPathTracer: graphs");
});

// ---- Mogwai scripts (Falcor's image tests) ----

/** A recorded command as comparable JSON: graphs and passes by name, Properties as dicts, callbacks as "fn". */
function commandShape(c: MogwaiCommand): unknown {
    const value = (v: unknown): unknown => {
        if (v instanceof RenderGraph) return { graph: graphShape(v) };
        if (typeof v === "function") return "fn";
        if (v && typeof v === "object" && "toJSON" in v && typeof (v as { toJSON: unknown }).toJSON === "function") return (v as { toJSON(): unknown }).toJSON();
        if (v && typeof v === "object" && "type" in v && "getProperties" in v) return { pass: (v as { name: string }).name };
        // Python paths are Pyodide file-system paths (/mogwai/...), mapped back to URLs at replay.
        if (typeof v === "string") return v.startsWith("/mogwai/") ? v.slice("/mogwai".length) : v;
        if (Array.isArray(v)) return v.map(value);
        if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, value(x)]));
        return v;
    };
    return value(c);
}

gpuTest("Py2Js.imageTestScriptsMatch", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const listing = (await (await fetch(`/__webfalcor/ls?path=${encodeURIComponent("/Falcor/tests/image_tests")}`)).json().catch(() => null)) as { dirs: string[] } | null;
    if (!listing) throw new SkipError("no directory listing (the dev server provides it)");
    let compared = 0;
    const failures: string[] = [];
    for (const dir of listing.dirs.filter((d) => d !== "__pycache__" && d !== "graphs")) {
        const files = ((await (await fetch(`/__webfalcor/ls?path=${encodeURIComponent(`/Falcor/tests/image_tests/${dir}`)}`)).json()) as { files: string[] }).files.filter((f) => /^test_.*\.py$/.test(f));
        for (const file of files) {
            const pyUrl = `/Falcor/tests/image_tests/${dir}/${file}`;
            const source = await (await fetch(pyUrl)).text();
            let pyCommands: MogwaiCommand[];
            try {
                pyCommands = await recordMogwaiSource(device, source, `/Falcor/tests/image_tests/${dir}`, file);
            } catch (e) {
                // Scripts the Python path can't record here aren't a JS question (e.g. DLSSPass).
                console.error(`# py2js image tests: skipped ${dir}/${file} (Python: ${String(e).split("\n").filter(Boolean).pop()?.slice(0, 120)})`);
                continue;
            }
            try {
                const jsCommands = await recordMogwaiModule(device, await generated(`Falcor/tests/image_tests/${dir}/${file.replace(/\.py$/, ".js")}`));
                const [a, b] = [JSON.stringify(jsCommands.map(commandShape)), JSON.stringify(pyCommands.map(commandShape))];
                if (a !== b) failures.push(`${dir}/${file}: ${firstDifference(a, b)}`);
            } catch (e) {
                if (e instanceof SkipError) throw e;
                failures.push(`${dir}/${file}: ${String(e)}`);
            }
            compared++;
        }
    }
    console.error(`# py2js image tests: ${compared} compared, ${failures.length} differ`);
    for (const f of failures.slice(0, 20)) console.error(`#   ${f}`);
    expectEq(failures.length, 0, "image-test scripts whose JS recording differs from the Python one");
});

function firstDifference(a: string, b: string): string {
    let i = 0;
    while (i < a.length && a[i] === b[i]) i++;
    return `js …${a.slice(Math.max(0, i - 80), i + 80)}… vs py …${b.slice(Math.max(0, i - 80), i + 80)}…`;
}

// ---- Scenes ----

/** A built scene's fingerprint: stats, shader defines, bounds, cameras and lights. */
function sceneShape(s: Scene): unknown {
    const round = (v: unknown): unknown =>
        typeof v === "number" ? Number(v.toPrecision(6)) : Array.isArray(v) ? v.map(round) : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, round(x)])) : v;
    return round({
        stats: s.getSceneStats(),
        defines: s.getSceneDefines(),
        bounds: s.bounds,
        cameras: s.getCameras().map((c) => ({ name: c.name, position: c.getPosition(), target: c.getTarget(), focalLength: c.getFocalLength() })),
        lights: s.lights.map((l) => ({ name: (l as { name?: string }).name, type: (l as { type?: unknown }).type })),
    });
}

async function renderScene(device: Parameters<Parameters<typeof gpuTest>[1]>[0]["device"], scene: Scene): Promise<Float32Array> {
    const size = 96;
    scene.camera.setAspectRatio(1);
    const graph = new RenderGraph(device, "Py2Js");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: true }), "VBufferRT");
    graph.addPass(createPass(device, "MinimalPathTracer", { maxBounces: 2 }), "PT");
    graph.addEdge("VBufferRT.vbuffer", "PT.vbuffer");
    graph.markOutput("PT.color");
    graph.setScene(scene);
    graph.execute(device.renderContext);
    return new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("PT.color")!)).buffer);
}

async function pyscenes(dir: string): Promise<string[]> {
    const ls = (await (await fetch(`/__webfalcor/ls?path=${encodeURIComponent(dir)}`)).json().catch(() => null)) as { files: string[]; dirs: string[] } | null;
    if (!ls) return [];
    const nested = await Promise.all(ls.dirs.filter((d) => d !== "__pycache__").map((d) => pyscenes(`${dir}/${d}`)));
    return [...ls.files.filter((f) => f.endsWith(".pyscene")).map((f) => `${dir}/${f}`), ...nested.flat()];
}

gpuTest("Py2Js.scenesMatch", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const all = [...(await pyscenes("/Falcor/media/test_scenes")), ...(await pyscenes("/tests/oracle/assets"))];
    if (!all.length) throw new SkipError("no scenes listed (the dev server provides listings)");
    let compared = 0;
    const failures: string[] = [];
    for (const path of all) {
        const dir = path.slice(0, path.lastIndexOf("/"));
        let pyScene: Scene;
        try {
            pyScene = await runSceneScript(device, await (await fetch(path)).text(), dir, { path });
        } catch {
            continue; // not loadable here in Python either
        }
        try {
            const jsUrl = await generated(`${path.slice(1).replace(/\.pyscene$/, ".scene.js")}`);
            const jsScene = await runSceneModule(device, jsUrl, { baseUrl: dir, path });
            const [a, b] = [JSON.stringify(sceneShape(jsScene)), JSON.stringify(sceneShape(pyScene))];
            if (a !== b) failures.push(`${path}: ${firstDifference(a, b)}`);
            else {
                const [ja, pa] = [await renderScene(device, jsScene), await renderScene(device, pyScene)];
                let diff = 0;
                for (let i = 0; i < ja.length; i++) if (ja[i] !== pa[i] && !(Number.isNaN(ja[i]) && Number.isNaN(pa[i]))) diff++;
                if (diff) failures.push(`${path}: ${diff} pixel values differ`);
            }
            jsScene.destroy();
        } catch (e) {
            if (e instanceof SkipError) throw e;
            failures.push(`${path}: ${String(e)}`);
        }
        pyScene.destroy();
        compared++;
    }
    console.error(`# py2js scenes: ${compared} compared, ${failures.length} differ`);
    for (const f of failures.slice(0, 20)) console.error(`#   ${f}`);
    expectEq(failures.length, 0, "scenes whose JS build differs from the Python one");
});

void isJsScript;
