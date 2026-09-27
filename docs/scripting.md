# Scene and render-graph scripts: JS and Python

Falcor describes scenes (`.pyscene`) and render graphs / Mogwai sessions (`.py`) as Python scripts. web-falcor
runs those unmodified through Pyodide, and it also runs **the same scripts written in JavaScript**, with no
Python involved:

| Python | JavaScript | Receives |
| --- | --- | --- |
| `cornell_box.pyscene` | `cornell_box.scene.js` | `sceneBuilder`, and what the scene prelude gives Python (`StandardMaterial`, `TriangleMesh`, `Transform`, `float3`, …) |
| `PathTracer.py` (graph / Mogwai script) | `PathTracer.js` | `m`, and what `from falcor import *` gives (`RenderGraph`, `createPass`, …), plus `t`, `fc`, `tc` |
| `helpers.py` (a module a script imports) | `helpers.js` | the importing script's names, except `m`/`sceneBuilder` (a Python module doesn't see them either) |

Both languages drive the same bridges (`createSceneBridgeModule`, `createMogwaiRecorder` in
`Utils/Scripting/Scripting.ts`), so a JS script builds the same scene, graph or recorded session as its Python
original. The [Py2Js GPU suite](#how-equivalence-is-checked) checks that for Falcor's whole script corpus.

## Writing a JS script

A JS script is an ES module whose default export receives its names as one object:

```js
// cornell_box.scene.js
export default function ({ sceneBuilder, StandardMaterial, TriangleMesh, Transform, PointLight, float3, float4 }) {
    const red = StandardMaterial("Red");
    red.baseColor = float4(0.63, 0.065, 0.05, 1);
    const quad = TriangleMesh.createQuad();
    sceneBuilder.addMeshInstance(
        sceneBuilder.addNode("Left", Transform({ translation: float3(-0.5, 0, 0), rotationEulerDeg: float3(0, 0, -90) })),
        sceneBuilder.addTriangleMesh(quad, red),
    );
    const light = PointLight("Light");
    light.intensity = float3(1);
    sceneBuilder.addLight(light);
}
```

```js
// MyGraph.js
export default function ({ m, RenderGraph, createPass }) {
    const g = RenderGraph("MyGraph");
    g.addPass(createPass("VBufferRT", { samplePattern: "Stratified", sampleCount: 16 }), "VBufferRT");
    g.addPass(createPass("PathTracer", { samplesPerPixel: 1 }), "PathTracer");
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    g.markOutput("PathTracer.color");
    m.addGraph(g);
}
```

The names behave as in Python:

- **Constructors work with or without `new`:** `StandardMaterial("x")` and `new StandardMaterial("x")` are the
  same, as are `float3(1, 2, 3)` and `new float3(1, 2, 3)`.
- **Vectors construct like Python's:** `float3(1)` is `(1, 1, 1)`. `int2`–`uint4` and `bool2`–`bool4` exist too.
- **Keyword arguments become an options object** where Python uses them for options (`Transform({ ... })`,
  `AABB({ min, max })`, `SDFGrid.createSBS({ brickWidth })`). Elsewhere they're positional, in Python's order
  (`StandardMaterial(name, model)`, `TriangleMesh.createSphere(radius, segmentsU, segmentsV)`).
- **Unknown properties throw,** like the Python bridge: `light.intesity = …` is an error instead of being ignored.
- **`with m.profiler.event("x"):`** is `m.profiler.event("x", () => { ... })`.
- **`m.script(url)` returns a promise:** `await m.script("other.js")` runs another script with the same `m`.
- **The script's own location:** `import.meta.url` (or `ctx.scriptUrl`); a scene also gets `ctx.sceneDir`, its asset
  directory.
- **Mogwai's recorded calls work as in Python:** `m.loadScene`, `m.renderFrame`, `m.frameCapture.capture()` and
  `m.clock.*` record and replay in order, and `g["PassName"]` is written `g.getPass("PassName")`.
- **`py` is the Python runtime** converted scripts use (`py.add` for vector arithmetic, `py.mod` for `%` formatting,
  `py.range`, `py.str`, …). Hand-written scripts can use it, or plain JS.

Old render scripts' enums (`ToneMapOp.Aces`, …) and options structs (`PathTracerParams(...)`) exist for JS too,
evaluating to the strings and dicts the Python shim gives. A script can also export `scene` or `graph` instead of
`default`. `export default async function` works: scene and Mogwai loaders await it.

## Loading JS scripts

- **Mogwai:** `?scene=path/x.scene.js`, `?script=path/x.js`. The open box and drag-and-drop tell them apart by name:
  `*.scene.js` is a scene, any other `.js` (or `.mjs`, or `.ts` on the dev server) is a Mogwai/graph script.
- **The console:** the toolbar's **script** selector switches it between Python and JS. In JS mode, commands run
  with `m`, the falcor names and `py` in scope, and the value of the last expression is printed.
- **Saving:** in JS mode, **Save Config** writes `MogwaiConfig.js` and the graph editor's **Save graph** writes
  `<graph>.js`: the Python Mogwai writes, converted (see below).
- **In code** (`@web-falcor/falcor`): `runSceneModule(device, url, { baseUrl })` builds a scene (the counterpart of
  `runSceneScript`, sharing the scene cache); `recordMogwaiModule(device, url)` records a Mogwai script for replay
  (`recordMogwaiScript`); `runGraphModule(device, url)` returns the graphs a graph script adds (`runGraphScript`);
  `runConsoleCommandJs` is the JS console. `createSceneScriptContext(builder)` gives the scene names for other uses.

Python keeps working everywhere JS does: nothing about `.pyscene` or `.py` handling changed.

## Converting Python scripts

```sh
npx @web-falcor/mogwai py2js scenes/ graphs/MyGraph.py --out js/   # from npm
npm run py2js -- Falcor/scripts/PathTracer.py                      # in this repo (writes PathTracer.js beside it)
```

Files keep their path relative to `--root` (default: the current directory) under `--out` (default: beside the
input). `x.pyscene` becomes `x.scene.js` and `x.py` becomes `x.js`. Local imports (`from helpers import …`, with
`sys.path.append` directories) become imports of the converted modules, `exec(open("x.py").read())` runs the
converted `x.js`, and comments are kept. `--check` syntax-checks the output and fails on warnings. The viewer
converts in the browser for its JS saves (`convertPythonSource`).

The converter parses with Python's own parser (Pyodide) and emits JS that follows Python's semantics where JS
differs:

- vector and list arithmetic, `%` and `str.format` formatting, floor division and modulo;
- truthiness of empty lists/dicts/strings, `in`, slicing, negative indices, `range`/`enumerate`/`zip`;
- keyword arguments (known signatures, and the converted file's own functions);
- `try`/`except` by exception name, and `exit()`;
- the file-system parts scripts use: `os.path` (against the script's directory, as the Python runner's working
  directory), `glob.glob` and `os.listdir` (through the dev server's directory listing).

It reports what it doesn't convert, with the line: classes, `**kwargs` parameters, slice assignment, and
`for`/`while` … `else`. None of these appear in Falcor's scene, graph or image-test scripts. JS numbers don't
remember they were floats, so `str(1.0)` would print `1`. The converter infers which variables hold floats (float
literals, loops over float lists, float arithmetic) and prints those as Python does; image-test capture names like
`fNumber.1.0` depend on it.

## How equivalence is checked

`tests/gpu/suites/py2js.gpu.test.ts` (run `npm run py2js:corpus` first, which converts Falcor's scripts, test
scenes, image tests and the oracle scenes into `tests/gpu/generated/py2js`):

| Test | Checks |
| --- | --- |
| `Py2Js.sceneApiCoversPython` | every name a `.pyscene` sees has a JS counterpart |
| `Py2Js.mogwaiApiCoversPython` | every name a Mogwai script sees, and every attribute of `m`, has one |
| `Py2Js.scenesMatch` | all 66 loadable `.pyscene` files and their conversions build the same scene (stats, defines, bounds, cameras, lights) and render pixel-identical images |
| `Py2Js.graphScriptsMatch` | Falcor's graph scripts record the same commands and graphs |
| `Py2Js.imageTestScriptsMatch` | all 56 runnable image-test scripts record the same commands (ops, arguments, graphs, pass properties) |

The three image tests left out need NVIDIA-only passes (DLSS, OptiX) that neither language can create. The unit
tests in `packages/falcor/tests/py2js.test.ts` cover the runtime and the converter without a GPU.
