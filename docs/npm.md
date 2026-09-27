# Using the npm packages

web-falcor is published as three packages:

| Package | Contents | Use it to… |
| --- | --- | --- |
| `@web-falcor/falcor` | the core library, compiled to JS with type declarations | build your own app on Falcor's framework |
| `@web-falcor/render-passes` | Falcor's render passes; importing it registers them | use `PathTracer`, `GBufferRT`, `NRD`, … in your graphs |
| `@web-falcor/mogwai` | the prebuilt Mogwai viewer, its runtime assets, the `web-falcor` CLI and a Vite plugin | run the viewer without a checkout, or supply the assets to your app |

Everything needs a browser with WebGPU and at least 16 storage buffers per shader stage
(`maxStorageBuffersPerShaderStage`). Safari, and Chrome or Edge on Windows and Linux, usually have that. Chrome and
Edge on macOS allow 10, too few for the path tracer; passes that need more stop with an error naming the pass and
the limit (see the [README](../README.md)). On a Mac, use Safari.

## Run the viewer

```sh
npx @web-falcor/mogwai                       # http://localhost:5173/, with the default Cornell box
npx @web-falcor/mogwai --media ./my-media    # serve your scenes at /Falcor/media/
npx @web-falcor/mogwai --plugins ./plugins   # serve plugins at /plugins/
```

The viewer takes the same URL parameters as the dev server: `?scene=` (a path under `/Falcor/media/`),
`?script=` (a render-graph script, `.py` or `.js`; see [scripting.md](scripting.md)) and `?plugin=` (a plugin module,
repeatable).

## Write a plugin

A plugin adds render passes (or scene importers) to the viewer without a web-falcor checkout:

```sh
npx @web-falcor/mogwai new MyPass   # scaffolds ./my-pass/: a compute pass, its shader, a graph script
cd my-pass
npm install
npm run dev                         # the prebuilt viewer with the plugin loaded, on the Cornell box
```

Edit `MyPass.ts` and `MyPass.cs.slang` and reload the page. `npm run typecheck` checks the pass against
`@web-falcor/falcor`'s types. `npm run build` writes `dist/MyPass.js`. Any viewer loads it with `?plugin=<its URL>`,
including the online demo and `npx @web-falcor/mogwai --plugins <dir>`.

`web-falcor dev` runs Vite over the project and serves the prebuilt viewer beside it. The plugin's
`@web-falcor/falcor` and `@web-falcor/render-passes` imports resolve to the viewer's own modules, as in a built
viewer. This keeps a second copy of the library out of the page; with one, the passes would register where the
viewer never looks. The `webFalcor` field of the plugin's `package.json` sets the scene, plugin and graph script that
`dev` opens.

## Build an app on the library

```sh
npm install @web-falcor/falcor @web-falcor/render-passes @web-falcor/mogwai vite
```

The library compiles Slang to WGSL in the browser, so at runtime it loads assets that don't fit in a JS bundle:
Falcor's shader sources, the Slang compiler (slang-wasm), Pyodide for Python scene scripts, and some wasm decoders.
They ship in `@web-falcor/mogwai`. With Vite, its plugin serves them in dev and copies them into the build:

```js
// vite.config.js
import { defineConfig } from "vite";
import { webFalcor } from "@web-falcor/mogwai/vite";

export default defineConfig({ plugins: [webFalcor()] });
```

The plugin also keeps the packages out of Vite's dependency pre-bundling, which breaks the relative URLs of their
wasm modules and worker. It also sets the build target to ES2022 unless you set one.

Then point the library at the assets before loading anything:

```ts
import { Device, RenderGraph, assetUrl, createPass, initProgramSystem, initScripting, kProjectMediaUrl, presentToCanvas, runSceneScript, setAssetBase } from "@web-falcor/falcor";
import "@web-falcor/render-passes";

setAssetBase(import.meta.env.BASE_URL + "web-falcor/");

const canvas = document.querySelector("canvas")!;
const device = await Device.create();
const context = canvas.getContext("webgpu") as GPUCanvasContext;
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device: device.gpuDevice, format });

await initProgramSystem(device); // shader sources + slang-wasm
await initScripting(assetUrl("/node_modules/pyodide")); // for .pyscene files

const sceneUrl = `${kProjectMediaUrl}/test_scenes/cornell_box.pyscene`;
const scene = await runSceneScript(device, await (await fetch(sceneUrl)).text(), `${kProjectMediaUrl}/test_scenes`, { path: sceneUrl });
scene.camera.setAspectRatio(canvas.width / canvas.height);

const graph = new RenderGraph(device, "PathTracer");
graph.onResize(canvas.width, canvas.height);
graph.addPass(createPass(device, "VBufferRT"), "VBufferRT");
graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1 }), "PathTracer");
graph.addPass(createPass(device, "AccumulatePass"), "Accumulate");
graph.addPass(createPass(device, "ToneMapper"), "ToneMapper");
graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
graph.addEdge("PathTracer.color", "Accumulate.input");
graph.addEdge("Accumulate.output", "ToneMapper.src");
graph.markOutput("ToneMapper.dst");
graph.setScene(scene);

const frame = () => {
    graph.execute(device.renderContext);
    presentToCanvas(device, graph.getOutput("ToneMapper.dst")!, context.getCurrentTexture(), format);
    requestAnimationFrame(frame);
};
frame();
```

Without Vite, copy the assets with `npx web-falcor assets <dir>`, serve that directory, and call
`setAssetBase("<its URL>/")`. The assets include only the Cornell box scene. Put your own scenes anywhere you like
and pass their URLs to the scene loaders.

## What loads from elsewhere

The NRD and RTXDI shader sources and the NanoVDB header are not in the packages, because their licences don't
allow redistributing the source. The library fetches them from the vendors' GitHub repositories at the pinned
versions when a pass needs them. `NRDPass` applies web-falcor's WGSL patches to NRD's shaders in the browser. An app
that uses these passes therefore needs access to `raw.githubusercontent.com`. See
[THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

## Releasing (maintainers)

The packages are built from the workspace by `npm run build:packages`, which stages them in `out/npm/`. The
workspace itself keeps resolving the TypeScript sources. `.github/workflows/release.yml` publishes all three when a
`vX.Y.Z` tag is pushed. The tag must match the `version` in the root `package.json` and in every
`packages/*/package.json`, which are bumped together:

```sh
npm version 0.2.0 --workspaces --include-workspace-root --no-git-tag-version
git add -A && git commit -m "chore: release 0.2.0"   # -A: a plain -am would skip new files
git tag v0.2.0
git push && git push origin v0.2.0
```

The workflow publishes through npm's trusted publishing (OIDC), so there is no token to store. Each package lists
this repo's `release.yml` as its trusted publisher (npmjs.com → package → Settings → Trusted publishing → GitHub
Actions: owner `windingwind`, repository `web-falcor`, workflow `release.yml`). npm only lets you configure that on
a package that exists, so a new package's first version is published by hand:

```sh
npm login
npm run build:packages
for p in falcor render-passes mogwai; do npm publish out/npm/$p --access public; done
```
