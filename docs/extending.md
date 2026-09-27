# Extending web-falcor

There are three ways to extend web-falcor. Pick the one that matches what your
change touches:

| Your change… | Route | Where it lives |
| --- | --- | --- |
| adds new passes or importers, using only the existing API | **plugin** (the default) | your own repo, cloned into `plugins/<Name>/`, loaded with `?plugin=` |
| ports a pass from upstream Falcor, or adds a pass every user should get | **in-tree pass** | `packages/render-passes/src/<Pass>/`, contributed by pull request |
| needs the framework itself to change (materials, scene, lights, `Core/API`, Slang→WGSL fixes) | **core change** | `packages/falcor/src/`, `packages/falcor/shaders/`, by pull request |

## Release your work as a plugin

If your work doesn't change the core, **write it as a plugin and release it from
your own repo.** Don't fork web-falcor or send it in as a pull request. A plugin
repo contains only your files, so:

- users try it from a link to the online demo with `?plugin=<url of its build>`,
  or `git clone` it into `plugins/` of an unmodified web-falcor checkout;
- you release on your own schedule, without waiting for a web-falcor review or
  merging web-falcor's updates into a fork;
- when web-falcor updates, users pull the new version and keep your plugin.

The template shows what a plugin can reach: `RenderPass`, `ComputePass`,
`RenderPassReflection`, the scene, the UI widgets, and every Falcor shader module
through Slang `import`. Everything exported from `@web-falcor/falcor` is available.

Change web-falcor itself only when a plugin can't do the job. For example, the
pass might need an API that doesn't exist, or a bug might need fixing in the core.
Keep that pull request to the core change and release the pass that needed it as
a plugin. The exception is a pass that belongs in web-falcor itself, which mostly
means a port of an upstream Falcor pass: web-falcor aims at parity with native
Falcor, so those passes go in-tree.

Most research work is a new render pass, so start with the scaffold below.

## Scaffold a pass

```sh
npx @web-falcor/mogwai new MyPass     # plugin in its own project, no checkout needed (see npm.md)
npm run new:pass -- MyPass --plugin   # plugin, under plugins/MyPass/ of this checkout
npm run new:pass -- MyPass            # in-tree, under packages/render-passes/src/MyPass/
```

The first command is all a plugin author needs: it uses the published packages, and `npm run dev` there serves
the prebuilt viewer with the plugin loaded ([npm.md](npm.md#write-a-plugin)). The other two are for work inside
this repo.

Either command writes three files: a working compute pass (`MyPass.ts`), its
shader (`MyPass.cs.slang`), and a Mogwai graph script (`MyPass.graph.js`) that runs the
path tracer and feeds the result through your pass. It then prints a URL that opens
the graph in the viewer:

```sh
npm run dev
# plugin:
http://localhost:5173/?scene=test_scenes/cornell_box.pyscene&script=/plugins/MyPass/MyPass.graph.js&plugin=/plugins/MyPass/MyPass.ts
# in-tree:
http://localhost:5173/?scene=test_scenes/cornell_box.pyscene&script=/packages/render-passes/src/MyPass/MyPass.graph.js
```

A plugin also gets a `README.md` that tells its users how to install it.

The template pass multiplies its input by a `scale` property. That property is
exposed as a slider in the pass UI and can be set from a graph script
(`createPass("MyPass", { scale: 0.5 })`, or `{'scale': 0.5}` in Python). Replace the body of `execute()` and the
shader with your own code. After editing an in-tree pass's shader, press **F6** in
the viewer to recompile without reloading the page. A plugin's shader is compiled
from the text loaded with the module, so reload the page instead.

To open the scaffold in the graph editor instead, load any scene, open **Graph**,
and pick `MyPass` from the pass list.

## Plugins

A plugin is an ES module whose top level calls `registerRenderPass(...)` and/or
`registerImporter(extensions, parser)`. Mogwai loads one plugin for each `?plugin=`
URL parameter (the parameter can be repeated), before it loads the scene or graph.
The console lists what each plugin registered:

```text
(Info) Loaded plugin /plugins/MyPass/MyPass.ts: passes [MyPass], importers []
```

Outside Mogwai, call `loadPluginLibrary(url)` (or `loadRenderPassLibrary(url)` if
the plugin only adds passes) from `@web-falcor/falcor`. Each returns the types it
added.

- **Imports.** A plugin imports `@web-falcor/falcor` (and `@web-falcor/render-passes`, to extend an existing pass)
  by package name and gets the whole API. On the dev server Vite resolves these names. In a prebuilt viewer
  (the Pages demo, `npx @web-falcor/mogwai`) an import map points them at the viewer's own copies, so a plugin
  shares the viewer's registry and classes. A `blob:` or `data:` module can also use the older
  `globalThis.webFalcorPlugins` hook, which only offers `registerRenderPass` and `registerImporter`.
- **Shaders.** A plugin's shader is not in the shader list. The template imports the
  shader text with Vite's `?raw` suffix and compiles it as a string module:
  `ComputePass.create(device, { modules: [{ sources: [{ string: src, path: "Plugins/MyPass/MyPass.cs.slang" }] }] })`.
  String modules can still `import` any Falcor shader module.
- **Where it lives.** `plugins/` is gitignored, so each plugin is its own git repo cloned into
  `plugins/<Name>/`. To publish a scaffolded plugin, run `git init` in its folder and push it. The dev server only
  serves files inside this repo, so clone the plugin there instead of symlinking it in from elsewhere.
- **Importers.** An importer's parser has the `ImporterPlugin` signature in
  [`ImporterRegistry.ts`](../packages/falcor/src/Scene/Importer/ImporterRegistry.ts).
  `importScene` tries registered importers before the built-in ones, as native does.

### Ship a prebuilt plugin

The scaffold includes a `package.json` and `vite.config.js`. Running `npm run build` in the plugin folder writes a
single `dist/<Name>.js`, with the shader inlined and `@web-falcor/*` left as imports for the viewer to provide.
Host that file anywhere that serves JavaScript with CORS allowed (GitHub Pages, a GitHub release, npm through a CDN)
and anyone can try it without installing anything:

```text
https://windingwind.github.io/web-falcor/?plugin=https://<you>.github.io/my-pass/MyPass.js&script=https://<you>.github.io/my-pass/MyPass.graph.js
```

Locally, `npx @web-falcor/mogwai --plugins ./plugins` serves the same build at `/plugins/`.

## In-tree passes

Use this route for passes that belong in web-falcor itself (see
[Release your work as a plugin](#release-your-work-as-a-plugin)); they go in by
pull request. The scaffold does the following for an in-tree pass (to add a pass by hand, do the
same steps). `node scripts/make-new-render-pass.mjs <Name>`, a port of Falcor's
`make_new_render_pass.py`, does steps 1 and 2 with the shaderless `RenderPassTemplate`:

1. Creates `packages/render-passes/src/MyPass/MyPass.ts`. Its module ends with
   `registerRenderPass("MyPass", factory)`, which is web-falcor's version of native's
   `registerPlugin()`.
2. Adds `export * from "./MyPass/MyPass.js";` to
   [`packages/render-passes/src/index.ts`](../packages/render-passes/src/index.ts).
   Importing `@web-falcor/render-passes` registers every pass that index exports.
3. Registers the shader in `packages/falcor/shaders/generated/shader-file-list.json`.
   The browser can only load shader files that are listed there. Shaders next to an
   in-tree pass get the key `RenderPasses/MyPass/MyPass.cs.slang`, which is the
   same layout native Falcor deploys, so the pass loads its shader with
   `ComputePass.create(device, { path: "RenderPasses/MyPass/MyPass.cs.slang" })`.

**Whenever you add, rename or delete a `.slang` file** in `packages/render-passes/src/`
or `packages/falcor/shaders/`, run:

```sh
npm run shaders:list
```

This command rewrites only the list entries for shaders owned by this repo, so it
works in a quick-start checkout without a Falcor clone. A new shader must not have
the same path as an upstream shader under `Falcor/Source/RenderPasses/`; the
command stops with an error if it does.

Porting an upstream pass, the main reason to go in-tree, works differently. Its
shaders already exist under `Falcor/Source/RenderPasses/<Pass>/`, so reference them
by their upstream path and don't copy them. If a shader needs changes to compile to WGSL, see
[Upstream shaders that don't compile](#upstream-shaders-that-dont-compile).

## Changing the core

Anything that isn't a pass or an importer is a change to
[`packages/falcor/src/`](../packages/falcor/src/). Its layout follows
`Falcor/Source/Falcor` one-to-one (see [module-mapping.md](module-mapping.md)), so
the web version of a native file is found by its native path.

### Upstream shaders that don't compile

Never edit files under `Falcor/Source/`: they are fetched copies of upstream at a
pinned commit. When an upstream shader uses something WGSL can't express, write a
replacement with the same interface (entry points, defines, cbuffers, binding names)
under `packages/falcor/shaders/WebFalcor/Overrides/<same path>`. Then map it in
[`ShaderOverrides.ts`](../packages/falcor/src/Core/Program/ShaderOverrides.ts) and
run `npm run shaders:list`. Host code keeps using the upstream path. Explain in the
override file how it differs from upstream. [shader-system.md](shader-system.md) §4
covers the common rewrites.

## Testing a change

- `npm run typecheck` and `npm test` must pass (CI runs both).
- GPU behaviour is tested with `npm run test:gpu`, whose suites live in
  [`tests/gpu/suites/`](../tests/gpu/suites/). A port of a native pass should get
  an image comparison against a native oracle. A new pass should at least have a
  test that renders it and checks its output. See [testing.md](testing.md).
- If your change affects parity with native Falcor, update its row in
  [parity-matrix.md](parity-matrix.md).
