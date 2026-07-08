# web-falcor

[![CI](https://github.com/windingwind/web-falcor/actions/workflows/ci.yml/badge.svg)](https://github.com/windingwind/web-falcor/actions/workflows/ci.yml)

WebGPU-based reimplementation of [NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor)
targeting 1:1 feature parity where the web platform allows. **See the [design docs](docs/)**
for the framework design, the Falcor→web module mapping, and the feature parity matrix
(including features that are impossible in the browser and why).

![Amazon Lumberyard Bistro path-traced in the browser with web-falcor on WebGPU](docs/assets/teaser-bistro.jpg)

*Amazon Lumberyard Bistro (interior), path-traced with web-falcor on WebGPU — full material system,
emissive lighting, and software ray tracing, cross-validated against native Falcor's DXR output
(see the [parity matrix](docs/parity-matrix.md) and [oracle results](docs/testing.md)).*

## Layout

- `packages/falcor/` — core library, mirrors `Falcor/Source/Falcor`
- `packages/render-passes/` — render pass plugins, mirrors `Falcor/Source/RenderPasses`
- `packages/mogwai/` — browser application, mirrors `Falcor/Source/Mogwai`
- `packages/slang-compiler/` — build-time Slang→WGSL+reflection driver
- `Falcor/` — upstream shader sources (+ native oracle for tests); fetched, not committed
- `tools/` — Slang toolchains; fetched, not committed

## Quick start (use the library / run the app)

The runtime compiles Slang→WGSL in the browser, so it needs the upstream Falcor
shader **sources** and the slang-wasm compiler. `setup:web` fetches both from
GitHub at pinned versions — **no Falcor clone and no native build required**:

```sh
npm install
npm run setup:web   # fetch Falcor shader sources + slang-wasm (~30 MB, no clone)
npm run typecheck
npm run dev         # Mogwai dev server (needs a WebGPU browser)
```

`setup:web` does not fetch media/test scenes or the RTXDI/NanoVDB SDK headers.
The common passes (path tracer, tone mapper, accumulate, scene debugger, …)
build and run without them; RTXDI and GridVolume passes need the full setup.

### Example scenes

To try the app on real content, fetch Falcor's example scenes (also no clone —
they come from the same media bundle Falcor's `setup.sh` pulls):

```sh
npm run download:scenes                 # all bundled scenes (~120 MB)
npm run download:scenes -- --list       # list what's available
npm run download:scenes -- Arcade       # just one scene (Arcade, test_scenes, …)
npm run download:scenes -- Bistro       # a large ORCA scene by name
npm run download:scenes -- cornell-box  # a Bitterli pbrt-v4 scene
npm run download:scenes -- --all        # everything, incl. the big ORCA scenes
```

Scenes land under `Falcor/media/<Scene>/`, which the dev server serves at
`/Falcor/media/…`. The Mogwai viewer loads `test_scenes/cornell_box.pyscene` by
default.

Three kinds of scene are covered by the one command:

- **Bundled scenes** (`Arcade`, `test_scenes`, `inv_rendering_scenes`,
  `test_images`) come from Falcor's official media bundle — one ~120 MB archive,
  so naming scenes only limits what is written to disk, not the download.
- **ORCA showcase scenes** (`Bistro`, `EmeraldSquare`, `SunTemple`, `ZeroDay`)
  are large individual downloads (~0.3–1 GB each) fetched from
  [NVIDIA ORCA](https://developer.nvidia.com/orca). They are opt-in: named
  explicitly or via `--all`; the plain default only pulls the bundled scenes.
- **Bitterli pbrt-v4 scenes** (`cornell-box`, `veach-mis`, `kitchen`,
  `staircase`, … and any other name from the
  [Rendering Resources](https://benedikt-bitterli.me/resources) page) are loaded
  through web-falcor's pbrt-v4 importer (a port of Falcor's `PBRTImporter`
  subset). `cornell-box` and `veach-mis` are GPU-verified end to end; other
  scenes load best-effort (unsupported materials/textures fall back with a
  warning).

Extracting needs a `unzip` and/or 7-Zip CLI (`p7zip-full` / `7-zip`); the script
prints install hints if one is missing.

To load a pbrt scene in the viewer, point it at the `.pbrt` file, e.g.
`Falcor/media/cornell-box/scene-v4.pbrt` (the importer dispatches on the
`.pbrt` extension).

## Full setup (develop + run the GPU/oracle tests)

The GPU image tests diff against **native Falcor** captures, which need the
upstream clone built and its media tree. In addition to the quick start:

```sh
# Upstream Falcor clone + media + packman SDK deps, then the native oracle build
git clone https://github.com/NVIDIAGameWorks/Falcor.git && (cd Falcor && ./setup.sh)
cmake --preset linux-gcc -S Falcor && cmake --build Falcor/build/linux-gcc -j16

# Native Slang toolchain (build-time Slang→WGSL for the shader compiler)
mkdir -p tools/slang && curl -L https://github.com/shader-slang/slang/releases/download/v2026.12.2/slang-2026.12.2-linux-x86_64.tar.gz | tar xz -C tools/slang
```

## Tests

```sh
npm test            # unit suite (Node); CI runs this. Media-dependent tests
                    # skip automatically when the Falcor media tree is absent.
npm run test:gpu    # GPU image tests vs native oracles — needs hardware WebGPU
                    # (Vulkan under xvfb) and the full setup above. Local only.
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the quick-start
setup path plus `typecheck` + `npm test` on every push and PR — so the badge
above also verifies that the no-clone setup keeps working.

## License

web-falcor's own code is licensed under the [MIT license](LICENSE).

It is a reimplementation derived from [NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor)
(BSD-3-Clause). That upstream license and copyright notice are retained in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), and each shader derived from an
upstream Falcor source is marked in its `WebFalcor/Overrides` header.
