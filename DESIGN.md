# web-falcor — Framework Design

A WebGPU-based reimplementation of [NVIDIA Falcor 8.0](https://github.com/NVIDIAGameWorks/Falcor)
targeting **exact 1:1 feature parity wherever the web platform allows**, with every
gap explicitly marked and explained.

Reference: upstream clone at `./Falcor` (commit `eb540f67`, built natively at
`Falcor/build/linux-gcc` for ground-truth comparison), research fork at `../Falcor`.

---

## 1. Goals and ground rules

1. **1:1 parity as the default.** Every Falcor class, render pass, shader module and
   script-visible API gets a web counterpart with the same name, same semantics, and —
   where meaningful — the same file layout (`Source/Falcor/Scene/Camera/Camera.h` →
   `packages/falcor/src/Scene/Camera/Camera.ts`).
2. **Reuse Falcor's shader library verbatim.** Falcor's value is concentrated in its
   257 `.slang` files (materials, BSDFs, samplers, light sampling, SDFs, path tracing).
   We compile the *unmodified upstream sources* with Slang's WGSL backend rather than
   hand-porting to WGSL. Verified working: Slang v2026.12.2 compiles
   `Utils.Math.*` / `Utils.Sampling.*` modules directly to WGSL
   (see `packages/falcor/shaders/SanityCompute.cs.slang` → `generated/`).
3. **Feature-gap honesty.** Everything that cannot work in the browser is marked in the
   parity matrix (§8) with one of:
   - ✅ **Portable** — direct implementation, same behavior.
   - 🟡 **Emulated** — same API and observable behavior, different mechanism
     (e.g. software ray tracing in compute).
   - 🔶 **Replaced** — same purpose, different technology (e.g. Python → TypeScript
     scripting), API kept shape-compatible.
   - ❌ **Impossible** — cannot be provided on the web platform at all; API exists but
     throws `UnsupportedFeatureError` with a pointer to this document.
4. **Verifiability.** The native `./Falcor` build is the oracle: image regression tests
   render the same scene + render graph in native Falcor and in web-falcor (headless
   Chromium on this host's RTX 5090) and compare with Falcor's own `ImageCompare`
   metrics (MSE / FLIP).

## 2. Platform decision: WebGPU primary, WebGL2 out of scope for parity

Falcor is compute-first: skinning, light-BVH build/refit, environment-map sampling
setup, parallel reduction, prefix sum, bitonic sort, SDF brick building, path tracing —
all compute shaders. **WebGL2 has no compute shaders** (the `WEBGL_compute` effort was
abandoned in 2020 in favor of WebGPU), no storage buffers, no arbitrary-format image
writes, and no indirect dispatch.

Consequences:

- **WebGPU is the sole full-parity backend.** `Device` (mirroring `Core/API/Device`)
  abstracts the backend as Falcor abstracts D3D12/Vulkan via slang-gfx.
- **WebGL2 fallback is possible only for a raster-only subset** (GBufferRaster,
  ForwardLighting-style passes, ToneMapper, blits) and would require a second shader
  pipeline (Slang → GLSL 300 es, which Slang supports). It is *designed for* (the
  `DeviceType.WebGL2` enum and the format/binding abstractions keep the door open) but
  **not implemented in the initial milestones** — implementing ~20 % of Falcor twice
  before the WebGPU path is complete would be wasted motion. It is marked per-feature
  in the parity matrix.

## 3. Architecture overview

Layering mirrors Falcor exactly; each box is a package/module with the same
responsibilities as its native counterpart:

```
┌──────────────────────────────────────────────────────────────────────┐
│  apps: Mogwai (browser app)  ·  FalcorTest (headless)  ·  Samples    │  packages/mogwai
├──────────────────────────────────────────────────────────────────────┤
│  RenderPasses (plugins): GBuffer, PathTracer, ToneMapper, TAA, …     │  packages/render-passes
├──────────────────────────────────────────────────────────────────────┤
│  RenderGraph: RenderGraph · RenderPass · Reflection · Compiler ·     │
│               ResourceCache · Exe · Import/Export (graph scripts)    │
├───────────────┬───────────────────┬──────────────────────────────────┤
│  Scene        │  Rendering        │  Utils                           │
│  SceneBuilder │  Lights/EnvMap    │  Math · Image · UI · Timing      │  packages/falcor
│  Materials    │  Materials(BSDF)  │  Sampling · Algorithm · Debug    │
│  Lights/Anim  │  Volumes          │  Scripting · Settings · Color    │
│  SDFs/Volumes │  SoftwareRT ★     │  SDF · Geometry · Neural         │
├───────────────┴───────────────────┴──────────────────────────────────┤
│  Core: API (Device, Buffer, Texture, Contexts, …) · Program          │
│        (ProgramManager, Reflection, ParameterBlock/ShaderVar) ·      │
│        Pass · State · Platform (browser) · Object/Error/Plugin       │
├──────────────────────────────────────────────────────────────────────┤
│  Shader system: Falcor .slang library (unmodified) + WebFalcor       │
│  platform shims  ──slangc──▶  WGSL + reflection JSON                 │  packages/slang-compiler
├──────────────────────────────────────────────────────────────────────┤
│  WebGPU (browser)              [WebGL2: raster-only subset, future]  │
└──────────────────────────────────────────────────────────────────────┘
     ★ SoftwareRT replaces DXR/VK ray tracing — see §5.
```

Repository layout:

```
web-falcor/
├── Falcor/                    # upstream clone: shader source of truth + native oracle
├── tools/slang/               # Slang v2026.12.2 (slangc with WGSL backend)
├── packages/
│   ├── falcor/                # @web-falcor/falcor — core library, mirrors Source/Falcor
│   │   ├── src/{Core,Scene,Rendering,RenderGraph,Utils,DiffRendering}/
│   │   └── shaders/           # web-falcor-owned .slang (shims) + generated WGSL
│   ├── render-passes/         # @web-falcor/render-passes — mirrors Source/RenderPasses
│   ├── mogwai/                # @web-falcor/mogwai — browser app, mirrors Source/Mogwai
│   └── slang-compiler/        # build-time slangc driver (→ runtime slang-wasm later)
├── tests/                     # unit + image-regression harness (native oracle diffing)
└── DESIGN.md                  # this document
```

## 4. Shader system (the load-bearing design decision)

### 4.1 Compilation pipeline

Falcor compiles Slang → DXIL/SPIR-V at runtime through slang-gfx. web-falcor compiles
Slang → **WGSL** with the same compiler front-end, in two modes:

1. **Build-time (primary)**: `packages/slang-compiler` drives native `slangc` over a
   shader manifest. Each entry (source, entry point, stage, defines, type conformances)
   produces `<name>.wgsl` + `<name>.reflection.json`. Verified end-to-end on this host.
2. **Runtime (for parity with Falcor's dynamic defines)**: Falcor specializes shaders
   at runtime via `DefineList`, type conformances, and scene-generated code
   (`MaterialSystem` emits Slang for registered material types; `Scene` defines control
   geometry types, light counts, etc.). A build-time-only pipeline cannot cover this.
   The official **slang-wasm** build (published per-release, e.g.
   `slang-2026.12.2-wasm.zip`, the same artifact powering the Slang Playground) runs the
   identical compiler in the browser (and in Node for tests). The `ProgramManager`
   mirrors Falcor's: program (files + defines + conformances) → hash → cache lookup →
   compile on miss. Ahead-of-time-compiled variants from the manifest pre-seed the cache
   so common paths never pay WASM compile time.

### 4.2 Reflection-driven binding (`ParameterBlock` / `ShaderVar`)

Falcor's host↔shader glue is its reflection system: `ProgramReflection` walks Slang's
reflection API; `ParameterBlock` lays out constant data and resource bindings;
`ShaderVar` gives `var["gScene"]["camera"]["viewMat"] = m` style access.

web-falcor keeps this design 1:1:

- Slang reflection JSON (build-time) or the slang-wasm reflection API (runtime)
  populates `ProgramReflection` — types, struct offsets (std140 for uniform, std430
  for storage), binding indices, spaces → **bind groups**.
- `ParameterBlock` owns a CPU-side `ArrayBuffer` mirror of the uniform data plus the
  resource table; `ShaderVar` is a thin proxy (JS `Proxy` for the indexing syntax plus
  explicit `.setFloat3(...)` typed setters) writing through reflection offsets.
- Falcor's register `space`s map to WebGPU **bind group indices**; slangc's WGSL
  backend already emits `@group/@binding` consistently with its reflection output.
- WebGPU's 4-bind-group default limit vs. Falcor's arbitrary spaces: the
  `RootSignature`-equivalent packing lives in `ProgramKernels`, which flattens
  parameter blocks into ≤ 4 groups (scene = group 1, per-pass = group 0, material
  system = group 2, misc = group 3) and rewrites bindings via slangc's layout
  parameters. This is internal; the `ParameterBlock` API is unchanged.

### 4.3 WebFalcor platform shims (`packages/falcor/shaders/WebFalcor/`)

A small set of *web-owned* Slang modules that implement Falcor interfaces whose native
implementations use features WGSL lacks. Because Falcor is already coded against
interfaces (`SceneRayQueryInterface`, `ISampleGenerator`, …), upstream shader code does
not change — we swap the implementation module at import/conformance level:

| Native module | WGSL blocker | Shim |
|---|---|---|
| `Scene/Raytracing.slang` (DXR pipeline: TraceRay, hit shaders, payloads) | no RT pipeline in WGSL | `WebFalcor/SoftwareRT`: megakernel compute path — see §5 |
| `Scene/RaytracingInline.slang` (RayQuery) | no `rayQuery` in WGSL | same software BVH traversal, inlined; identical `SceneRayQuery` interface |
| `Utils/NVAPI.slang(.slangh)` (SER, special registers) | NVIDIA-only | no-op shim (SER is a perf hint; results identical) |
| 64-bit atomics (`AtomicAdd` on u64 in LightCollection/PixelStats) | WGSL has no i64/u64 | paired-u32 CAS emulation module |
| float atomics (DiffRendering gradient accumulation) | WGSL has no atomic\<f32\> | CAS-loop on `atomic<u32>` bitcast |
| `printf`-style `PixelDebug` GPU prints | no printf | already buffer-based in Falcor → portable; only the host decode changes |
| Wave intrinsics (`WaveActiveSum` etc. in WarpProfiler, some samplers) | WGSL `subgroups` feature (shipped, but optional) | subgroup ops when available; scalar/workgroup-shared fallback otherwise |

The manifest/`ProgramManager` selects shims through Slang's module search path and
type conformances — never by editing upstream files. `./Falcor` stays a pristine
checkout that doubles as the image-test oracle.

## 5. Ray tracing without RT hardware (the biggest emulation)

WebGPU has **no ray tracing API** (neither RT pipelines nor ray queries; proposals
exist but nothing is shipped in any browser). Falcor's flagship passes (PathTracer,
GBufferRT/VBufferRT, WhittedRayTracer, RTXDI) are all RT-based. Strategy:

1. **BVH construction in compute** (`Core/API/RtAccelerationStructure` keeps its API):
   LBVH build (Morton sort → Karras-style hierarchy → refit), two-level as in DXR:
   BLAS per mesh group / TLAS over instances, matching Falcor's BLAS grouping strategy
   so `InstanceID`/`GeometryIndex` semantics are preserved. Compaction and
   update(refit) flags honored.
2. **Traversal in compute**: a stack-based traversal kernel implementing
   `SceneRayQuery` (inline-query style, `TraceRayInline` semantics including
   `RAY_FLAG_*`, instance masks, alpha-test any-hit via material system hooks).
3. **RT pipelines emulated as megakernels**: Falcor's raygen/miss/closest-hit/any-hit
   modules are Slang functions; the shim links them into one compute kernel per
   `RtProgram`, with the `RtBindingTable`'s hit-group indexing lowered to a switch over
   material/geometry type (Slang link-time specialization keeps this static where
   possible). Recursion (Whitted) is bounded-depth loop-converted — same approach
   Falcor itself uses for inline-RT variants of its passes (`PathTracer` already has a
   compute path, which becomes the default).
4. **Performance is not parity-gated**: a 5090-class GPU runs compute path tracing at
   interactive rates, but we mark every RT feature 🟡 with "software BVH; expect
   single-digit× slower than native DXR".

## 6. Module-by-module mapping

### 6.1 `Core/` (mirrors `Source/Falcor/Core/`)

| Falcor | web-falcor | Notes |
|---|---|---|
| `Object` (intrusive refcount, `ref<T>`) | GC + explicit `destroy()` for GPU objects | JS is GC'd; deterministic release only where WebGPU needs it (buffers/textures). `ref<T>` not reproduced — documented divergence, no observable API change. |
| `Error` (exceptions) | `FalcorError` hierarchy + `UnsupportedFeatureError` | done (scaffold) |
| `Plugin` (dynamic .so/.dll) | ES-module dynamic `import()` registry | render passes/importers self-register, same `PluginManager` API |
| `AssetResolver` | URL/OPFS resolver with search paths | fetch()-backed; drag-&-drop and File System Access mounts |
| `SampleApp` / `Testbed` | browser main-loop (rAF) / headless (OffscreenCanvas) | same lifecycle callbacks (onLoad/onFrameRender/onResize/…) |
| `HotReloadFlags` | Vite HMR hooks → `ProgramManager.reloadAllPrograms()` | web is *better* here |
| **API/** `Device` | `GPUAdapter/GPUDevice` wrapper | done (scaffold); async factory |
| `Buffer`, `Texture`, `Sampler`, `ResourceViews` | `GPUBuffer/GPUTexture/GPUSampler/GPUTextureView` | typed/structured/raw buffers map to storage buffers; counter-buffers emulated (no D3D UAV counters) via side-buffer + atomics |
| `Formats` | `ResourceFormat` → `GPUTextureFormat` | done (scaffold); ❌ gaps: 24-bit depth readback, RGB32F *texture* (buffer only), A8/L8 legacy |
| `CopyContext/ComputeContext/RenderContext` | one `CommandEncoder` wrapper hierarchy, same class split | `blit()` via cached fullscreen pipeline, `resolveSubresource`, `clearUAV` via compute |
| `FBO`, `VAO` | render-pass descriptor / vertex-layout caches | same API, lowered at pipeline-creation time |
| `GraphicsStateObject/ComputeStateObject` | `GPURenderPipeline/GPUComputePipeline` + async cache | Falcor's PSO hash-cache pattern kept |
| `Fence`, `FencedPool` | `onSubmittedWorkDone` promises + frame-ring | no user-visible timeline semaphores in WebGPU; API preserved, values internal |
| `GpuMemoryHeap` (upload/readback rings) | ring of `mappedAtCreation`/`MAP_READ` staging buffers | same role; readback is async-only (§9 divergences) |
| `QueryHeap`, `GpuTimer` | `GPUQuerySet` (timestamp) | ✅ behind `timestamp-query` feature (available on target browsers) |
| `Swapchain` | canvas context `configure()` | vsync/HDR: `toneMapping`/`colorSpace` where Chrome supports; ❌ exclusive fullscreen/refresh control |
| `RtAccelerationStructure`, `RtStateObject`, `ShaderTable` | software BVH + megakernel + virtual SBT | 🟡 §5 |
| `Aftermath`, `NvApiExDesc`, D3D12 descriptor classes, CUDA interop in `CopyContext` | — | ❌ NVIDIA/D3D12-specific (see matrix) |
| **Program/** all 13 files | same classes | §4; `DefineList`, type conformances, `RtBindingTable` all preserved |
| **Pass/** `ComputePass/RasterPass/FullScreenPass` | same | FullScreenPass uses the same vertex-in-shader trick |
| **State/** `GraphicsState/ComputeState` | same mutable-state + lazy-PSO-resolve design | |
| **Platform/** `Window` (GLFW) | canvas + Pointer/Keyboard events, `ResizeObserver` | same `Window::ICallbacks`; `MonitorInfo` from `window.screen` (limited) |
| `OS.h` (file dialogs, env, processes) | File System Access API pickers; ❌ processes, ❌ env | `MemoryMappedFile` → streamed `fetch`/OPFS; `LockFile` → Web Locks API |

### 6.2 `Scene/`

All host classes port 1:1 (they are data management + compute dispatch, no exotic API
use): `Scene`, `SceneBuilder`, `SceneCache` (→ IndexedDB/OPFS), `Camera(+Controller)`,
`Light` hierarchy, `LightCollection` (emissive triangle extraction in compute),
`EnvMap`, `LightProfile` (IES), `MaterialSystem` + all material types (Standard, Cloth,
Hair, MERL, MERLMix, RGL, all six PBRT materials), `Animation` (+ GPU skinning, morph
targets, vertex caches), `CurveTessellation`, `SDFGrid` ×4 back-ends (NDSDF, SVS, SBS,
SVO — all pure compute), `GridVolume`/`Grid` (NanoVDB parsing in TS; BC4-in-shader
decode as native), `TriangleMesh`, `HitInfo`, `Transform`.

Scene GPU access (`Scene.slang`, `SceneBlock`, geometry/material/light buffers) is the
same reflection-bound parameter block. The **bindless problem**: Falcor binds all
material textures as an unbounded descriptor array (`Texture2D gTextures[]`).
WGSL has no runtime-sized binding arrays (`binding_array` exists only in native wgpu,
not the browser). Mitigation, in order:
1. group material textures by (format, size-class) into **texture-2d-arrays** with a
   per-texture layer index in `TextureHandle` (transparent — `TextureHandle.slang`
   is already an abstraction);
2. large-scene overflow → mip-biased atlas fallback;
3. marked 🟡 with limits documented (`maxSampledTexturesPerShaderStage` typically 16;
   arrays count as one binding each).

Importers (plugin package, like upstream):
- **glTF/OBJ/PLY**: native TS loaders (glTF is the primary web format) — replaces the
  Assimp path for these formats.
- **Assimp (FBX, DAE, …)**: `assimpjs` (official Emscripten build) 🔶.
- **PBRT / Mitsuba**: TS ports of Falcor's parsers ✅ (pure text parsing).
- **USD**: 🔶 via Autodesk/Pixar `usd-wasm`; heavy (tens of MB WASM) and lags native
  OpenUSD → optional plugin, off by default. Marked partial: usdz + core schemas work;
  full nv-usd parity not promised.
- **PythonImporter**: 🔶 via Pyodide (see §6.7).

### 6.3 `Rendering/`

| Subsystem | Status | Notes |
|---|---|---|
| `Lights/` Emissive samplers (Uniform/Power/LightBVH), `LightBVH(+Builder/Refit)`, `EnvMapSampler` | ✅ | pure Slang + compute; LightBVH build is compute shaders already |
| `Materials/` all BSDF modules (Lambert, OrenNayar, Disney/Frostbite diffuse, GGX iso/aniso, StandardBSDF, Sheen, Hair Chiang16, Cloth, MERL/RGL, PBRT set, LayeredBSDF, Fresnel/Microfacet/NDF, TexLOD) | ✅ | compile as-is to WGSL; TexLOD ray-cone variants fine, ray-diff variants fine |
| `Volumes/` GridVolumeSampler, phase functions | ✅ | NanoVDB traversal is shader code, portable |
| `RTXDI/` | 🟡 | RTXDI **SDK** is open source (BSD): the resampling shaders port; visibility rays go through SoftwareRT. Full ReSTIR DI parity feasible but scheduled late |
| `Utils/PixelStats` | ✅ | needs 64-bit-atomic shim (paired u32) |

### 6.4 `RenderGraph/`

Pure host-side logic — ports 1:1 with no platform caveats: `RenderGraph`,
`RenderPass`, `RenderPassReflection`, `RenderGraphCompiler` (pass order, resource
lifetime, field compatibility), `ResourceCache` (transient pool honoring
`RenderPassHelpers::IOSize`), `RenderGraphExe`, `RenderGraphIR`,
import/export of graph scripts (§6.7), `RenderGraphUI` (graph editor) on the web UI
stack (§6.6).

### 6.5 `RenderPasses/` — see parity matrix §8.2 for all 31.

### 6.6 UI (`Utils/UI`, Mogwai)

Falcor uses Dear ImGui (+ ImGuizmo). web-falcor uses **Dear ImGui compiled to WASM**
(`jsimgui` / imgui-wasm bindings with the WebGPU backend) so widget code translates
1:1 (`Gui::Widgets` API preserved), including ProfilerUI, PixelZoom, SpectrumUI,
TextRenderer, and the RenderGraphUI node editor. Fallback plan if the binding layer
proves brittle: same `Gui` API over Tweakpane/custom DOM (uglier, zero-WASM).
Mogwai itself (menus, graph loading, FrameCapture → PNG/EXR download, VideoCapture →
WebCodecs, TimingCapture → JSON) is a straightforward port.

### 6.7 Scripting & Python API

Falcor embeds Python (pybind11): render-graph scripts, Mogwai console, `Testbed`
notebooks. Browser reality: no CPython. Design:

1. **Primary 🔶**: a TypeScript scripting API that is *shape-identical* to the Python
   one (`createPass("ToneMapper", {autoExposure: false})`, `g.addEdge(...)`,
   `m.addGraph(...)`) — Falcor's own graph `.py` files are ~declarative Python that
   maps 1:1 onto this.
2. **Graph-script compatibility ✅**: a small parser executes upstream, unmodified
   render-graph `.py` files (the subset actually used by all 40+ graph files in
   `tests/image_tests/renderpasses/graphs/`) so existing content Just Works.
3. **Full Python 🔶 (optional plugin)**: Pyodide runs real CPython in the browser with
   a `falcor` bridge module for arbitrary scripts (Mogwai console, PythonImporter).
   numpy interop works via Pyodide; **PyTorch does not exist in the browser** →
   `test_pytorch`-style workflows are ❌ (closest substitute: ONNX Runtime Web /
   tfjs, out of scope).

### 6.8 `Utils/`

Everything ports ✅ unless noted: Math (TS vector/matrix/quaternion lib with Falcor's
exact conventions + the `.slang` math modules as-is), Sampling (all generators are
Slang → as-is; CPU sample patterns trivial), Algorithm (ParallelReduction, PrefixSum,
BitonicSort — compute, as-is), Color/Spectrum, Geometry, SDF draw utils, Image
(PNG/JPG via browser codecs, **EXR/DDS/HDR via TS/WASM codecs**, NVTT-based BC
encoding 🔶 → WASM encoder e.g. Binomial basis_universal or texture-compressor;
decode of BC is native via `texture-compression-bc`), TextureManager/async loader
(fetch + `createImageBitmap`), Timing (Clock/FrameRate/Profiler with timestamp
queries), Debug (PixelDebug ✅, WarpProfiler 🟡 subgroups-gated), Scripting (§6.7),
Settings (JSON + localStorage), CryptoUtils (SHA-1 → WebCrypto), Threading/TaskManager
(→ Web Workers pool; scene build off-main-thread), `CudaUtils` ❌.

### 6.9 `DiffRendering/` (WARDiffPathTracer)

Slang **autodiff is a compiler feature**, not an API feature — `fwd_diff`/`bwd_diff`
lower to plain compute code, so it compiles to WGSL. Gradient accumulation needs the
float-atomic CAS shim. The PyTorch training loop does not exist in-browser; gradients
are exposed as buffers (readable into JS / ONNX-web pipelines). Marked 🟡 (mechanism
works; ecosystem differs).

## 7. Testing strategy

1. **Unit tests (vitest, Node)**: math, reflection layout, graph compilation — CPU-only.
2. **GPU unit tests**: FalcorTest's GPU-unit-test pattern (dispatch kernel → readback →
   assert) ported; run in headless Chromium (`--headless=new --enable-unsafe-webgpu`)
   via Playwright on this host's RTX 5090. Falcor's `Testbed`-style tests map directly.
3. **Image regression**: for each graph in `tests/image_tests/renderpasses/graphs/`,
   render N frames native (`./Falcor/build/linux-gcc/bin/Debug/Mogwai --script ...
   --headless`) and in web-falcor (same graph file, same scene, fixed seeds), compare
   with `ImageCompare` (MSE + FLIP thresholds). Bit-exactness is *not* the bar
   (different fp contraction/driver); Falcor's own image-test tolerances are.
4. **Shader-compile CI gate**: every manifest entry must compile to WGSL and pass
   `wgpu`-level validation (via Chromium) on every commit.

### 7.1 Verified oracle results (web vs native hardware DXR, same GPU)

All comparisons render the identical scene/camera/seed natively (Mogwai
`--headless`, hardware Vulkan RT) and on web (WebGPU compute + software BVH),
then diff per-pixel (mean |Δ| over RGB; "bad" = pixels with any channel off by
more than 0.05). Suite: `npm run test:gpu` (53 GPU tests + 23 unit green).

| Oracle | Web pass under test | mean abs Δ | bad px (of 65536) |
|---|---|---|---|
| GBufferRT posW/texC | GBufferRaster (raster) | per-pixel match | 0 |
| GBufferRT posW | SceneRayQuery primary rays | 4.4e-6 | 0 |
| MinimalPathTracer, quad+point light | MinimalPathTracer (megakernel) | 9.5e-7 | 0 |
| MinimalPathTracer, textured quad | + TextureManager array | 1.4e-6 | 0 |
| MinimalPathTracer, env light+background | + EnvMap | 7.5e-5 | 0 |
| MinimalPathTracer, emissive two-quad | + emissive surfaces | 1.5e-4 | 14 (silhouette) |
| MinimalPathTracer, ClothMaterial | static material dispatch | 1.9e-5 | 0 |
| MinimalPathTracer, HairMaterial (Chiang16) | static material dispatch | 1.3e-3 | 0 |
| MinimalPathTracer, PBRTDiffuse | static material dispatch | 8.9e-6 | 0 |
| MinimalPathTracer, PBRTConductor (aniso GGX) | static material dispatch | 2.9e-5 | 0 |
| **PathTracer** (NEE+MIS, emissive uniform sampler) | full PathTracer port | 1.6e-4 | 13 (silhouette) |
| **PathTracer**, env light | + EnvMapSampler importance sampling | 7.4e-5 | 0 |
| **PathTracer**, emissive power sampler | + alias table (bit-replicated) | 1.6e-4 | 13 (silhouette) |
| `.pyscene` importScene path (Pyodide) | SceneBuilder bridge | 9.5e-7 | 0 |
| `.pyscene` builder-geometry path (Pyodide) | SceneBuilder bridge | 1.9e-5 | 0 |
| **PathTracer**, emissive LightBVH sampler | CPU LightBVH builder port | 1.6e-4 | 13 (silhouette) |
| PathTracer, 24-emitter cluster scene (LightBVH) | SAOH splits, multi-level bitmasks | 5.4e-6 | 0 |
| **upstream `cornell_box.pyscene`, unmodified** | full stack: Pyodide + PathTracer + LightBVH | 2.6e-4 | 49 (silhouette) |
| **upstream `sphere_array.pyscene`, unmodified** | 131k tris, metal/rough grid, HDR env sampling | 8.2e-4 | 267 (silhouette) |
| Delta reflection (perfect mirror, point + emissive lit) | delta-lobe paths | 6.1e-5 / 2.9e-5 | 0 / 1 |
| DistantLight minimal scene | distant-light packing/sampling | 1.0e-4 | 0 |
| **upstream `convergence_test.pyscene`, unmodified** | 16-material stress (mirrors/metals/glass/thin, 4 area lights) | 3.2e-2 ⚠ | 6338 — combined delta-lobe/caustic paths; isolated features all match, combination residual under investigation |
| **upstream image test `MinimalPathTracer.py` graph + `cornell_box.pyscene`, both unmodified** | full Mogwai workflow: graph script + scene script + 4-pass chain | 1.5e-4 | 72 (silhouette) |
| upstream image test `ToneMapping.py` (scene-less, PNG oracle) | ImageLoader + ACES + blit | sRGB MSE 1.2e-4 | — |
| upstream image test `VBufferRT.py` — depth / viewW / mask channels | V-buffer aux outputs | 1.1e-4 / 6.1e-5 / exact | 0 / 0 / 0 |
| upstream image test `GaussianBlur.py` (hdr ImageLoader, EXR oracle) | connectedResources-shaped IO + separable fullscreen blur + weights buffer | 4.2e-5 | 0 |
| upstream image test `CompositePass.py` (jpg+png ImageLoaders, EXR oracle) | scaled add compositing | 2.3e-3 † | 14 px > 0.02 |
| upstream image test `CrossFadePass.py` (jpg+png ImageLoaders, EXR oracle) | auto-fade frame counting (frame 0 ⇒ out = A) | 2.3e-3 † | 14 px > 0.02 |

| upstream image test `ColorMapPass.py` (hdr, PNG oracle) | Jet color map + auto-range (frame-0 static range, reduction consumed a frame later like native) | sRGB MSE 3.8e-7 | max 1 byte |
| upstream image test `SideBySide.py` (jpg ×2 raw/sRGB, PNG oracle) | ComparisonPass split view | sRGB MSE 4.4e-5 † | max 22 bytes |
| upstream image test `SplitScreen.py` (jpg ×2 raw/sRGB, PNG oracle) | interactive split (headless no-mouse state) | sRGB MSE 4.4e-5 † | max 22 bytes |
| upstream image test `ModulateIllumination.py` (jpg+png, EXR oracle) | optional-input compositing (radiance × reflectance) | 1.1e-3 † | 10 px > 0.02 |
| upstream image test `SimplePostFX.py` (hdr, PNG oracle) | bloom pyramid + star + CA + barrel + grading, all params non-default | sRGB MSE 1.1e-3 ⚠ | edge-localized: border sampling emulated in-shader (no border mode in WebGPU); HW TMU sub-texel precision differs, compounds over the 8-level pyramid |
| upstream image test `FLIPPass.py` (jpg raw vs sRGB, PNG oracle) | full FLIP perceptual metric (CIELab/CSF, magma display map) | byte MSE 6.7e-5 | max 10 bytes |
| upstream image test `PathTracerDielectrics.py` over its own upstream scene (nested_dielectrics, 4 frames) | nested volumes (priorities, absorption), rotated env, stratified jitter, 20 bounces, 30x mesh instancing | sRGB MSE 4.9e-5 | 91 @0.05 |
| volume-absorption isolation oracles (single + nested-priority cubes, IoR 1) | interior list + evalTransmittance vs analytic exp(-sigma d) | 1.2e-4 / 1.1e-4 | analytic-exact centers |
| upstream image test `PathTracerMaterials.py` over its own upstream scene (material_test, 4 frames) | 99-material zoo: metallic/roughness/transmission/delta/thin/IoR/diffuseTransmission rows, emissive strips, heavy instancing | sRGB MSE 1.3e-4 | 377 @0.05 |
| upstream image test `VBufferRTInline.py` — depth / viewW (mvec loose: native frame-0 prev-camera) | inline ray queries (the web-default path) | 1.1e-4 / 6.1e-5 | 0 / 0 |
| StratifiedSamplePattern (camera jitter) vs gcc/libstdc++ reference | std::mt19937 + std::shuffle + generate_canonical\<float\> replicated | bit-exact | 0 (unit-pinned) |
| upstream `HalfRes.py` graph over cornell_box (web-side only ⚠) | IOSize Half plumbing + stratified jitter + 16-frame accumulation | runs, half-res, jitter advances | no native oracle: the oracle GPU's Vulkan driver lacks ROVs, so native Mogwai cannot construct GBufferRaster at all |
| upstream image test `PathTracer.py` over cornell_box (4 frames) — color / guideNormal / reflectionPosW / albedo / specularAlbedo / indirectAlbedo / ToneMapper.dst | guide outputs + ResolvePass | 2.5e-4 / 1.4e-5 / 3.9e-5 / byte-exact / 1.6e-9 / byte-exact / 4.7e-6 | 38 @0.05 (stochastic silhouette, cornell policy) / 0 / 0 / — |

† residual is entirely the jpg *input decode* (browser vs FreeImage IDCT/chroma
upsampling, ≤3 sRGB LSB): the png-fed pixels contribute zero error (Composite
and CrossFade have identical stats), and the hdr-fed GaussianBlur sits at 4.2e-5.

RNG parity is exact: TinyUniform (LCG+TEA) and xoshiro128** (SplitMix64 seeding
emulated as paired u32) produce bit-identical streams, so 1-spp renders match
native to float tolerance rather than statistically.

### 7.2 Upstream image-test graph pass-rate (tests/image_tests/renderpasses/graphs, 39 graphs)

Status as of M7+verify. "Verified" = the unmodified graph runs on web and its
output is diffed against native Mogwai running the same file.

| Status | Count | Graphs |
|---|---|---|
| ✅ verified vs native | 14 | MinimalPathTracer, ToneMapping, VBufferRT, CompositePass, CrossFadePass, GaussianBlur, ColorMapPass, SideBySide, SplitScreen, ModulateIllumination, SimplePostFX, FLIPPass, PathTracer*, PathTracerDielectrics |

\* PathTracer.py: rayCount/pathLength outputs allocate but await the PixelStats port; the other 8 marked outputs verified.
| 🟢 runnable now (passes exist; oracle pending) | 1 | VBufferRTInline (same pass; inline variant is our default) |
| 🟡 PathTracer siblings | 2 | PathTracerAdaptive (dynamic spp / sampleCount input unsupported), SDFEditorRenderGraphV2 (SDF grids) |
| 🟠 runnable on web; native oracle impossible on this machine | 1 | HalfRes (needs FBX importer for Arcade.pyscene; and the oracle GPU lacks ROV support, so native Mogwai cannot run GBufferRaster-based graphs at all) |
| 🟡 needs GBufferRaster extra channels / GBufferRT pass | 7 | GBufferRaster, GBufferRasterAlpha, GBufferRT, GBufferRTInline, GBufferRTTexGrads, MVecRT, MVecRaster — ⚠ raster-based ones share the native-ROV oracle blocker |
| 🟡 needs larger pass ports (M8 scope) | 7 | SVGF, TAA, VBufferRaster, VBufferRasterAlpha, BSDFViewer, WhittedRayTracer, SceneDebugger |
| 🟡 M8 flagship items | 4 | RTXDI, WARDiffPathTracer ×3 |
| ❌ impossible on web (CUDA/driver tech) | 2 | OptixDenoiser, DLSS |

\* also needs SDF grid geometry (M7 remainder).

## 8. Feature parity matrix

### 8.1 Platform / Core capabilities

| Feature | Status | Explanation / strategy |
|---|---|---|
| D3D12 / Vulkan backends | 🔶 | WebGPU is the backend (itself lowered to D3D12/Vulkan/Metal by the browser) |
| WebGL2 backend | 🔶 partial-by-design | raster-only subset possible (no compute in WebGL2); deferred, see §2 |
| Slang shading language, full library | ✅ | Slang WGSL backend, verified on this host (§4) |
| Runtime shader specialization (DefineList, type conformances) | ✅ | slang-wasm in-browser compilation + AOT cache |
| Shader reflection → ParameterBlock/ShaderVar | ✅ | slang reflection JSON / wasm API |
| Hardware RT (DXR pipelines, inline RayQuery, SBTs) | 🟡 | **No WebGPU ray tracing API exists.** Software LBVH + compute traversal + megakernel lowering (§5); semantics preserved, performance lower |
| Shader Execution Reordering (NVAPI) | ❌ | NVIDIA hardware/driver feature; no web analog. No-op shim (perf-only) |
| Wave/subgroup intrinsics | 🟡 | WebGPU `subgroups` feature where available; workgroup-shared fallback |
| 64-bit shader integers/atomics | 🟡 | not in WGSL; paired-u32 emulation shim (verified bit-identical: SplitMix64 seeding, xoshiro128** streams) |
| fp16 in shaders | 🟡 | Chromium does not expose `shader-f16` on this host (driver supports it): token-level f16→f32 demotion; f16 rounding only at pack boundaries. 16-bit ints demoted likewise (absent from WGSL entirely) |
| fp64 in shaders | ❌ | absent from WGSL entirely (native Falcor uses it in a few reduction/accumulation paths → those switch to compensated-f32 🟡) |
| Bindless resources / unbounded descriptor arrays | 🟡 | not in browser WebGPU; texture-array packing per format class (§6.2), documented limits |
| Indirect draw/dispatch | ✅ | WebGPU native (`drawIndirect`, `dispatchWorkgroupsIndirect`); ExecuteIndirect-style multi-draw 🟡 loop-emulated |
| UAV counters / append buffers | 🟡 | emulated with explicit atomic counter buffers |
| Timestamp queries / GpuTimer / Profiler | ✅ | `timestamp-query` feature |
| Occlusion queries | ✅ | WebGPU native |
| Async compute / multiple queues | ❌ | WebGPU exposes a single queue; Falcor's LowLevelContextData queue selection becomes a no-op (correctness unaffected) |
| CUDA interop (buffers, semaphores, PyTorch tensors) | ❌ | no CUDA in browsers, full stop. `CudaUtils`/`CudaInterop` throw `UnsupportedFeatureError` |
| NSight Aftermath | ❌ | driver crash-dump tech; browser substitute is WebGPU validation + device-lost logs |
| Multi-GPU / LUID adapter selection | ❌ | browser picks adapter; only `powerPreference` hint exposed |
| Exclusive fullscreen / vsync control / HDR swapchain | 🟡 | Fullscreen API + canvas `toneMapping` (HDR in Chrome); no vsync-off, no refresh-rate control |
| Memory-mapped files, raw file paths, process spawn, registry/env | ❌ | sandboxed platform; OPFS + File System Access + fetch replace file I/O (`AssetResolver`) |
| Hot reload | ✅ | Vite HMR (superior to native) |
| Multithreaded scene build (TaskManager) | 🔶 | Web Workers (+ SharedArrayBuffer w/ COOP/COEP headers) |

### 8.2 Render passes (all 31 upstream directories)

| Pass | Status | Notes |
|---|---|---|
| AccumulatePass | ✅ | double-precision mode 🟡 → compensated f32 (fp64 gap) |
| BlitPass | ✅ | |
| BSDFOptimizer | 🟡 | uses diff rendering; gradients ✅, no in-browser torch optimizer → TS optimizer (Adam) provided |
| BSDFViewer | ✅ | |
| DebugPasses (ColorMap/Comparison/SideBySide/SplitScreen/InvalidPixelDetection) | ✅ | |
| DLSSPass | ❌ | NVIDIA NGX driver + hardware black box; nearest substitutes: TAA-upscale ✅ or FSR2-WGSL port 🔶 (separate pass, not DLSS parity) |
| ErrorMeasurePass | ✅ | |
| FLIPPass | ✅ | pure compute |
| GBuffer (GBufferRaster / GBufferRT / VBufferRaster / VBufferRT / DepthPass) | ✅ raster / 🟡 RT | RT variants via SoftwareRT (§5) |
| ImageLoader | ✅ | EXR/DDS via WASM codecs |
| MinimalPathTracer | ✅ | SoftwareRT megakernel; oracle-verified (9.5e-7, §7.1) |
| ModulateIllumination | ✅ | |
| NRDPass | 🟡 | NRD shader source is public (HLSL) → genuine port attempted in M8 (ReBLUR/SIGMA subset); host SDK reimplemented in TS. SVGF ✅ available meanwhile (§11.4) |
| OptixDenoiser | ❌ | requires CUDA+OptiX. Same substitutes as NRD |
| OverlaySamplePass | ✅ | |
| PathTracer | ✅ core | SoftwareRT megakernel, oracle-verified w/ NEE+MIS, Uniform/Power emissive samplers + EnvMapSampler (§7.1). v1 limits: fixed spp=1, no guide/NRD outputs, LightBVH sampler pending, volumes/SDF pending |
| PixelInspectorPass | ✅ | |
| RenderPassTemplate | ✅ | |
| RTXDIPass | 🟡 | RTXDI SDK shaders are BSD-licensed & portable; visibility via SoftwareRT; scheduled after PathTracer |
| SceneDebugger | ✅ | |
| SDFEditor | ✅ | pure compute + UI |
| SimplePostFX | ✅ | |
| SVGFPass | ✅ | becomes the default denoiser (replacing NRD/Optix use-cases) |
| TAA | ✅ | |
| TestPasses | ✅/❌ | GPU-test passes ✅; PyTorch interop pass ❌ (CUDA) |
| ToneMapper | ✅ | |
| Utils (Composite/CrossFade/GaussianBlur) | ✅ | |
| WARDiffPathTracer | 🟡 | §6.9 |
| WhittedRayTracer | 🟡 | SoftwareRT, recursion → loop |

### 8.3 Ecosystem / tooling

| Component | Status | Notes |
|---|---|---|
| Mogwai app (graph loading, UI, capture) | ✅ | browser app; FrameCapture→download, VideoCapture→WebCodecs (FFmpeg ❌) |
| Python scripting / console | 🔶 | TS API (shape-identical) + graph-`.py` compatibility layer; full CPython via optional Pyodide plugin |
| PyTorch interop (`falcor.pytorch`) | ❌ | no CUDA/torch in browser; gradient buffers exposed to JS/ONNX-web instead |
| FalcorTest | ✅ | vitest + Playwright harness (§7) |
| RenderGraphEditor | ✅ | RenderGraphUI in-browser |
| ImageCompare | ✅ | reused natively on CI host for oracle diffing; TS port for in-browser use |
| Importers | see §6.2 | glTF/OBJ ✅, PBRT/Mitsuba ✅ (parser ports), Assimp 🔶 WASM, USD 🔶 usd-wasm (partial), Python importer 🔶 Pyodide |
| SceneCache | 🔶 | OPFS/IndexedDB instead of disk cache |
| NVTT texture compression (import path) | 🔶 | WASM BC encoders; decode is native WebGPU |

## 9. Known behavioral divergences (accepted, documented)

1. **Async boundaries.** Device creation, shader compilation, buffer readback and
   screenshot capture are `async` on the web (native Falcor blocks). APIs that are
   synchronous in Falcor and *cannot* be async-hidden return Promises; graph execution
   itself stays synchronous per-frame (encoders are synchronous).
2. **No raw pointers/interop handles.** `getNativeHandle()` returns the WebGPU object.
3. **Float determinism.** WGSL→driver compilation differs from DXIL/SPIR-V; image
   tests use tolerance thresholds, not bit-exactness (same policy as Falcor's own
   cross-vendor tests).
4. **Performance envelope.** Software RT and no SER/bindless put the ceiling below
   native; parity target is *feature/semantics*, not frame-time.

## 10. Roadmap (milestones map to upstream test coverage)

| M | Scope | Exit criterion |
|---|---|---|
| **M0** ✔ | env: Falcor clone + native oracle build, Slang WGSL toolchain, workspace scaffold, shader pipeline PoC | done on this host |
| **M1** ✔ | Core/API: Buffer/Texture/Sampler/Formats/Contexts/FBO/State/PSO caches, GpuMemoryHeap, GpuTimer, Fence | GPU unit tests green (hardware WebGPU under Xvfb) |
| **M2** ✔ | Program system: ProgramManager (slang-wasm), reflection, ParameterBlock/ShaderVar, ComputePass/RasterPass/FullScreenPass | ParameterBlock/program tests green |
| **M3** ✔ | Utils: Math lib, Algorithm passes (ParallelReduction/PrefixSum), sample generators | algorithm tests green vs CPU refs (BitonicSort deferred: warp-32 assumptions) |
| **M4** ✔ | RenderGraph core + **Pyodide graph-`.py` loader** (§11.1); ToneMapper, Blit, Accumulate, ImageLoader | unmodified upstream `ToneMapping.py` runs end-to-end in browser |
| **M5** ✔ | Scene host driving unmodified upstream Scene.slang, glTF import, Camera, Lights, MaterialSystem (Standard); GBufferRaster | GBuffer matches native GBufferRT oracle per-pixel |
| **M6** ✔ | SoftwareRT: CPU BVH, SceneRayQuery override; VBufferRT, MinimalPathTracer | MinimalPathTracer matches native hardware DXR at 9.5e-7 (§7.1) |
| **M7** ✔ core | Material zoo (Cloth/Hair/PBRT ×6), LightCollection, EnvMap+EnvMapSampler, emissive Uniform/Power samplers, **full PathTracer**, **`.pyscene` on web** (§11.1) | PathTracer matches native at 1.6e-4; 15 oracle comparisons green (§7.1). Open: LightBVH sampler, MERL/RGL, GridVolumes, SDF grids ×4, animation/skinning |
| **M8** | Mogwai UI (ImGui-wasm, RenderGraphUI, capture), RTXDI, NRD port, WARDiffPathTracer, Assimp/USD importers, WebGL2 raster subset (stretch) | upstream image-test graph suite pass-rate report; parity matrix finalized |

## 11. Resolved design questions (user decisions, 2026-07-05)

1. **Graph-script compat layer** — *Resolved: `.py`-first.* Upstream `.py` graph files
   **and** `.pyscene` scene files are the primary content path, executed via Pyodide
   with a `falcor` bridge module (mini-interpreter dropped — pyscenes are real Python).
   The shape-identical TS API comes after the `.py` path is verified. Pyodide moves
   from "optional M8 plugin" onto the critical path (graphs in M4, pyscene in M5).
2. **USD priority** — *Resolved: as 1:1 as possible.* USD import via usd-wasm is a real
   deliverable (M8), not an optional stub; partiality only where usd-wasm itself lags
   OpenUSD.
3. **WebGL2 subset** — *Resolved: 1:1 where possible; use WebGL2 to fill WebGPU gaps.*
   WebGPU remains primary. Audit note: WebGL2 capabilities are a strict subset of
   WebGPU for everything Falcor needs, with one exception worth tracking —
   `EXT_disjoint_timer_query` availability vs. `timestamp-query` on some platforms.
   Raster-only WebGL2 backend remains an M8 stretch goal.
4. **Denoiser policy** — *Resolved: as 1:1 as possible.* NRD's shader source is public
   (github.com/NVIDIA-RTX/NRD, HLSL): attempt a genuine NRD port (ReBLUR/SIGMA subset)
   in M8 instead of treating NRDPass as ❌-with-substitute. NRDPass reclassified 🟡
   (host-side SDK → TS reimplementation; shaders via Slang/WGSL). DLSS remains ❌
   (closed driver binary). OptixDenoiser remains ❌ (CUDA); OIDN-web offered as an
   optional non-parity pass.
5. **Fork deltas** — *Resolved: track upstream 8.0 only* (`eb540f67`); the research
   fork's custom passes are out of scope.
