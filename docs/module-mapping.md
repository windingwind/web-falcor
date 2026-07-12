# web-falcor — Module-by-module mapping

Part of the [web-falcor design docs](README.md). Section numbers (§6) are
kept stable across the split so the `§N` cross-references throughout the docs
stay valid.

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

This paragraph is the port *strategy*; see §8.4 for per-feature implementation
status. Landed since the 2026-07-09 audit: `CurveTessellation` + curve rendering,
displacement mapping, camera DoF, runtime material/light edits (incl. emissive
flux rebuild). Still unimplemented ⏳: `SceneCache`, `LightProfile` (IES),
MERL/MERLMix/RGL materials (no `.brdf`/`.bsdf` assets in the drop), vertex
caches (Alembic).

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

Importers (plugin package, like upstream; status per §8.4):
- **glTF**: native TS loader ✅ (meshes, skinning, morph targets, animations,
  cameras, lights; KTX2/Draco ⏳).
- **Assimp (FBX, DAE, OBJ, PLY, …)**: `assimpjs` (official Emscripten build) 🔶 —
  full-scene import is `.fbx`-only today; other formats load mesh-only via
  `TriangleMesh.createFromFile` ⏳.
- **PBRT**: TS port of Falcor's parser ✅ subset (materials map to Standard;
  textures/spectra/media/curves ⏳). **Mitsuba**: ⏳ not started.
- **USD**: 🔶 subset via `tinyusdz`-wasm (1.9 MB; reads usda/usdc/usdz): meshes,
  xform hierarchy, BasisCurves (USDA text), UsdPreviewSurface incl. textures —
  verified vs the native USDImporter (§8.4). Lights/cameras/UsdSkel/subdiv wait
  on tinyusdz API coverage.
- **PythonImporter**: 🔶 via Pyodide (see §6.7) — runs unmodified `.pyscene`.

### 6.3 `Rendering/`

| Subsystem | Status | Notes |
|---|---|---|
| `Lights/` Emissive samplers (Uniform/Power/LightBVH), `LightBVH(+Builder)`, `EnvMapSampler` | ✅ | Uniform/Power/LightBVH + EnvMapSampler ported and exercised by PathTracer; LightBVH GPU refit ⏳ (rebuild-only) and builder options not plumbed ⏳ |
| `Materials/` BSDF modules (Lambert, OrenNayar, Disney/Frostbite diffuse, GGX iso/aniso, StandardBSDF, Sheen, Hair Chiang16, Cloth, MERL/RGL, PBRT set, LayeredBSDF, Fresnel/Microfacet/NDF, TexLOD) | ✅ exercised subset | Standard + PBRTConductor verified end-to-end; Cloth/Hair/PBRTDiffuse plumbed but WGSL-unvetted 🟡; MERL/RGL and the other PBRT types have no host instantiation path ⏳ (§8.4) |
| `Volumes/` grid volumes (NanoVDB), GridVolumeSampler, phase functions | ✅ GPU-verified | Full chain verified vs native: browser parses the UNMODIFIED smoke.pyscene/.vdb -> byte-identical NanoVDB buffer -> gScene.grid0 + gridVolumes GPU plumbing -> SceneDebugger's 500-step ray-marched transmittance matches native at mean 3.7e-5 / 0 bad px (feature-smoke-debugger). PNanoVDB point lookups GPU-exact (0/500). NOTE: no upstream render pass consumes GridVolumeSampler in light transport in this drop (PathTracer handles homogeneous media only) — SceneDebugger is the upstream GPU consumer. ⚠ native openvdb broken on this machine -> native oracles use .nvdb (byte-identical, tools/vdb/) |
| `RTXDI/` | ✅ GPU-verified | Full ReSTIR DI port: RTXDI SDK 1.3.0 resampling shaders (WGSL via 6 overrides), host orchestration (rtxdi::Context math, LightUpdater/EnvLightUpdater, presampling, spatiotemporal resampling), visibility rays through SoftwareRT. Upstream RTXDI.py over Arcade matches native at bias <6e-4, ≤5/3600 bad 8x8 blocks (frames 1/16/64). Web divergences: localLightPdf R32Float (no r16float storage), boiling filter compiled out (WaveActiveCountBits unmapped; native default off), lightInfo+compactLightInfo share one buffer (16-storage-buffer budget) |
| `SDFs/` NDSDFGrid (+ base/voxel utils) | ✅ GPU-verified | Full ND chain: pyscene bridge (createNDGrid/generateCheeseValues, bit-exact host build incl. gcc mt19937 + RIGHT-TO-LEFT float3 ctor draws), R8Snorm Z-stacked LOD atlas (WGSL has no binding arrays; LOD widths 1+(c<<lod) are no mip chain), sphere tracing through SoftwareRT (upstream SDFGridIntersector). Web render matches a faithful CPU port of intersectSDF PIXEL-EXACTLY (0/230400; oracle mask committed). ⚠ native NDSDF is broken on this machine (SPIR-V offset texel fetches): gradients NaN, footprint dilated +24568 px, mesh-less TLAS instance IDs off by one — background still verified vs native. SparseBrickSet (createSBS): CPU brick build (validity→prefix-sum→bricks+AABBs) + brick-AABB-loop tracing; SDFSBS.pyscene body matches the shared surface footprint at 3/230400 px (native can't oracle SBS — Mogwai core-dumps the brick BLAS build). SVS/SVO compile to WGSL and the hosts are portable, but their runtime puts one AABB per SURFACE VOXEL (tens of thousands for the cheese) — impractical on the linear brick-loop; a procedural-AABB BVH over the SDF primitives would be needed (the triangle BVH is a natural base). NDSDF (dense) + SBS (sparse-brick) already cover both SDF representation classes |
| `Utils/PixelStats` | ✅ ported | per-pixel counters ported (binding array + texture atomics -> one packed `Atomic<uint>` buffer, 5 regions; rayCount/pathLength verified vs native per-pixel). Aggregate CPU-readback stats (`getStats()`) pending |

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
lower to plain compute code, so it compiles to WGSL. **The primitive is VERIFIED
on-device** (M8 feasibility gate): the `autodiff-feasibility` GPU test runs a
`bwd_diff`/`fwd_diff` kernel through the real WebGPU device and both produce the exact
analytic gradient (f(x)=x²k+sin(x) → 2xk+cos(x) = 11.58385 at x=2,k=3).

**BUT the full WARDiffPathTracer is blocked by a compiler crash** (🟠, tooling, not a
web limitation). Porting the pass past the mechanical steps —

1. RT-pipeline raygen → compute megakernel over `SceneRayQuery` (`import
   Scene.Raytracing` is vestigial; the diff-scene queries already use
   `SceneRayQuery`/`RaytracingInline`);
2. `TriangleHit` brace-init → member-wise (the HitInfo override adds an
   `__init(PackedHitInfo)` that shadows aggregate init);
3. explicit `detach()` at each implicit derivative-drop into a non-differentiable
   path record (newer Slang enforces `E41031` where 2024.1.34 did not) —

reaches a point where **`slangc` v2026.12.2 SEGFAULTS while differentiating the full
`tracePaths` path tracer**, for BOTH forward and reverse mode and BOTH the `wgsl` and
`hlsl` targets (so it is an autodiff-codegen crash, not a WGSL issue). Native Falcor
builds this pass with Slang **2024.1.34** (no WGSL backend, but does not crash on the
autodiff). The web port is therefore gated on a Slang version that has both the WGSL
backend **and** the fix for this large-function autodiff crash. When unblocked, the
mechanical override recipe above applies; gradient accumulation needs the float-atomic
CAS shim, and the PyTorch training loop is replaced by gradient buffers readable into
JS / ONNX-web pipelines.
