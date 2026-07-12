# web-falcor — Feature parity matrix & divergences

Part of the [web-falcor design docs](README.md). Section numbers (§8–§9) are
kept stable across the split so the `§N` cross-references throughout the docs
stay valid. The ✅ / 🟡 / 🔶 / ⏳ / ❌ / 🟠 status markers are defined in
[architecture.md §1](architecture.md#1-goals-and-ground-rules).

Every row below was verified against the actual code (2026-07-09 audit of
`packages/` vs upstream `Falcor/Source/`, rows re-verified as features landed
through 2026-07-12), not against intent: ✅/🟡/🔶 mean the
thing exists and works today, ⏳ means portable-but-not-built, 🟠 means blocked
by a documented toolchain/asset gap, ❌ means the web platform cannot provide it.

## 8. Feature parity matrix

### 8.1 Platform / Core capabilities

| Feature | Status | Explanation / strategy |
|---|---|---|
| D3D12 / Vulkan backends | 🔶 | WebGPU is the backend (itself lowered to D3D12/Vulkan/Metal by the browser) |
| WebGL2 backend | 🔶 partial-by-design | raster-only subset possible (no compute in WebGL2); deferred, see §2 |
| Slang shading language, full library | ✅ | Slang WGSL backend; unmodified upstream `.slang` tree compiled in-browser (slang-wasm) with a small override set |
| Runtime shader specialization (DefineList) | ✅ | slang-wasm in-browser compilation + AOT cache |
| Type conformances (first-class API) | ⏳ | web relies on `WEBFALCOR_MTL_*` static specialization + overrides instead; no `TypeConformance` API surface |
| Shader reflection → ParameterBlock/ShaderVar | ✅ | slang reflection JSON / wasm API; ShaderVar folded into `ParameterBlock.makeRootVar()` |
| Hardware RT (DXR pipelines, inline RayQuery, SBTs) | 🟡 | **No WebGPU ray tracing API exists.** Software CPU-built BVH + compute traversal + megakernel lowering (§5); semantics preserved, performance lower. Hardware TLAS/BLAS, compaction, refit, `RtStateObject`, `ShaderTable` themselves ❌ |
| Shader Execution Reordering (NVAPI) | ❌ | NVIDIA hardware/driver feature; no web analog. No-op shim (perf-only) |
| Wave/subgroup intrinsics | 🟡 | WebGPU `subgroups` feature where available; workgroup-shared fallback |
| 64-bit shader integers/atomics | 🟡 | not in WGSL; paired-u32 emulation shim (verified bit-identical: SplitMix64 seeding, xoshiro128** streams) |
| fp16 in shaders | 🟡 | Chromium does not expose `shader-f16` on this host (driver supports it): token-level f16→f32 demotion; f16 rounding only at pack boundaries. 16-bit ints demoted likewise (absent from WGSL entirely) |
| fp64 in shaders | ❌ | absent from WGSL entirely; native's few uses switch to compensated-f32 🟡 (e.g. the watertight-triangle tiebreak in `IntersectionHelpers.slang` → Kahan difference-of-products via `fma`). Caution: fp64 reaching slangc's WGSL backend aborts emission *silently* — the compiler now fails loudly on empty entry points |
| Bindless resources / unbounded descriptor arrays | 🟡 | not in browser WebGPU; texture-array packing per format class (§6.2), documented limits |
| Min/max-reduction texture samplers | ❌ | D3D12 `TextureReductionMode` has no WebGPU equivalent. Sole upstream consumer (displacement shell tightening) is overridden to conservative global bounds — identical intersections (§9.7); host-built min/max mip pyramids are the substitute if tight bounds are ever needed |
| Indirect dispatch | ✅ | `dispatchWorkgroupsIndirect` wired (`ComputeContext.dispatchRawIndirect`) |
| Indirect draw / ExecuteIndirect | ⏳ | WebGPU has `drawIndirect`, not wired yet; ExecuteIndirect-style multi-draw would be loop-emulated 🟡 |
| UAV counters / append buffers | 🟡 | emulated with explicit atomic counter buffers (packed-region pattern, see PixelStats) |
| GpuTimer / timestamp queries | ✅ | standard `timestampWrites` on every compute/render pass via `Core/API/Profiler.ts` (RenderGraph labels each pass); legacy `Core/API/GpuTimer.ts` (`writeTimestamp`) kept for ad-hoc use |
| Profiler framework (FALCOR_PROFILE, Clock/FrameRate/TimeReport) | 🟡 partial | per-pass GPU ms via `Profiler.getStats()` (timestampWrites + async resolve, ~1 frame late), shown in the Mogwai status line; no CPU events/timing tree/TimeReport |
| Occlusion queries | ⏳ | WebGPU supports occlusion query sets; no QueryHeap/host API built yet. Pipeline-statistics queries ❌ (not in WebGPU) |
| Async compute / multiple queues | ❌ | WebGPU exposes a single queue; Falcor's LowLevelContextData queue selection becomes a no-op (correctness unaffected) |
| CUDA interop (buffers, semaphores, PyTorch tensors) | ❌ | no CUDA in browsers, full stop. `CudaUtils`/`CudaInterop` throw `UnsupportedFeatureError` |
| NSight Aftermath | ❌ | driver crash-dump tech; browser substitute is WebGPU validation + device-lost logs |
| Multi-GPU / LUID adapter selection | ❌ | browser picks adapter; only `powerPreference` hint exposed |
| Exclusive fullscreen / vsync control / HDR swapchain | 🟡 | Fullscreen API + canvas `toneMapping` (HDR in Chrome); no vsync-off, no refresh-rate control |
| Memory-mapped files, raw file paths, process spawn, registry/env | ❌ | sandboxed platform; OPFS + File System Access + fetch replace file I/O |
| Shader hot reload (in-app `reloadShaders`) | ⏳ | Vite HMR 🔶 reloads the dev app; Falcor's F5-style in-session shader reload not built |
| Multithreaded scene build (TaskManager) | ⏳ | scene build is single-threaded today; Web Workers (+ SharedArrayBuffer) possible |
| Plugin system (dynamic pass/importer loading) | 🔶 static | native .dll/.so loading ❌; passes register via a static factory + side-effect import. A dynamic JS plugin registry ⏳ |
| Settings system (global `Settings`, attribute filters) | ⏳ | per-pass `Properties` dicts only |

### 8.2 Render passes (29 upstream directories, 38 registered pass classes)

Tallies today: 20 pass classes fully implemented, 4 partial, 14 not implemented
(of which 4 ❌ NVIDIA-SDK-bound, 2 🟠 autodiff-blocked).

| Pass | Status | Notes |
|---|---|---|
| AccumulatePass | ✅ | `Double` mode maps to SingleCompensated (fp64 gap 🟡) |
| BlitPass | ✅ | |
| BSDFOptimizer | 🟠 | no host port; depends on the same slangc autodiff blocker as WARDiffPathTracer (§6.9), plus an optimizer loop |
| BSDFViewer | ✅ | verified vs native (1.9e-4) |
| DebugPasses: ColorMapPass / SideBySidePass / SplitScreenPass | ✅ | verified; TextRenderer overlay labels + interactive divider ⏳ |
| DebugPasses: InvalidPixelDetectionPass | ✅ | unmodified upstream shader; functional GPU test (injected NaN→red, Inf→green, valid→black) |
| DLSSPass | ❌ | NVIDIA NGX driver + hardware black box; nearest substitutes: TAA-upscale ✅ or FSR2-WGSL port 🔶 (separate pass, not DLSS parity) |
| ErrorMeasurePass | ✅ core | difference kernel (WTexture2D override — rgba32float storage is write-only) + GPU-reduced mean error; reference from input or file (EXR/HDR/browser formats). Measurements surface on `measurements` via async readback — csv file output ❌ (no file IO), running-error smoothing ⏳ |
| FLIPPass | ✅ core | LDR path verified (byte MSE 6.7e-5); HDR auto-exposure path + pooled UI values ⏳ |
| GBufferRaster | ✅ | native oracle impossible on this host (ROV), RT-cross-verified |
| GBufferRT | 🟡 | SoftwareRT; verified incl. texGrads (byte-exact) |
| VBufferRT | 🟡 | SoftwareRT; verified |
| VBufferRaster | ⏳ | rasterized V-buffer; portable, not built |
| ImageLoader | ✅ | browser-decodable formats + `.hdr` + `.dds`/BC + `.exr` (parse-exr; GPU-verified exact vs CPU decode) |
| MinimalPathTracer | ✅ | SoftwareRT megakernel; oracle-verified (9.5e-7, §7.1); shades triangle, curve, displaced-triangle and SDF hits |
| ModulateIllumination | ✅ | lives under `Utils/` in the web tree |
| NRDPass | 🟠 SDK absent | NRD SDK not bundled in this Falcor drop (no `external/packman/nrd/`) → denoiser shaders uncompilable here. Host portable; NRD's HLSL source is public, genuine port stays the plan (§11.4). SVGF ✅ meanwhile |
| OptixDenoiser | ❌ | requires CUDA+OptiX. Same substitutes as NRD |
| OverlaySamplePass | ⏳ | demo/example pass; portable, not built |
| PathTracer | ✅ verified | full upstream loop: NEE+MIS, Uniform/Power/LightBVH emissive samplers, EnvMapSampler, dielectrics/nested priority, guide outputs, adaptive spp (`sampleCount` input), rayCount/pathLength stats. Fixed spp 1–16 + variable spp verified (spp=4 vs native: 10/65536 bad px); curve geometry (`USE_CURVES` + Hair BSDF) verified vs native. Remaining ⏳: `USE_RTXDI` in-tracer integration, NRD guide outputs; SER ❌ |
| PixelInspectorPass | ⏳ | cursor pixel/material inspector; portable, not built |
| RenderPassTemplate | ⏳ | boilerplate template, not a feature |
| RTXDIPass | ✅ verified vs native | Full port (PrepareSurfaceData + ReSTIR spatiotemporal resampling + FinalShading). Upstream RTXDI.py replica over Arcade at frames 1/16/64: bias <6e-4, ≤5/3600 bad 8x8 blocks. Overrides: texel buffers → structured, boiling filter compiled out (WaveActiveCountBits; native default off), bool cbuffer members → uint, outputs moved into the FinalShading block (4-bind-group cap), lightInfo+compactLightInfo merged (16-storage-buffer cap) |
| SceneDebugger | ✅ | verified (1.0e-5) |
| SDFEditor | ⏳ + 🟠 oracle | not built: needs `.sdf`/`.sdfg` IO + runtime SDF editing/bake (§8.4) + interactive UI. Upstream flagship scene asset absent from the media drop, so native oracle impossible anyway |
| SimplePostFX | ✅ | verified |
| SVGFPass | ✅ ported | feature-verified vs native (sphere_array graph, mean 4.2e-4); all 5 kernels verbatim. Default denoiser (replacing NRD/Optix use-cases) |
| TAA | ✅ ported | feature-verified vs native (jittered GBufferRT graph, mse 8.4e-7) |
| TestPasses: TestRtProgram | ⏳ | exercises RT shader-table/hit-group plumbing; mostly moot on software RT but portable as a megakernel |
| TestPasses: TestPyTorchPass | ❌ | CUDA + PyTorch tensor interop |
| ToneMapper | ✅ | all 6 operators + manual exposure verified; auto-exposure (log-luminance mip chain) verified vs native (sRGB MSE 2.1e-4, zero mean bias); fNumber/shutter/filmSpeed physical exposure verified (upstream test variants, sub-byte bias) |
| Utils (Composite/CrossFade/GaussianBlur) | ✅ | verified |
| WARDiffPathTracer | 🟠 compiler-blocked | §6.9: autodiff primitive device-verified on WebGPU; slangc v2026.12.2 segfaults differentiating the full tracePaths (both wgsl and hlsl targets) → needs a Slang release with the large-function autodiff fix |
| WhittedRayTracer | 🟡 | SoftwareRT, recursion → loop; verified byte-exact |

### 8.3 Ecosystem / tooling

| Component | Status | Notes |
|---|---|---|
| Mogwai app (functional viewer) | ✅ core | loads graph `.py` + `.pyscene`/`.pbrt`, per-frame execute, presents marked output (swapchain blit), play/pause + graph/output pickers, first-person camera, per-pass DOM `renderUI` panel, URL params. FrameCapture ✅ (Capture button: float outputs download as EXR, 8-bit as PNG). Missing ⏳: VideoCapture (WebCodecs) / TimingCapture extensions, python console, scene/material/light UI, profiler overlay |
| Python scripting / console | 🔶 | Pyodide runs **unmodified** upstream `.py` graphs and `.pyscene` files via a curated `falcor` bridge (factories + SceneBuilderBridge). No auto-generated ScriptBindings, no ScriptWriter, no interactive console ⏳ |
| PyTorch interop (`falcor.pytorch`) | ❌ | no CUDA/torch in browser; ONNX-web-style substitute would be non-parity ⏳ |
| FalcorTest | 🔶 | vitest (unit) + Playwright GPU harness w/ native-oracle image compares (§7); no slang-driven `GPU_TEST` framework ⏳ |
| RenderGraphEditor (ImGui node UI) | ⏳ stretch | functional viewer done; node editor is dev tooling orthogonal to rendering parity |
| RenderGraph `.py` export / RenderGraphIR | ✅ | `exportScript()` emits the camelCase image-test dialect (round-trip fixpoint verified); `removeEdge`/`unmarkOutput` added; divergence (§9): no snake_case IR, markOutput channel masks untracked |
| ImageCompare | 🔶 | native tool used on CI host for oracle diffing; its MSE/FLIP gate policy reimplemented inline in the GPU suites + FLIPPass. No standalone in-browser tool ⏳ |
| Importers | see §8.4 | glTF ✅ (TS), FBX 🔶 (assimpjs, `.fbx` full scenes only), PBRT ✅ subset, `.pyscene` ✅, USD 🔶 subset (tinyusdz-wasm: meshes/xforms/UsdPreviewSurface incl. baseColor/ORM/normal/emissive textures, verified vs native — lights/cameras/skel/subdiv ⏳), Mitsuba ⏳ (no Mitsuba content in the media drop → no oracle) |
| SceneCache | ⏳ | binary scene cache not built (OPFS/IndexedDB route available) |
| Image IO (Bitmap/EXR read+write, image save) | 🟡 | `.hdr`/DDS-BC/`.exr` decode (EXR via parse-exr, wired into ImageLoader + EnvMap + ErrorMeasure); EXR write (uncompressed float scanlines, bit-exact round-trip) feeds the viewer capture; unified Bitmap class ⏳ |
| NVTT texture compression | ❌ native / ⏳ substitute | decode side covered (DDS/BC parse + `texture-compression-bc` upload + CPU BC1/3/5 decode); a WASM BC *encoder* would be a substitute, not NVTT parity |

### 8.4 Scene, materials, lights, animation (audit 2026-07-09)

| Feature | Status | Notes |
|---|---|---|
| TriangleMesh geometry | ✅ | full path incl. instancing, alpha test (Mask; Blend mode ⏳) |
| DisplacedTriangleMesh / displacement mapping | ✅ core | full ray-marched displacement: displaced meshes get conservative per-tri AABBs in the merged BVH; upstream DisplacedTriangleMeshIntersector compiles via two overrides (max-shell-thickness — WebGPU lacks min/max-reduction samplers; Kahan diff-of-products replaces the fp64 watertight tiebreak). cornell_box_displaced FaceNormals match native (mean 5.1e-3; brick-edge march micro-divergence) and 256spp shaded radiance at bias 6.6e-4. v1: one displacement texture per scene |
| Curve geometry (LSS) / hair | ✅ core | USD BasisCurves import (USDA text) → CurveTessellation port → linear-swept-sphere segments traversed via a segment-AABB BVH in the merged BVH buffer (zero extra bindings; verified footprint-invariant) → upstream CurveIntersector/Han19. two_curves.pyscene FaceNormals match native pixel-exactly (0/65536); shaded curves (native default HairMaterial, Chiang16) match native through MPT and the full PathTracer at 64spp (bias ≤1.1e-3). `.hair` files (no assets in the drop), CurveOTS/ribbon modes ⏳ |
| SDF grids (NDSDF/SBS/SVS/SVO) | ✅ | all 4 representations GPU-verified. Content path limited to the procedural generator: `.sdf`/`.sdfg` file IO + runtime editing/`bake()` ⏳ (prereq for SDFEditor) |
| Custom primitives (procedural AABBs) | 🟡 approximated | rendered as box meshes; app-supplied intersection shaders have no software-RT equivalent (true parity ❌, fixed-function intersectors ⏳) |
| StandardMaterial | ✅ | verified across the oracle suite |
| PBRTConductor | ✅ | override shader + bridge |
| Cloth / Hair / PBRTDiffuse materials | 🟡 unverified | factory dispatch + bridge exist; upstream shaders un-vetted for WGSL, no oracle yet |
| PBRTDielectric / CoatedConductor / CoatedDiffuse / DiffuseTransmission | ⏳ | factory cases exist but no host constructor reaches them (PBRT importer maps to Standard) |
| MERL / MERLMix / RGL measured materials | ⏳ | not instantiable: no factory case, no `.brdf`/`.bsdf` loaders, no data packers, no RGLAcquisition |
| Texture LOD (ray cones / ray diffs) | 🟡 partial | explicit-gradient path verified (GBufferRT texGrads byte-exact); ray-cone mode not wired through the megakernels everywhere |
| Analytic lights (Point/Directional/Distant/Rect/Disc/Sphere) | ✅ | verified incl. area-light sampling |
| Emissive geometry (LightCollection) | ✅ | incl. textured-emissive flux integration; LightBVH sampler ✅ (GPU refit ⏳ — rebuild-only; options not plumbed ⏳) |
| EnvMap | ✅ | rotation/intensity/tint; loads Radiance `.hdr` and OpenEXR `.exr` (decodeExr) |
| LightProfile (IES) | ⏳ | dummy binding only; no IES loader/bake |
| Camera (pinhole, jitter, motion vectors) | ✅ | verified (incl. prev-matrix roll) |
| Camera DoF / physical camera | ✅ core | `apertureRadius`/`focalDistance` + thin-lens sampling verified vs native (VBufferRT depth/viewW 0 bad px; PathTracer consumes `viewW`); shutter/ISO exposure params ⏳ |
| Camera controllers | 🔶 partial | FirstPerson ✅ (app layer, mouse+WASD); Orbiter / SixDoF ⏳; gamepad ⏳ |
| Node / skinned / morph animation | ✅ | CPU skinning + morph (upstream does GPU skinning 🟡); camera/light animation ✅; LINEAR/STEP/CUBICSPLINE |
| Animated vertex caches (Alembic) | ⏳ | no `.abc`/AnimatedVertexCache support |
| Per-clip loop behaviors / global time control | 🟡 partial | time loops `fmod(t, length)` and clamps before the first key — matches native defaults (AnimationController + Constant pre-behavior); per-clip pre/post-infinity behaviors (Linear/Cycle/Oscillate extrapolation, e.g. animated_cubes) and a global time-control API ⏳ |
| Motion vectors for animated geometry | ✅ | rigid (prev world matrices) native-exact (mean 1.2e-7); skinned/morphed (prev-position double buffer + IsDynamic) verified by reprojection — native itself writes zero skinned mvecs on this content (probed) |
| Animated-scene BVH | 🟡 | full CPU rebuild per frame (correct, no refit path) |
| GridVolumes (NanoVDB) | ✅ | `.vdb` parsed in-browser → NanoVDB, verified vs native; uncompressed codecs only ⏳ (zip/blosc), `.vdb` frame sequences ⏳, blackbody emission conversion ⏳ |
| Runtime material/light property edits | ✅ | `Scene.getLight/getMaterial` + `updateLights/updateMaterial` re-pack GPU data post-build; verified vs native applying identical python edits (mean 7.1e-4). Emissive edits rebuild the NEE flux tables (dimmed-cornell PT vs native: bias 2.5e-4, 0/1024 blocks). Emissive-presence toggles that flip scene defines still need pass recreation |
| Importer: glTF | ✅ | TS importer: meshes, skinning, morph targets, animations, cameras, lights; KTX2/Draco ⏳ |
| Importer: Assimp | 🔶 partial | assimpjs: full scenes `.fbx` only (other formats mesh-only via `TriangleMesh.createFromFile`); >2 GB-heap FBX aborts (wasm32) — BistroExterior; DDS ✅ / TGA ⏳ textures |
| Importer: PBRT (pbrt-v4) | ✅ subset | camera/lights/shapes/area lights verified; all materials → Standard (`usePBRTMaterials=true` path ⏳), textures/spectra/media/curves ⏳ |
| Importer: `.pyscene` | ✅ | unmodified upstream scenes via Pyodide bridge |
| Importer: USD | 🔶 subset | tinyusdz-wasm (1.9MB, reads usda/usdc/usdz): meshes + xform hierarchy + UsdPreviewSurface → Standard incl. UsdUVTexture baseColor (sRGB, V-flip) and roughness/metallic packed ORM like native's CreateSpecularTexture — verified vs native (mask/viewW exact; textured 64spp radiance bias ~1e-5, per-region ≤0.3%). Normal/emissive texture slots plumbed (same path, not separately oracled). ⏳: lights/cameras (not exposed by tinyusdz RenderScene), texture channel selectors (r assumed), UsdSkel, subdivision refinement, instancing |
| Importer: Mitsuba | ⏳ | not started |
| SceneBuilder flags (optimize/merge/dedup) | ⏳ | tangent generation ✅; optimization flags not built |

### 8.5 Framework, debugging & test infrastructure

| Feature | Status | Notes |
|---|---|---|
| RenderGraph core (compile, alloc, I/O merge, `.py` load) | ✅ | verified across all feature graphs |
| RenderPassReflection completeness | ⏳ gaps | `Persistent` flag, resolve-size callbacks, non-texture2D field builders missing |
| PixelDebug host (shader `print()` readback/console) | ⏳ | shaders compile with `printSetPixel`, but no host class binds/reads `gPixelDebug` — shader print output never surfaces |
| PixelStats | ✅ | packed-atomic-buffer collection (verified vs native rayCount/pathLength dumps) + `Rendering/Utils/PixelStats.ts` `getStats()` aggregate (GPU region sums, async readback ~1 frame late, docs §9); PathTracer exposes `getPixelStats()` |
| WarpProfiler | ⏳ | subgroup-dependent; portable where `subgroups` exists |
| BSDFIntegrator (white furnace) | ⏳ | BSDFViewer pass exists; integrator harness not built |
| Algorithm library | ✅ | ParallelReduction ✅, PrefixSum ✅, BitonicSort ✅ (portable shared-memory override of the NVAPI warp-shuffle kernel, §9; exact vs CPU chunk sort incl. 2D dispatch + tail padding) |
| Utils/Math | ✅ core | Vector/Matrix/Quaternion ✅; CubicSpline host / SphericalHarmonics ⏳ |
| Gui (Dear ImGui) | 🔶 | DOM `UIWidgets` (text/button/checkbox/slider/dropdown/group); `renderUI` implemented on 2 passes so far ⏳; TextRenderer/Font/PixelZoom ⏳ |
| Video (FFmpeg encode/decode) | ❌ native / ⏳ substitute | WebCodecs route not built |
| AssetResolver | 🔶 ad-hoc | URL resolution inline in the app; no search-path API |

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
5. **Animated scenes rebuild the CPU BVH each frame** (no refit); results are
   correct but scale linearly with geometry, unlike native BLAS refit.
6. **BC-textured scenes currently sample CPU-decoded RGBA8 (≤512px per texture)**
   instead of native full-res BC arrays — a quality/memory divergence until the
   per-format BC texture-array material path lands (⏳, §8.4).
7. **Displacement marching uses the conservative global shell** (max thickness)
   instead of native's per-triangle min/max-sampler tightening — WebGPU has no
   reduction samplers (§8.1). The shell only bounds the search range, so
   intersections are identical; marching is somewhat slower. v1 also binds a
   single displacement texture per scene.
8. **The fp64 watertight-triangle tiebreak is replaced by a Kahan
   difference-of-products (`fma`)** — extended precision rather than true fp64;
   affects only exactly-edge-grazing rays.
9. **Custom primitives render as visible box meshes** (the 🟡 approximation in
   §8.4); native draws nothing for them unless an app supplies an intersection
   shader. Scene compares strip `addCustomPrimitive` calls.
10. **Animation before the first keyframe clamps to it** (Constant behavior,
   matching native defaults); scenes assigning per-clip pre/post-infinity
   behaviors extrapolate natively but stay clamped on the web (§8.4).
