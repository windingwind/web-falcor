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
| fp16 in shaders | 🟡 | Chromium does not expose `shader-f16` on this host (driver supports it): token-level f16→f32 demotion; f16 rounding only at pack boundaries. Host-side packs (materials, vertices, emissive UVs, alias tables) use `Utils/Math/Float16.ts`, a bit-faithful port of native `Float16.cpp` (round to nearest, ties up) — an earlier truncating pack biased material params by up to one fp16 ULP (found by the BSDFIntegrator oracle). 16-bit ints demoted likewise (absent from WGSL entirely) |
| fp64 in shaders | ❌ | absent from WGSL entirely; native's few uses switch to compensated-f32 🟡 (e.g. the watertight-triangle tiebreak in `IntersectionHelpers.slang` → Kahan difference-of-products via `fma`). Caution: fp64 reaching slangc's WGSL backend aborts emission *silently* — the compiler now fails loudly on empty entry points |
| Bindless resources / unbounded descriptor arrays | 🟡 | not in browser WebGPU; texture-array packing per format class (§6.2), documented limits |
| Min/max-reduction texture samplers | ❌ | D3D12 `TextureReductionMode` has no WebGPU equivalent. Sole upstream consumer (displacement shell tightening) is overridden to conservative global bounds — identical intersections (§9.7); host-built min/max mip pyramids are the substitute if tight bounds are ever needed |
| Indirect dispatch | ✅ | `dispatchWorkgroupsIndirect` wired (`ComputeContext.dispatchRawIndirect`) |
| Indirect draw / ExecuteIndirect | ✅ | `RasterPass.drawIndirect`/`drawIndexedIndirect` (GPU-driven args verified incl. zero-count command); §9: no GPU count buffer in WebGPU — multi-command loops over the arg stride |
| UAV counters / append buffers | 🟡 | emulated with explicit atomic counter buffers (packed-region pattern, see PixelStats) |
| GpuTimer / timestamp queries | ✅ | standard `timestampWrites` on every compute/render pass via `Core/API/Profiler.ts` (RenderGraph labels each pass); legacy `Core/API/GpuTimer.ts` (`writeTimestamp`) kept for ad-hoc use |
| Profiler framework (FALCOR_PROFILE, Clock/FrameRate/TimeReport) | ✅ core | `Core/API/Profiler.ts` ports the native event model: nested `startEvent`/`endEvent` names (`/RenderGraphExe::execute()/<pass>`), CPU time per event, GPU time from pass `timestampWrites` attributed to every active event (parents include children like native GpuTimer spans; verified parent == Σ children), EMA averages (σ 0.98), 512-frame history stats, pause, multi-frame capture → native JSON lanes; `ProfilerUI` DOM panel (Event / CPU / CPU % / GPU / GPU % table, bars, stacked history graph, Start/End Capture download) toggled with P in Mogwai; `m.profiler` python binding (`enabled`/`paused`/`events`/`start_capture`/`end_capture`); Clock ✅ (`m.clock`), FrameRate ✅, TimeReport ✅. §9: GPU readback lands a frame or two late and frames beyond the 4-deep readback ring are dropped rather than fenced; enabled by default; no `with profiler.event()` context manager (no PIX markers) |
| Occlusion queries | ✅ | `Core/API/QueryHeap.ts` (occlusion + timestamp types) + `RasterPass.setOcclusionQuery`; verified: depth-rejected draw reads 0 samples, visible draw counts all. Pipeline-statistics queries ❌ (not in WebGPU) |
| Async compute / multiple queues | ❌ | WebGPU exposes a single queue; Falcor's LowLevelContextData queue selection becomes a no-op (correctness unaffected) |
| CUDA interop (buffers, semaphores, PyTorch tensors) | ❌ | no CUDA in browsers, full stop. `CudaUtils`/`CudaInterop` throw `UnsupportedFeatureError` |
| NSight Aftermath | ❌ | driver crash-dump tech; browser substitute is WebGPU validation + device-lost logs |
| Multi-GPU / LUID adapter selection | ❌ | browser picks adapter; only `powerPreference` hint exposed |
| Exclusive fullscreen / vsync control / HDR swapchain | 🟡 | Fullscreen API + canvas `toneMapping` (HDR in Chrome); no vsync-off, no refresh-rate control |
| Memory-mapped files, raw file paths, process spawn, registry/env | ❌ | sandboxed platform; OPFS + File System Access + fetch replace file I/O |
| Shader hot reload (in-app `reloadShaders`) | ⏳ | Vite HMR 🔶 reloads the dev app; Falcor's F5-style in-session shader reload not built |
| Multithreaded scene build (TaskManager) | ⏳ | scene build is single-threaded today; Web Workers (+ SharedArrayBuffer) possible |
| Plugin system (dynamic pass/importer loading) | 🔶 static | native .dll/.so loading ❌; passes register via a static factory + side-effect import. A dynamic JS plugin registry ⏳ |
| Settings system (global `Settings`, attribute filters) | ✅ | `Utils/Settings.ts`: colon-flattened options, ordered attribute filters (full-match regex, nested flatten, deprecated `.filter` + negation), `searchpath:`/`standardsearchpath:` directory categories (Renderman `;`/`@`/`&` list semantics via `Utils/PathResolving.ts`; `searchpath:media` feeds the default AssetResolver like SampleApp), `m.settings` python binding, `RenderPass.onOptionsChange` hook; §9: no settings.json autoload, `${ENV}` expands to empty |

### 8.2 Render passes (29 upstream directories, 38 registered pass classes)

Tallies today: 21 pass classes fully implemented, 4 partial, 13 not implemented
(of which 4 ❌ NVIDIA-SDK-bound, 2 🟠 autodiff-blocked).

| Pass | Status | Notes |
|---|---|---|
| AccumulatePass | ✅ | `Double` mode maps to SingleCompensated (fp64 gap 🟡); `maxFrameCount` + `overflowMode` (Stop / Reset / EMA) honoured like native (moving-average kernel mode, count stops at the limit; not in SingleCompensated, like native) |
| BlitPass | ✅ | |
| BSDFOptimizer | 🟠 | no host port; depends on the same slangc autodiff blocker as WARDiffPathTracer (§6.9), plus an optimizer loop |
| BSDFViewer | ✅ | verified vs native (1.9e-4); Slice viewer mode (`viewerMode`) exposed on the host |
| DebugPasses: ColorMapPass / SideBySidePass / SplitScreenPass | ✅ | verified; interactive divider ✅ (native `onMouseEvent`: hover highlight within max(6, dividerSize) px, left-drag moves it, double click recenters, "Show Arrows" — the viewer forwards canvas mouse events to passes before the camera controller, like `Renderer::onMouseEvent`); TextRenderer overlay labels ⏳ |
| DebugPasses: InvalidPixelDetectionPass | ✅ | unmodified upstream shader; functional GPU test (injected NaN→red, Inf→green, valid→black) |
| DLSSPass | ❌ | NVIDIA NGX driver + hardware black box; nearest substitutes: TAA-upscale ✅ or FSR2-WGSL port 🔶 (separate pass, not DLSS parity) |
| ErrorMeasurePass | ✅ | difference kernel (WTexture2D override — rgba32float storage is write-only) + GPU-reduced mean error; reference from input or file (EXR/HDR/browser formats); running-error EMA (one step per landed measurement) + `renderUI`. Measurements surface on `measurements`/`runningError` via async readback — csv file output ❌ (no file IO); `UseLoadedReference` toggle + `getProperties` (native key spellings) |
| FLIPPass | ✅ | LDR path verified vs native (byte MSE 6.7e-5); HDR auto-exposure (reference-luminance median/max → exposure range, async readback §9 — same 1-frame latency as native's member-after-cbuffer-write order) + pooled avg/min/max FLIP via ParallelReduction, both pinned by GPU test vs CPU recomputation |
| GBufferRaster | ✅ | native oracle impossible on this host (ROV), RT-cross-verified; native `forceCullMode` / `cull` options + controls (§9: culls nothing unless forced) |
| GBufferRT | 🟡 | SoftwareRT; verified incl. texGrads (byte-exact) |
| VBufferRT | 🟡 | SoftwareRT; verified |
| VBufferRaster | ✅ | non-indexed vertex-pulling raster (§9: WGSL lacks fragment barycentrics/primitive id — per-corner varyings + flat triangle index instead); cross-verified vs VBufferRT on cornell (1/50175 hit-pixel diff, 0 barycentric outliers); mvec/mask extra channels ⏳ (ROV-dependent); native `forceCullMode` / `cull` options + controls (§9: culls nothing unless forced) |
| ImageLoader | ✅ | browser-decodable formats + `.hdr` + `.dds`/BC + `.exr` (parse-exr; GPU-verified exact vs CPU decode); mips/mipLevel/arrayIndex/outputSize(Fixed = image size)/outputFormat honoured (were accepted-but-ignored) |
| MinimalPathTracer | ✅ | SoftwareRT megakernel; oracle-verified (9.5e-7, §7.1); shades triangle, curve, displaced-triangle and SDF hits |
| ModulateIllumination | ✅ | lives under `Utils/` in the web tree |
| NRDPass | 🟠 SDK absent | NRD SDK not bundled in this Falcor drop (no `external/packman/nrd/`) → denoiser shaders uncompilable here. Host portable; NRD's HLSL source is public, genuine port stays the plan (§11.4). SVGF ✅ meanwhile |
| OptixDenoiser | ❌ | requires CUDA+OptiX. Same substitutes as NRD |
| OverlaySamplePass | ❌ | demo draws via raw ImGui draw lists (no web ImGui); closest equivalent would be DOM overlays — not a 1:1 port target |
| PathTracer | ✅ verified | full upstream loop: NEE+MIS, Uniform/Power/LightBVH emissive samplers, EnvMapSampler, dielectrics/nested priority, guide outputs, adaptive spp (`sampleCount` input), rayCount/pathLength stats. Fixed spp 1–16 + variable spp verified (spp=4 vs native: 10/65536 bad px); curve geometry (`USE_CURVES` + Hair BSDF) verified vs native. `USE_RTXDI` in-tracer ReSTIR direct lighting verified vs native (16-frame temporal reservoir chain, meanAbs 8.9e-4, 42 bad px; needed two storage-buffer-budget moves — see §9). Remaining ⏳: NRD guide outputs; SER ❌ |
| PixelInspectorPass | ✅ | pixel/material inspector: PixelData record via async readback (§9) + `renderUI` panel; two overrides (`ShadingData sd = {}` frontend error, `this = {};` WGSL abort); functional GPU test cross-checks the record against the G-buffer inputs; viewer canvas click-to-select wired (§8.3) |
| RenderPassTemplate | ✅ | authoring skeleton at `render-passes/src/RenderPassTemplate.ts` (registered; pass-through verified in a graph) |
| RTXDIPass | ✅ verified vs native | Full port (PrepareSurfaceData + ReSTIR spatiotemporal resampling + FinalShading). Upstream RTXDI.py replica over Arcade at frames 1/16/64: bias <6e-4, ≤5/3600 bad 8x8 blocks. Overrides: texel buffers → structured, boiling filter compiled out (WaveActiveCountBits; native default off), bool cbuffer members → uint, outputs moved into the FinalShading block (4-bind-group cap), lightInfo+compactLightInfo merged (16-storage-buffer cap); options UI ✅ (`RTXDIPass.renderUI` mirrors RTXDI::renderUI; edits recreate the context like setOptions) |
| SceneDebugger | ✅ | verified (1.0e-5) |
| SDFEditor | ⏳ + 🟠 oracle | not built: needs `.sdf`/`.sdfg` IO + runtime SDF editing/bake (§8.4) + interactive UI. Upstream flagship scene asset absent from the media drop, so native oracle impossible anyway |
| SimplePostFX | ✅ | verified |
| SVGFPass | ✅ ported | feature-verified vs native (sphere_array graph, mean 4.2e-4); all 5 kernels verbatim. Default denoiser (replacing NRD/Optix use-cases) |
| TAA | ✅ ported | feature-verified vs native (jittered GBufferRT graph, mse 8.4e-7) |
| TestPasses: TestRtProgram | ⏳ | exercises RT shader-table/hit-group plumbing; mostly moot on software RT but portable as a megakernel |
| TestPasses: TestPyTorchPass | ❌ | CUDA + PyTorch tensor interop |
| ToneMapper | ✅ | all 6 operators + manual exposure verified; auto-exposure (log-luminance mip chain) verified vs native (sRGB MSE 2.1e-4, zero mean bias); fNumber/shutter/filmSpeed physical exposure verified (upstream test variants, sub-byte bias); native option set completed: exposure mode (Aperture/Shutter priority) + exposure value, white balance (CAT02 von Kries port of `Utils/Color/ColorUtils.h`, D65 preserved at 6500 K) + white point, ReinhardModified white luminance / HableUc2 linear white as options; `useSceneMetadata` accepted (no camera metadata on web scenes yet) |
| Utils (Composite/CrossFade/GaussianBlur) | ✅ | verified |
| WARDiffPathTracer | 🟠 compiler-blocked | §6.9: autodiff primitive device-verified on WebGPU; slangc v2026.12.2 segfaults differentiating the full tracePaths (both wgsl and hlsl targets) → needs a Slang release with the large-function autodiff fix |
| WhittedRayTracer | 🟡 | SoftwareRT, recursion → loop; verified byte-exact |

### 8.3 Ecosystem / tooling

| Component | Status | Notes |
|---|---|---|
| Mogwai app (functional viewer) | ✅ core | loads graph `.py` + `.pyscene`/`.pbrt`, per-frame execute, presents marked output (swapchain blit), play/pause + graph/output pickers, first-person camera, per-pass DOM `renderUI` panel, URL params. FrameCapture ✅ (Capture button: float outputs download as EXR, 8-bit as PNG). VideoCapture ✅ (Record button; §9: MediaRecorder WebM, frames pushed per presented frame like native). Python console ✅ (`m.scene`/`m.activeGraph`/`m.settings` bound to live state, expression echo + print capture, history; Playwright-verified live camera/material edits); canvas click-to-pick wired to PixelInspectorPass. TimingCapture ✅ (`m.timingCapture.captureFrameTime` — collects per-frame CPU ms, downloads on stop, §9: no file IO). Profiler panel ✅ (P key: per-event CPU/GPU table + history graph + capture download). Scene panel ✅ (camera, env map, lights, materials, Animate Scene — native Scene::renderUI subset; no render-settings toggles: the web Scene derives useEnvLight/useAnalyticLights/useEmissiveLights from content) |
| Python scripting / console | 🔶 | Pyodide runs **unmodified** upstream `.py` graphs and `.pyscene` files via a curated `falcor` bridge (factories + SceneBuilderBridge + `AssetResolver`/`AssetCategory`/`SearchPathPriority`). Interactive console ✅ (viewer panel; `m.scene`/`m.activeGraph`/`m.settings`/`m.clock`/`m.profiler` on live state). No auto-generated ScriptBindings, no ScriptWriter ⏳ |
| PyTorch interop (`falcor.pytorch`) | ❌ | no CUDA/torch in browser; ONNX-web-style substitute would be non-parity ⏳ |
| FalcorTest | 🔶 | vitest (unit) + Playwright GPU harness w/ native-oracle image compares (§7); no slang-driven `GPU_TEST` framework ⏳ |
| RenderGraphEditor (ImGui node UI) | 🔶 | basic node editor in the Mogwai viewer (Graph button): passes in dependency layers with input/output ports, edges as curves (`RenderGraph.getEdges()`), click-to-connect / click-to-remove, ★ marks graph outputs, add/remove passes; mid-edit graphs show the compile error in the status line instead of stalling; no drag layout or per-node property editing yet (the pass panel covers properties); Save graph .py downloads `RenderGraph.exportScript()` |
| RenderGraph `.py` export / RenderGraphIR | ✅ | `exportScript()` emits the camelCase image-test dialect (round-trip fixpoint verified); `removeEdge`/`unmarkOutput` added; divergence (§9): no snake_case IR, markOutput channel masks untracked |
| ImageCompare | 🔶 | native tool used on CI host for oracle diffing; its MSE/FLIP gate policy reimplemented inline in the GPU suites + FLIPPass. No standalone in-browser tool ⏳ |
| Importers | see §8.4 | glTF ✅ (TS), FBX 🔶 (assimpjs, `.fbx` full scenes only), PBRT ✅ subset (materials → Standard, or the PBRT material classes via `PBRTImporter:usePBRTMaterials`), `.pyscene` ✅, USD 🔶 subset (tinyusdz-wasm: meshes/xforms/UsdPreviewSurface incl. baseColor/ORM/normal/emissive textures, verified vs native — lights/cameras/skel/subdiv ⏳), Mitsuba ⏳ (no Mitsuba content in the media drop → no oracle) |
| SceneCache | ✅ | `Scene/SceneCache.ts` v4: OPFS binary cache keyed by SHA-256 of the scene source (§9: no file timestamps); geometry incl. skin/morph data + materials + textures (original compressed bytes) + env map (original .hdr/.exr bytes) + lights + camera + curves + node animations/weight tracks + SDF grids (rebuildable recipes: type/params/generator calls, regenerated deterministically on load) + grid volumes (NanoVDB buffers); cache-hit renders byte-identical (cornell, textured tutorial, two_curves, sphere_array, animated_cubes, NDSDF, smoke verified). Only programmatic env maps without source bytes stay uncacheable |
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
| StandardMaterial | ✅ | verified across the oracle suite (incl. specular/normal-mapped pyscene materials — the tutorial G-buffer channel test pins normW/specRough/emissive vs native; a ~6% PT deficit found by the texLOD cross-oracle traced to the pyscene bridge dropping `addNode`'s parent argument, fixed) |
| PBRTConductor | ✅ | override shader + bridge |
| Cloth / Hair / PBRTDiffuse materials | ✅ | oracle-verified vs native (cloth-pt / hair-pt / pbrt-diffuse-pt / pbrt-conductor-pt suites); Hair additionally exercised by the curve scenes (Chiang16) — the old "unverified" marker was stale |
| PBRTDielectric / CoatedConductor / CoatedDiffuse / DiffuseTransmission | ✅ | instantiable via the native Settings key `PBRTImporter:usePBRTMaterials` (first web Settings consumer); one override (CoatedConductor instance aggregate-init vs explicit `__init`); verified vs native (`m.addOptions` oracle, 256-frame PT: per-material region means within 0.8%, bias 2.3e-3, 10/256 16×16 blocks) |
| MERL / MERLMix / RGL measured materials | ⏳ | not instantiable: no factory case, no `.brdf`/`.bsdf` loaders, no data packers, no RGLAcquisition |
| Texture LOD (ray cones / ray diffs) | ✅ core | material texture arrays carry full mip chains (layered blit-chain `generateMips`; material sampler = native trilinear + anisotropy 8) — previously every LOD mode silently sampled mip 0. GBufferRT `texLOD=RayCones` verified vs native (12× closer to the RayCones capture than Mip0; filtering detail is implementation-defined across APIs, §9); explicit-gradient path byte-exact; PathTracer `primaryLodMode` verified (Mip0/RayDiffs; RayCones warns → Mip0 like native): 2×2 cross-oracle — each web mode matches its native counterpart best and the web LOD-effect magnitude (3.13e-2) matches native's (3.0e-2); one override diff (runtime-selected ITextureSampler existential crashes the WGSL backend → single concrete gradient sampler under RayDiffs). Atlas caveat (§9): sub-layer textures (uvScale<1) get a small LOD bias — gradients are texture-space against array dims |
| Analytic lights (Point/Directional/Distant/Rect/Disc/Sphere) | ✅ | verified incl. area-light sampling |
| Emissive geometry (LightCollection) | ✅ | incl. textured-emissive flux integration; LightBVH sampler ✅ incl. `lightBVHOptions` plumbing (sampler + builder options via the native nested-dict keys; BinnedSAH heuristic not ported — warns and uses BinnedSAOH; unit-tested tree structure + defines, end-to-end options oracle vs native); GPU refit ⏳ — rebuild-only |
| EnvMap | ✅ | rotation/intensity/tint; loads Radiance `.hdr` and OpenEXR `.exr` (decodeExr) |
| LightProfile (IES) | ⏳ | dummy binding only; no IES loader/bake |
| Camera (pinhole, jitter, motion vectors) | ✅ | verified (incl. prev-matrix roll) |
| Camera DoF / physical camera | ✅ | `apertureRadius`/`focalDistance` + thin-lens sampling verified vs native (VBufferRT depth/viewW 0 bad px; PathTracer consumes `viewW`); `shutterSpeed`/`ISOSpeed` plumbed (CameraData, pyscene `camera.shutterSpeed`, scene cache) — metadata only, like native |
| Camera controllers | ✅ | `Scene/Camera/CameraController.ts` ports Orbiter / FirstPerson / SixDoF incl. gamepad (dead zone + power curve + rotation/movement mapping, native math verbatim, unit-tested); the viewer drives these ported controllers through a DOM input adapter (normalized mouse pos, key letters + shift/ctrl, gamepad polled per frame; update(now) instead of CpuTimer) and exposes the native "Camera Controller" / "Up Direction" / "Camera Speed" controls in its Scene panel. §9: the Orbiter keeps the current view (target + distance/3.5) because the web Scene has no AABB yet; first-person wheel dolly is a web extra |
| Node / skinned / morph animation | ✅ | CPU skinning + morph (upstream does GPU skinning 🟡); camera/light animation ✅; LINEAR/STEP/CUBICSPLINE |
| Animated vertex caches (Alembic) | ⏳ | no `.abc`/AnimatedVertexCache support |
| Per-clip loop behaviors / global time control | ✅ core | time loops `fmod(t, length)` (AnimationController parity); per-clip pre/post-infinity behaviors (Constant/Linear/Cycle/Oscillate, set via `sceneBuilder.animations[i]` like native) verified vs native on animated_cubes — pre-infinity depth 0 bad px, hit coverage exactly equal; global time control ✅ via `m.clock` (Clock.ts) |
| Motion vectors for animated geometry | ✅ | rigid (prev world matrices) native-exact (mean 1.2e-7); skinned/morphed (prev-position double buffer + IsDynamic) verified by reprojection — native itself writes zero skinned mvecs on this content (probed) |
| Animated-scene BVH | 🟡 | full CPU rebuild per frame (correct, no refit path) |
| GridVolumes (NanoVDB) | ✅ | `.vdb` parsed in-browser → NanoVDB, verified vs native; uncompressed codecs only ⏳ (zip/blosc), `.vdb` frame sequences ⏳, blackbody emission conversion ⏳ |
| Runtime material/light property edits | ✅ | `Scene.getLight/getMaterial` + `updateLights/updateMaterial` re-pack GPU data post-build; verified vs native applying identical python edits (mean 7.1e-4). Emissive edits rebuild the NEE flux tables (dimmed-cornell PT vs native: bias 2.5e-4, 0/1024 blocks). Emissive-presence toggles that flip scene defines still need pass recreation |
| Importer: glTF | ✅ | TS importer: meshes, skinning, morph targets, animations, cameras, lights; KTX2/Draco ⏳ |
| Importer: Assimp | 🔶 partial | assimpjs: full scenes `.fbx` only (other formats mesh-only via `TriangleMesh.createFromFile`); >2 GB-heap FBX aborts (wasm32) — BistroExterior; DDS ✅ / TGA ⏳ textures |
| Importer: PBRT (pbrt-v4) | ✅ subset | camera/lights/shapes/area lights verified; materials → Standard by default, or the dedicated PBRT classes via the `PBRTImporter:usePBRTMaterials` Settings option (native key; area lights stay Standard); textures/spectra/media/curves ⏳ |
| Importer: `.pyscene` | ✅ | unmodified upstream scenes via Pyodide bridge |
| Importer: USD | 🔶 subset | tinyusdz-wasm (1.9MB, reads usda/usdc/usdz): meshes + xform hierarchy + UsdPreviewSurface → Standard incl. UsdUVTexture baseColor (sRGB, V-flip) and roughness/metallic packed ORM like native's CreateSpecularTexture — verified vs native (mask/viewW exact; textured 64spp radiance bias ~1e-5, per-region ≤0.3%). Normal/emissive texture slots plumbed (same path, not separately oracled). ⏳: lights/cameras (not exposed by tinyusdz RenderScene), texture channel selectors (r assumed), UsdSkel, subdivision refinement, instancing |
| Importer: Mitsuba | ⏳ | not started |
| SceneBuilder flags (optimize/merge/dedup) | ✅ core | `SceneBuilderFlags` (native values) via `runSceneScript(..., {flags})` and python `SceneBuilderFlags`; honoured: `AssumeLinearSpaceTextures` (pyscene + FBX/glTF/USD texture loads), `DontUseDisplacement`, `UseOriginalTangentSpace` (builder meshes), `UseCache`/`RebuildCache` (OPFS cache, flags part of the key); accepted no-ops because the web already behaves that way: `DontMergeMaterials`/`DontOptimizeMaterials`/`DontOptimizeGraph` (no merging/optimisation passes), `Force32BitIndices` (always 32-bit), `FlattenStaticMeshInstances` (imports flatten), `UseCompressedHitInfo` (uncompressed), `RTDontMerge*` (software BVH); `NonIndexedVertices`, `DontMergeMeshes`, `TessellateCurvesIntoPolyTubes`, `UseSpecGloss/MetalRough` ⏳ accepted-but-ignored |

### 8.5 Framework, debugging & test infrastructure

| Feature | Status | Notes |
|---|---|---|
| RenderGraph core (compile, alloc, I/O merge, `.py` load) | ✅ | verified across all feature graphs |
| RenderPassReflection completeness | ✅ | full `Field` API (`rawBuffer`/`texture1D`/`texture2D`/`texture3D`/`textureCube`/`resourceType`, `kMaxMipLevels`, `name`/`desc`, `isValid`, native `merge` conflict errors + `operator==`); allocation mirrors `ResourceCache::createResourceForPass` (raw buffers, 1D/3D/cube/MSAA textures, size-0 → graph dims, Unknown → graph default format, `None` bind flags resolved from the format's WebGPU capabilities via `Device.getFormatBindFlags`); external inputs surface in `connectedResources`; `Persistent` fields keep their resource across recompiles while unchanged (§9). Web rules: 1D outputs never get render-target usage; UAV-bound r8/r16uint promote to r32uint |
| PixelDebug host (shader `print()` readback/console) | ✅ | `Utils/Debug/PixelDebug.ts` + portable override (Atomic counters + flat record buffer replace the ParameterBlock/UAV-counter layout): print/assert records captured per selected pixel, typed decode verified end-to-end; §9: async readback, message strings surface as hashes (slang-wasm lacks hashed-string reflection); pass UI wiring ⏳ |
| PixelStats | ✅ | packed-atomic-buffer collection (verified vs native rayCount/pathLength dumps) + `Rendering/Utils/PixelStats.ts` `getStats()` aggregate (GPU region sums, async readback ~1 frame late, docs §9); PathTracer exposes `getPixelStats()` |
| WarpProfiler | ⏳ | subgroup-dependent; portable where `subgroups` exists |
| BSDFIntegrator (white furnace) | ✅ | `Rendering/Materials/BSDFIntegrator.ts` + one override (wave reduction → shared-memory tree, no `= {}` ShadingData); upstream FalcorTest values reproduced (four incident angles, rough dielectric StandardMaterial); async readback (§9) |
| Algorithm library | ✅ | ParallelReduction ✅, PrefixSum ✅, BitonicSort ✅ (portable shared-memory override of the NVAPI warp-shuffle kernel, §9; exact vs CPU chunk sort incl. 2D dispatch + tail padding) |
| Utils/Math | ✅ | Vector/Matrix/Quaternion ✅; CubicSpline host port (`Utils/Math/CubicSpline.ts`, drives curve tessellation, unit-tested); `SphericalHarmonics.slang` compiles to WGSL — 16 basis functions GPU-verified against a CPU transcription |
| Gui (Dear ImGui) | 🔶 | DOM `UIWidgets` (text/button/checkbox/slider/dropdown/group); `renderUI` ported on 26 of 28 pass classes (native controls via text/button/checkbox/slider/dropdown/group; "Output size"/"Size in pixels" wired through `RenderPass.requestRecompile()`; InvalidPixelDetection and RenderPassTemplate have no controls natively); TextRenderer/Font/PixelZoom ⏳ |
| Video (FFmpeg encode/decode) | ❌ native / ⏳ substitute | WebCodecs route not built |
| AssetResolver | ✅ | `Core/AssetResolver.ts`: per-category search paths (Any/Scene/Texture, First/Last priority, category → Any fallback), `getDefaultResolver` seeded with `/Falcor/media` (SampleApp's project media dir); `.pyscene`/`.pbrt` loads push the script directory first and restore afterwards (Mogwai `loadScript`); SceneBuilder imports/`createFromFile`/`loadTexture`/EnvMap/Grid, ImageLoader and ErrorMeasure resolve through it; python `AssetResolver.default_resolver.add_search_path`/`resolve_path` + `AssetCategory`/`SearchPathPriority`. §9: async (HTTP HEAD existence probe), no cwd-relative lookup, `resolvePathPattern` (UDIM/`.vdb` sequence globbing) ❌ — no directory listing |

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
10. **The 16-storage-buffer-per-stage cap shapes bindings.** This Chromium's
   per-stage limit is 16 even at the adapter maximum. Small read-only tables
   (RTXDI neighbor offsets, per-texture uv-scale info) live in 1-row textures
   behind buffer-style `__subscript` wrappers. The PathTracer+RTXDI megakernel
   sits exactly at 16; combining USE_RTXDI with pixel stats or curve scenes
   would exceed it (not currently co-usable).
11. **`Persistent` reflection fields outlive recompiles.** Native only promises
   the resource is stable between `execute()` calls; the web graph recompiles far
   more often (resize, scene change, pass edits), so a `Persistent` field also
   keeps its resource and contents across recompiles while its resolved
   description (type, dims, format, flags) is unchanged. Graph default format
   for `Unknown` outputs stays `RGBA32Float` (native: swapchain format) unless
   `onResize(w, h, format)` supplies one.
12. **Asset resolution is asynchronous and URL-based.** `AssetResolver.resolvePath`
   probes candidates with HTTP HEAD (GET-range fallback; a dev server's
   `index.html` fallback does not count as a hit) and returns a Promise; native
   returns synchronously from the filesystem. There is no working-directory
   lookup step and no `resolvePathPattern` (browsers cannot list directories),
   so `<UDIM>` texture sets and `.vdb` frame sequences stay unsupported.
13. **Animation clip behaviors match native** (Constant default; per-clip
   Linear/Cycle/Oscillate honored, §8.4); the Linear edge-slope extrapolation
   quantizes at f32 keyframe precision on both sides, so it is
   tolerance-compared like all float outputs.
14. **Raster G-buffer passes cull nothing unless `forceCullMode` is set** (native
   defaults to back-face culling) so raster and software-RT coverage agree; the native
   `forceCullMode` / `cull` options are honoured when given.
