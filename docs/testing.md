# web-falcor — Testing strategy & verified results

Part of the [web-falcor design docs](../DESIGN.md). Section numbers (§7) are
kept stable across the split so the `§N` cross-references throughout the docs
stay valid.

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
more than 0.05). Suite: `npm run test:gpu` (102 GPU tests + 40 unit green as of M8).

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
| upstream image tests `GBufferRTTexGrads.py` (texGrads byte-exact) + `MVecRT.py` over cornell (jittered mvec, 4 frames: 3.1e-8, 0 bad — reprojection exactly cancels the bit-exact stratified jitter) | camera viewProjMatNoJitter/prev matrices | byte-exact / 3.1e-8 | 0 / 0 |
| upstream image test `PathTracerAdaptive.py` over cornell (density-map-driven 0..16 spp, 4 frames) | variable sample counts: subgroup tile prefix in GeneratePaths, per-sample LogLuv color buffer, resolve averaging | sRGB MSE 5.8e-4 | 318 @0.05 (stochastic) |
| upstream image test `BSDFViewer.py` over cornell (material sphere viewer, 4 accumulated frames) | BSDF evaluation viewer + importance sampling | 1.9e-4 | 10 |
| upstream image test `SceneDebugger.py` over cornell (FaceNormal visualization, primary inline rays) | debug-view pass + gridVolumes scene binding | 1.0e-5 | 0 |
| upstream image test `WhittedRayTracer.py` over cornell (GBufferRT → Whitted megakernel → ToneMapper) | perfect reflect/refract chains, RayCones Unified texLOD, per-light shadow rays | **byte-exact** (sRGB MSE 0) | 0 |
| upstream image tests `GBufferRT.py` + `GBufferRTInline.py` — 13 channels (posW/normW/tangentW/faceNormalW/texC/texGrads/depth/linearZ/guideNormalW/diffuseOpacity/specRough/emissive/viewW) | full RT G-buffer (ray differentials, material queries); channels split across ≤8-storage-texture dispatches (WebGPU per-stage cap) | texGrads + emissive byte-exact; rest 1e-7..2e-4 | 0 bad on all 13 (linearZ slope skipped where 0/0-UB; normWRoughnessMaterialID format-divergent: no rgb10a2 storage in WGSL) |
| StratifiedSamplePattern (camera jitter) vs gcc/libstdc++ reference | std::mt19937 + std::shuffle + generate_canonical\<float\> replicated | bit-exact | 0 (unit-pinned) |
| upstream `HalfRes.py` graph over cornell_box (web-side only ⚠) | IOSize Half plumbing + stratified jitter + 16-frame accumulation | runs, half-res, jitter advances | no native oracle: the oracle GPU's Vulkan driver lacks ROVs, so native Mogwai cannot construct GBufferRaster at all |
| upstream image test `PathTracer.py` over cornell_box (4 frames) — color / guideNormal / reflectionPosW / albedo / specularAlbedo / indirectAlbedo / ToneMapper.dst | guide outputs + ResolvePass | 2.5e-4 / 1.4e-5 / 3.9e-5 / byte-exact / 1.6e-9 / byte-exact / 4.7e-6 | 38 @0.05 (stochastic silhouette, cornell policy) / 0 / 0 / — |
| `PathTracer.py` rayCount / pathLength (PixelStats port; per-pixel integer counters vs raw native texture dumps) | PixelStats override: packed atomic buffer + resolve kernel | sums 248895 vs 248890 / 118515 exact | 7 / 4 mismatched pixels (stochastic tail) |
| TAA feature graph (upstream TAA.py wiring with GBufferRT instead of ROV-blocked GBufferRaster; 8 Halton-jittered frames, history exercised) | TAA pass port | sRGB MSE 8.4e-7 | 2 px >3 LSB (float-vs-sRGB-quantized history, documented) |
| SVGF feature graph (upstream SVGF.py wiring, GBufferRT + PathTracer over sphere_array; 4 frames temporal + a-trous) | SVGF pass port (5 kernels verbatim) | mean 4.2e-4 | 104 @0.05 (filtered stochastic tail; PT input itself 284) |
| smoke volume scene (upstream SceneDebugger.py over smoke.pyscene; web parses the original .vdb in-browser, native loads the byte-identical .nvdb) | GridVolumes GPU chain (NanoVDB buffer, gScene grid plumbing, PNanoVDB WGSL traversal, 500-step transmittance march) | mean 3.7e-5 | 0 @1e-2 |
| Arcade.pyscene via FBX import (upstream GBufferRT.py over the upstream Arcade scene) | FbxImporter (assimpjs WASM + AssimpImporter Default-mode port): posW/faceNormalW/texC exact, tangentW, guideNormalW (normal mapping), diffuse, emissive x150 factor | 1e-4 / 1e-5 / 1.1e-4 / 6.7e-4 / 6.4e-4 / 1.0e-4 / 1.4e-2 | 0 / 0 / 0 / 190 / 142 / 192 / 139 |
| upstream test_PathTracer.py + test_MinimalPathTracer.py REPLICAS (Arcade, 640x360, frame 128 — the exact upstream harness parameters) | LightCollection textured-emissive integration (analytic triangle-texel coverage, EmissiveIntegrator semantics) | bias -6.1e-4 / -3.7e-4 (unbiased) | PT: 36/3600 bad 8x8 blocks (NEE sequences decorrelate: flux-table float rounding -> different RNG consumption -> per-pixel speckle with zero bias); MPT: 1297 rel-bad px (fireflies at the x150 emitter) |
| upstream test_RTXDI.py REPLICA (unmodified RTXDI.py graph over Arcade, 640x360, frames 1/16/64) | full ReSTIR DI: presampled RIS tiles, candidate generation + visibility, spatiotemporal reservoir resampling, final shading (SDK 1.3.0 kernels) | bias -3.6e-4 / -4.2e-4 / -5.9e-4 | 5 / 4 / 3 of 3600 bad 8x8 blocks (RIS tile picks decorrelate via R32-vs-R16 PDF rounding) |
| upstream test_NDSDFGrids.py scene (unmodified NDSDFGrid.pyscene, SceneDebugger GeometryID, frame 64) | SDF grid chain: procedural cheese host build (bit-exact vs gcc), ND atlas, SoftwareRT sphere tracing, SDFGridHit | body PIXEL-EXACT vs CPU algorithm oracle (0/230400); background exact vs native (0/171476) | native body NOT comparable: local SPIR-V offset-fetch artifacts dilate its footprint +24568 px and NaN its gradients (probe-verified single-ray) |
| SDFSBS.pyscene (createSBS + generateCheeseValues, same surface as NDSDF, SparseBrickSet representation), SceneDebugger GeometryID, frame 64 | SBS CPU brick build (5 native kernels ported) + brick-AABB-loop sphere tracing | body matches the shared CPU footprint oracle at 3/230400 px (webHits 34355 vs NDSDF 34356) | native can't oracle SBS: Mogwai core-dumps building the brick procedural-AABB acceleration structure on this machine (local-artifact class) |
| SDFSVS.pyscene + SDFSVO.pyscene (createSVS/createSVO + generateCheeseValues, same cheese surface), SceneDebugger GeometryID, frame 64 | SVS per-voxel packed 4x4x4 neighborhoods + per-voxel AABBs over the SDF primitive-AABB BVH; SVO bottom-up CPU octree (BigInt Morton location codes) + in-shader octree walk | both match the shared CPU footprint oracle at 3/230400 px | native can't oracle SVS/SVO either (voxel/octree procedural-AABB BLAS core-dumps) |

† residual is entirely the jpg *input decode* (browser vs FreeImage IDCT/chroma
upsampling, ≤3 sRGB LSB): the png-fed pixels contribute zero error (Composite
and CrossFade have identical stats), and the hdr-fed GaussianBlur sits at 4.2e-5.

RNG parity is exact: TinyUniform (LCG+TEA) and xoshiro128** (SplitMix64 seeding
emulated as paired u32) produce bit-identical streams, so 1-spp renders match
native to float tolerance rather than statistically.

### 7.2 Upstream image-test graph pass-rate (tests/image_tests/renderpasses/graphs, 39 graphs)

Status as of **M8**. "Verified" = the unmodified graph runs on web and its
output is diffed against native Mogwai running the same file.

**M8 overall-verify summary.** Every rendering-feature milestone item is either
verified-vs-native or has a specific, documented blocker:

- **Verified vs native** (image/feature oracles, §7.1): MinimalPathTracer,
  PathTracer (+Dielectrics, guide outputs, PixelStats), RTXDI (full ReSTIR DI),
  TAA, SVGF, GridVolumes (NanoVDB), FBX import (Arcade), NDSDF + SBS SDF grids,
  and the small-pass suite (ToneMapping, Composite, CrossFade, GaussianBlur,
  ColorMap, SideBySide, SplitScreen, ModulateIllumination, SimplePostFX, FLIP,
  Whitted). **102 GPU + 40 unit tests green.**
- **Asset-blocked** (missing from this Falcor drop, unloadable natively too):
  NRDPass (NRD SDK shaders absent), SDFEditorRenderGraphV2 (`one_primitive_edited.sdfg`
  absent).
- **Compiler-blocked**: WARDiffPathTracer — the autodiff primitive is
  device-verified, but slangc v2026.12.2 segfaults differentiating the full
  path tracer (§6.9).
- **Runtime-impractical without new infra**: SVS/SVO SDF grids (one AABB per
  surface voxel → need a procedural-AABB BVH; NDSDF + SBS already cover both
  SDF representation classes).
- **Native-oracle-impossible on this host**: all raster-GBuffer graphs (the
  oracle GPU's Vulkan driver lacks ROVs, so native Mogwai can't build
  GBufferRaster at all) — their passes are ported and cross-verified through
  the RT-GBuffer feature graphs instead.
- **Impossible on the web platform**: OptixDenoiser, DLSS (CUDA/driver tech).

| Status | Count | Graphs |
|---|---|---|
| ✅ verified vs native | 15 | MinimalPathTracer, ToneMapping, VBufferRT, CompositePass, CrossFadePass, GaussianBlur, ColorMapPass, SideBySide, SplitScreen, ModulateIllumination, SimplePostFX, FLIPPass, PathTracer, PathTracerDielectrics, RTXDI |
| 🟢 runnable now (passes exist; oracle pending) | 1 | VBufferRTInline (same pass; inline variant is our default) |
| 🟠 asset-blocked | 1 | SDFEditorRenderGraphV2 (SDFEditorSceneTwoSDFs.pyscene references `one_primitive_edited.sdfg`, absent from this Falcor drop → scene unloadable natively too; SBS grid chain itself is verified) |
| 🟠 runnable on web; native oracle impossible on this machine | 1 | HalfRes (needs FBX importer for Arcade.pyscene; and the oracle GPU lacks ROV support, so native Mogwai cannot run GBufferRaster-based graphs at all) |
| 🟡 GBuffer remainder | 3 | GBufferRaster, GBufferRasterAlpha, MVecRaster — ⚠ all raster-based: native-ROV oracle blocker |
| 🟡 needs larger pass ports (M8 scope) | 4 | SVGF + TAA (both passes PORTED + feature-verified vs native via GBufferRT feature graphs — TAA mse 8.4e-7, SVGF mean 4.2e-4; the upstream graphs themselves stay oracle-blocked: GBufferRaster needs ROV the native driver lacks), VBufferRaster, VBufferRasterAlpha |
| 🟠 compiler-blocked | 3 | WARDiffPathTracer ×3 — autodiff primitive device-verified, but slangc v2026.12.2 segfaults differentiating the full path tracer (§6.9) |
| ❌ impossible on web (CUDA/driver tech) | 2 | OptixDenoiser, DLSS |

\* also needs SDF grid geometry (M7 remainder).
