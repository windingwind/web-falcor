# web-falcor — Feature parity matrix & divergences

Part of the [web-falcor design docs](../DESIGN.md). Section numbers (§8–§9) are
kept stable across the split so the `§N` cross-references throughout the docs
stay valid. The ✅ / 🟡 / 🔶 / ❌ / 🟠 status markers are defined in
[architecture.md §1](architecture.md#1-goals-and-ground-rules).

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
| NRDPass | 🟠 SDK absent | The NRD SDK is NOT bundled in this Falcor drop (`external/packman` has no `nrd/`; NRDPass loads `nrd/Shaders/Source/*.hlsl` that don't exist) → the denoiser shaders can't be compiled here. Host pass is portable; shaders need the SDK checkout. SVGF ✅ available meanwhile (§11.4) |
| OptixDenoiser | ❌ | requires CUDA+OptiX. Same substitutes as NRD |
| OverlaySamplePass | ✅ | |
| PathTracer | ✅ core | SoftwareRT megakernel, oracle-verified w/ NEE+MIS, Uniform/Power emissive samplers + EnvMapSampler (§7.1). v1 limits: fixed spp=1, no guide/NRD outputs, LightBVH sampler pending, volumes/SDF pending |
| PixelInspectorPass | ✅ | |
| RenderPassTemplate | ✅ | |
| RTXDIPass | ✅ verified vs native | Full port (PrepareSurfaceData + ReSTIR spatiotemporal resampling + FinalShading). Upstream RTXDI.py replica over Arcade at frames 1/16/64: bias <6e-4, ≤5/3600 bad 8x8 blocks. Overrides: texel buffers -> structured, boiling filter compiled out (WaveActiveCountBits; native default off), bool cbuffer members -> uint (non-host-shareable), outputs moved into the FinalShading block (4-bind-group cap) + write-only, lightInfo+compactLightInfo merged (16-storage-buffer cap) |
| SceneDebugger | ✅ | |
| SDFEditor | ✅ | pure compute + UI |
| SimplePostFX | ✅ | |
| SVGFPass | ✅ ported | feature-verified vs native (sphere_array feature graph, mean 4.2e-4); all 5 kernels verbatim, no overrides. Default denoiser (replacing NRD/Optix use-cases) |
| TAA | ✅ ported | feature-verified vs native (jittered GBufferRT graph, mse 8.4e-7); bool->uint cbuffer override only |
| TestPasses | ✅/❌ | GPU-test passes ✅; PyTorch interop pass ❌ (CUDA) |
| ToneMapper | ✅ | |
| Utils (Composite/CrossFade/GaussianBlur) | ✅ | |
| WARDiffPathTracer | 🟠 compiler-blocked | §6.9 (autodiff primitive works; full-path-tracer diff crashes slangc v2026.12.2) |
| WhittedRayTracer | 🟡 | SoftwareRT, recursion → loop |

### 8.3 Ecosystem / tooling

| Component | Status | Notes |
|---|---|---|
| Mogwai app (functional viewer) | ✅ verified | Interactive browser viewer: loads a render-graph .py + .pyscene, runs the graph each frame, presents a marked output to the canvas (presentToCanvas swapchain blit); DOM controls (play/pause, graph picker, output picker). The render loop is the verified GPU-harness code; a headless smoke test renders + presents the cornell path-tracer end to end. FrameCapture→download, VideoCapture→WebCodecs (FFmpeg ❌) |
| Python scripting / console | 🔶 | TS API (shape-identical) + graph-`.py` compatibility layer; full CPython via optional Pyodide plugin |
| PyTorch interop (`falcor.pytorch`) | ❌ | no CUDA/torch in browser; gradient buffers exposed to JS/ONNX-web instead |
| FalcorTest | ✅ | vitest + Playwright harness (§7) |
| RenderGraphEditor (ImGui node UI) | ⏳ stretch | The functional viewer is done; the Dear-ImGui-wasm RenderGraphUI node editor is dev tooling orthogonal to Falcor's rendering parity — scoped as a stretch, not a rendering feature |
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
