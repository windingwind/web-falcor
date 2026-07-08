# web-falcor — Roadmap & resolved design questions

Part of the [web-falcor design docs](README.md). Section numbers (§10–§11)
are kept stable across the split so the `§N` cross-references throughout the
docs stay valid.

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
| **M8** (in progress) | ✅ RTXDI (full ReSTIR DI, verified), ✅ FBX/Assimp import (Arcade, verified), ✅ SDF grids NDSDF+SBS (verified); 🟠 NRD (SDK absent), 🟠 WARDiffPathTracer (autodiff device-verified, full-path-tracer diff crashes slangc v2026.12.2), 🟠 SDFEditor (asset absent); ✅ SDF grids (all 4 types verified), ✅ Mogwai functional viewer (render loop + present, smoke-tested); ⏳ ImGui-wasm node editor + WebGL2 raster (stretch tooling) | pass-rate report + parity matrix finalized (§7.2); every rendering-feature item verified-vs-native or with a documented blocker |

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
