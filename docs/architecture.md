# web-falcor — Goals, platform & architecture

Part of the [web-falcor design docs](../DESIGN.md). Section numbers (§1–§3) are
kept stable across the split so the `§N` cross-references throughout the docs
stay valid.

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
3. **Feature-gap honesty.** Every feature is classified in the parity matrix (§8) with
   one of these status markers (the canonical legend, used throughout the docs):
   - ✅ **Portable** — direct implementation, same behavior.
   - 🟡 **Emulated** — same API and observable behavior, different mechanism
     (e.g. software ray tracing in compute).
   - 🔶 **Replaced** — same purpose, different technology (e.g. Python → TypeScript
     scripting), API kept shape-compatible.
   - ❌ **Impossible** — cannot be provided on the web platform at all; API exists but
     throws `UnsupportedFeatureError` with a pointer to this document.
   - 🟠 **Blocked** — a tooling or asset gap (*not* a web-platform limitation) that
     prevents completion, with the specific blocker documented (e.g. a slangc crash,
     or an SDK/asset absent from this Falcor drop).
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
├── DESIGN.md                  # design-docs index
└── docs/                      # this document + the rest of the design docs
```
