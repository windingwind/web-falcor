# web-falcor — Shader system & software ray tracing

Part of the [web-falcor design docs](README.md). Section numbers (§4–§5) are
kept stable across the split so the `§N` cross-references throughout the docs
stay valid.

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
