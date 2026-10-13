# Migrating from native Falcor

Part of the [web-falcor design docs](README.md). This guide is for people with their own work on native Falcor
(custom render passes, Slang shaders, changes to Falcor itself) who want to run it in web-falcor. The
[README](../README.md#migrating-from-native-falcor) has the short version.

Most of a Falcor project carries over: scenes, render-graph scripts, pass names, properties and the Slang shader
library. The rest falls into three groups:

1. **Host code** (C++ → TypeScript). This port is mechanical; [§2](#2-porting-host-code-c--typescript) maps the
   API.
2. **Shaders that need a rewrite** to compile to WGSL or to run under WebGPU's limits. This is where the effort
   goes; [§3](#3-porting-shaders) lists every kind of rewrite this project has needed.
3. **Code that compiles but behaves differently.** It needs to be known about, because nothing reports it;
   [§4](#4-differences-that-compile-but-change-results) lists every case.

Contents:

- [1. Before you start](#1-before-you-start)
- [2. Porting host code (C++ → TypeScript)](#2-porting-host-code-c--typescript)
- [3. Porting shaders](#3-porting-shaders)
- [4. Differences that compile but change results](#4-differences-that-compile-but-change-results)
- [5. Browser and GPU differences](#5-browser-and-gpu-differences)
- [6. Reading the errors](#6-reading-the-errors)
- [7. Checking a port](#7-checking-a-port)

## 1. Before you start

**Pick the route.** Each kind of change has its own route ([extending.md](extending.md)):

| What you have | Port it as |
| --- | --- |
| New render passes, using Falcor's API as it is | A **plugin** in your own repo: `npx @web-falcor/mogwai new MyPass` |
| Changes to Falcor itself (materials, scene, lights, `Core/API`) | A **core change** under `packages/falcor/src/`, which mirrors `Falcor/Source/Falcor` file for file |
| Edits to upstream shaders | A **shader override** under `packages/falcor/shaders/WebFalcor/Overrides/<same path>` ([extending.md](extending.md#upstream-shaders-that-dont-compile)) |

**Find the closest template.** Every upstream pass has already been ported, in
[`packages/render-passes/src/`](../packages/render-passes/src/). If an upstream pass needed shader changes, its
web version is in [`WebFalcor/Overrides/`](../packages/falcor/shaders/WebFalcor/Overrides/). For any pass yours was
derived from, diff the native and web versions: that shows most of what your port needs.

**Know the compiler version.** Native Falcor 8.0 pins Slang **2024.1.34**. web-falcor compiles with slang-wasm
**2026.18.2**. Some code that compiled natively is rejected by the newer front end even before WGSL is involved
([§3.10](#310-slang-front-end-changes-2024134--2026182)).

## 2. Porting host code (C++ → TypeScript)

### 2.1 API mapping

| Native Falcor (C++) | web-falcor (TypeScript) |
| --- | --- |
| `class X : public RenderPass`, `FALCOR_PLUGIN_CLASS`, `registerPlugin()` | `class X extends RenderPass`; at module top level, `registerRenderPass("X", (device, props) => new X(device, props))` |
| Plugin DLL | ES module loaded with `?plugin=<url>` or `loadPluginLibrary(url)` |
| `X(ref<Device>, const Properties&)`, `props.get<T>(k, def)` | `constructor(device, props)`, `props.get(k, def)`, `props.getOpt(k)`, `props.has(k)` |
| `reflect`, `compile`, `execute`, `renderUI`, `getProperties`, `setProperties` | The same methods, with `override`. `compileData.defaultTexDims` is a `[w, h]` tuple |
| `setScene(RenderContext*, const ref<Scene>&)` | `override setScene(scene: Scene \| null)`, with no context argument |
| `r.addOutput(n, d).format(f).bindFlags(b).texture2D(w, h)` | The same chain. `.flags(FieldFlags.Optional)` |
| `renderData.getTexture(n)`, `renderData[n]->asBuffer()` | `renderData.getTexture(n)`, `renderData.getBuffer(n)`. They return **`undefined`** when the field is unconnected |
| `renderData.getDictionary()` | `renderData.dictionary`, a `Map<string, unknown>` |
| `ComputePass::create(device, path, "main", defines)` | `ComputePass.create(device, { path, csEntry, defines, typeConformances })` |
| `FullScreenPass::create`, `RasterPass::create` | `FullScreenPass.create(device, { path, defines })`, `RasterPass.create(device, { path, vsEntry, psEntry, defines })` |
| `pass->addDefine(n, v)` | `pass.program.addDefine(n, v)`, or recreate the pass (§2.2) |
| `DefineList{{"A", "1"}}` | `new DefineList().add("A", 1)`, or a plain object `{ A: 1 }` |
| `var["CB"]["gFrameDim"] = uint2(w, h);` | `root["CB"]["gFrameDim"] = [w, h];` (array-likes for vectors, see §2.3) |
| `var["gScene"]` + `mpScene->bindShaderData(var["gScene"])` | `scene.bindShaderData(root)`. It takes the **root** var |
| `mpScene->getSceneDefines()` + `addShaderModules` + `getTypeConformances()` | `scene.getSceneDefines()` alone. Material dispatch is selected by defines |
| `mpScene->getCamera()`, `mpScene->useEnvLight()` | `scene.camera`, `scene.useEnvLight`. Both are getters, not methods |
| `RtProgram`, `RtBindingTable`, `mpScene->raytrace(...)` | A `ComputePass` whose shader traces with `SceneRayQuery` ([§3.1](#31-ray-tracing-dxr-pipelines-become-compute)) |
| `mpScene->rasterize(ctx, state, vars, cull)` | None. Encode the draws from `scene.getMeshDrawData()`, as [GBufferRaster.ts](../packages/render-passes/src/GBuffer/GBufferRaster.ts) does |
| `device->createTexture2D(w, h, fmt, arr, mips, init, flags)` | The same signature. `createTexture1D` does not exist; use `new Texture(device, { type, ... })` |
| `createStructuredBuffer(size, n, flags, memType, init)` | `createStructuredBuffer(structSize, n, flags, init)`, without `memType`. There is no `ShaderVar` overload; take the size from reflection (§2.3) |
| `buf->map()`, `buf->getBlob()` | `await buf.getBlob()`, `await ctx.readTextureSubresource(...)`. `map()` works only for `MemoryType.Upload` |
| `clearUAV(uav, value)` | `ctx.clearBuffer(buf)` (zeros only), `ctx.clearTexture(tex, [r, g, b, a])` |
| `blit(srcSrv, dstRtv)` | `ctx.blit(srcTex, dstTex, ...)`, which takes textures, not views |
| `Fbo::create2D(device, w, h, Fbo::Desc)` | `new Fbo()` + `attachColorTarget(tex, i)`. `Fbo.create2D` takes a single color format |
| `Sampler::Desc d; d.setFilterMode(...)` | `device.createSampler({ magFilter, minFilter, addressModeU, ... })` |
| `widget.var("Scale", mScale, 0.f, 4.f)`; `if (w.checkbox(...))` | `ui.slider("Scale", this.scale, 0, 4, 0.01, (v) => (this.scale = v))`, `ui.checkbox(label, v, cb)`, `ui.dropdown`, `ui.button`, `ui.text`, `ui.group`. That is the whole widget set |
| `onMouseEvent`, `onKeyEvent` | Declare `onMouseEvent(e)` / `onKeyEvent(e)` without `override`. Mogwai calls them if they exist; `e.pos` is in [0, 1] |
| `FALCOR_PROFILE(ctx, "name")` | The graph profiles each pass automatically. Inside a pass, use `this.device.profilerHook?.scope("name", fn)` |
| `logInfo`, `logWarning`, `logError`, `logFatal` | `Logger.info`, `Logger.warning`, `Logger.error`, `Logger.fatal`. `Logger.fatal` does **not** throw |
| `FALCOR_ASSERT`, `FALCOR_THROW` | `assert(cond, msg)` (always on), `throw new RuntimeError(...)` |
| pybind11 `ScriptBindings` for the pass | Nothing to register: the pass's public JS getters, setters and methods are visible to Python and JS scripts |
| `ref<T>`, destructors | Garbage collection. GPU resources have `destroy()`, but render passes have no destroy hook |
| `float3`, `float4x4`, `uint2` | The `float2/3/4` and `float4x4` classes in `Utils/Math`. There is no host `uint2`; use `[x, y]` tuples. JS numbers are f64, so use `Math.fround` where native rounding matters |

### 2.2 Behaviour that differs from native

- **Shader compilation is synchronous and blocking.** `ComputePass.create` compiles Slang → WGSL in slang-wasm and
  creates the pipeline before it returns. So does the first `execute` after a define change. That frame stalls;
  there is no "not ready" frame to handle.
- **Define changes.** The ports set `this.pass = null` wherever a define changes (a UI callback, a property
  setter, `setScene`) and recreate the pass lazily in `execute`. `Program` keeps a **reference** to the
  `DefineList` you pass in, so pass a `clone()` if several passes share one list.
- **Fetch the root var each frame and rebind everything.** A define change or an F6 shader reload rebuilds the
  vars, which drops earlier bindings. Call `pass.getRootVar()` inside `execute`, never cache it.
- **Every declared resource must be bound.** Native binds null views silently. Here, an unbound resource that
  survives dead-code elimination throws `Shader parameter 'X' is not bound`. Bind 1×1 dummy textures and buffers
  for optional inputs ([PathTracer.ts](../packages/render-passes/src/PathTracer/PathTracer.ts) shows the
  pattern). A writable dummy must match the shader's storage format, and two bindings in one dispatch must not
  share one writable dummy. Setting a parameter that the shader declares but never uses is a no-op.
- **Scene updates.** There is no `IScene::UpdateFlags`, `onSceneUpdates` or `camera->getChanges()`. Compare
  `scene.updateVersion` and `scene.camera.getChangeKey()` between frames. When scene defines or render settings
  change, the graph calls `setScene()` on **every** pass again, so drop cached programs there.
- **Refresh flags.** `kRenderPassRefreshFlags` does not exist, so a custom pass cannot tell a downstream
  AccumulatePass to reset. AccumulatePass resets only on scene and camera changes.
- **Readback is async.** `getBlob`, `readBuffer` and `readTextureSubresource` return Promises, and `execute` is
  synchronous. Start a readback with `.then()`, guard it with an in-flight flag, and use the result one frame
  late (see [PixelInspectorPass.ts](../packages/render-passes/src/PixelInspectorPass/PixelInspectorPass.ts)).
  `initAsync()` is the place for async setup; the graph awaits it before the first frame.
- **The UI is retained, not immediate mode.** `renderUI` runs when Mogwai rebuilds the panel, not every frame.
  Use the widget callbacks to change state. Text such as "frames accumulated" does not refresh on its own.
- **Errors arrive at different times.** Slang errors throw synchronously from `create`. WebGPU validation errors
  arrive later, through the device's `uncapturederror` handler, so they don't point at the call that caused them.
- **Files.** Assets are fetched by URL: from the served media directory, or from a local folder the user opened
  ([npm.md](npm.md#run-the-viewer)). There are no absolute paths, environment variables or child processes.

### 2.3 Binding values from TypeScript

| Shader type | Assign |
| --- | --- |
| `float`, `int`, `uint`, `bool` | A JS `number` or `boolean`. A `bool` is stored as a u32 |
| `float3`, `uint2`, … | An array-like of the right length (`[x, y, z]`, `Float32Array`). A `float3` instance needs `.toArray()` |
| `float4x4`, `float3x4` | A `float4x4` instance, or a flat **row-major** array (12 floats for `float3x4`) |
| `float a[N]` | The whole array: `root["CB"]["a"] = [...]`. `root["CB"]["a"][i]` is not supported |
| A struct member | Set it field by field. Assigning a whole struct throws `not implemented`; for a raw constant buffer use `pass.getParameterBlock().setCBufferBlob(name, bytes)` |
| `Texture2D`, `RWTexture2D`, buffers, `SamplerState` | A `Texture`, `Buffer` or `Sampler`, or a `GPUTextureView` from `tex.getSRV()` |

**Layouts follow WGSL, not C++.** Structs in buffers use WGSL std430 and constant buffers std140. A `float3`
member takes 16 bytes and the struct size rounds up to its alignment. A buffer sized with C++ `sizeof` is too
small, and WebGPU then rejects the **whole command buffer**: the frame goes black with only a validation message
in the console. Take strides and sizes from the reflected type, not from a host struct.

### 2.4 Shaders of a plugin

A plugin's shaders are not in the shader file list, so a plugin passes the source as a string module:

```ts
import src from "./MyPass.cs.slang?raw";
ComputePass.create(device, { modules: [{ sources: [{ string: src, path: "Plugins/MyPass/MyPass.cs.slang" }] }] });
```

A string module can `import` any Falcor module. `FullScreenPass` and `RasterPass` only take a `path`, and so does
any file that another shader imports. Register those files first with
`device.programManager.addShaderFiles({ "Plugins/MyPass/MyPass.ps.slang": src })`, then use the path. This
reloads every program, so do it once at startup.

## 3. Porting shaders

### 3.0 What the compiler handles for you

These rewrites run on every shader in the runtime path (`SlangCompiler.ts`, `WgslBoolLowering.ts`, `Program.ts`).
Leave your code as it is:

| Native construct | What web-falcor does |
| --- | --- |
| `Buffer<T>`, `RWBuffer<T>` | Turned into `StructuredBuffer<T>`, `RWStructuredBuffer<T>` (see the caveat in [§3.5](#35-types-64-bit-16-bit-bool-literals)) |
| `bool` members of constant and structured buffers | Lowered to `u32` in WGSL, which forbids `bool` in host-shareable memory |
| `createDynamicObject<I, T>(id, data)` and type conformances | Turned into a generated `switch` (slang-wasm has no conformance API) |
| `RWTexture` that is only written | Emitted as a write-only storage texture |
| Storage texture format | `ComputePass` retargets the WGSL format to the bound texture's format, within the same float/uint/sint class |
| Wave ops in divergent control flow | Allowed: `diagnostic(off, subgroup_uniformity)` is added |
| `float16_t*` without the `shader-f16` feature; `uint16_t`, `int16_t` | Demoted to 32-bit types; `WEBFALCOR_NO_F16=1` is defined when this happens |
| `UINT32_MAX` and other Falcor constants that become i64 literals | Falcor's `MathConstants.slangh` is overridden |
| `#include` and `import` paths | Resolved against the Falcor shader roots, as natively |
| Upstream shaders that needed changes | Replaced by their override, which the host code still loads by the upstream path |

None of this applies to the build-time `slangc` path in `packages/slang-compiler`, which emits raw WGSL. Test
shaders in the browser, not only with `slangc`.

There is no general "web target" define. `WEBFALCOR_NO_F16` is the only one the platform sets. If you keep one
source for both native and web, add your own define from the TS side.

### 3.1 Ray tracing: DXR pipelines become compute

WebGPU has no ray-tracing API: no RT pipelines, no `RayQuery` and no acceleration-structure binding. web-falcor
traces with a software BVH behind the same `SceneRayQuery` interface that Falcor's inline-RT code uses.

- **Passes that already use `SceneRayQuery`** (`import Scene.RaytracingInline;`) port as they are. This includes
  the compute variants of Falcor's own passes.
- **`RtProgram` passes** become one compute kernel:
  - The raygen shader becomes a `[numthreads(16, 16, 1)]` compute `main`.
  - `DispatchRaysIndex()` becomes the dispatch thread ID. `DispatchRaysDimensions()` does not exist; pass the
    frame size in a constant buffer.
  - `TraceRay` plus miss and hit shaders become `query.traceRay` or `query.traceVisibilityRay`, followed by inline
    code: the miss branch when the hit is invalid, otherwise the closest-hit code.
  - Any-hit shaders become a hit filter, and hit-group selection becomes a `switch` on the geometry type.
  - Recursion becomes a loop. Falcor's own passes already loop.
  - SER (`HitObject`, `ReorderThread`) must be off (`USE_SER 0`).

```slang
// native
TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, 1, rayTypeCount, 1, ray, rayData);
// web-falcor
SceneRayQuery<1> query;   // template argument: UseAlphaTest
bool visible = query.traceVisibilityRay(Ray(origin, dir, 0.f, dist), RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff);
float hitT; const HitInfo hit = query.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff);
if (hit.isValid()) handleHit(hit, rayData); else handleMiss(rayData);
```

The [MinimalPathTracer override](../packages/falcor/shaders/WebFalcor/Overrides/RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang)
is the smallest complete conversion. [WhittedRayTracer](../packages/falcor/shaders/WebFalcor/Overrides/RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang)
and [TracePass](../packages/falcor/shaders/WebFalcor/Overrides/RenderPasses/PathTracer/TracePass.rt.slang) show
the larger ones. Expect it to run several times slower than DXR on the same GPU.

### 3.2 Binding limits

WebGPU fixes the number of bindings per shader stage. Native D3D12 and Vulkan have no comparable caps.

| Limit | Value | Notes |
| --- | --- | --- |
| Storage buffers per stage | **16** (Chromium's cap); **10** on Chrome/Edge for macOS | Includes every `StructuredBuffer`, `RWStructuredBuffer` and `ByteAddressBuffer`, read-only or not, and all those `gScene` brings in |
| Storage textures per stage | 8 on typical desktop GPUs (WebGPU's default is 4) | Every `RWTexture*` |
| Bind groups | **4** | Group 0 holds the loose globals. Every `ParameterBlock<T>` takes its own group |
| Resource arrays | none | `Texture2D t[N]`, bindless, and arrays of structs containing resources are all rejected |

Scene-bound passes already use most of the storage-buffer budget: PathTracer with RTXDI sits at exactly 16.
`ComputePass` checks these limits before it dispatches and throws an error naming the pass, the count and the
limit. `RasterPass` and `FullScreenPass` don't check, and over the limit Dawn silently drops the command buffer
(black frame).

Fixes:

- **Too many storage buffers:**
  - Merge two buffers into one and pass the second's offset.
  - Merge per-element records into one struct (array of structs).
  - Move a small read-only table into a one-row texture behind a `__subscript` wrapper. Sampled textures have a
    separate, larger budget:

    ```slang
    struct TableTexture {                            // replaces StructuredBuffer<float4>
        Texture2D<float4> tex;
        __subscript(uint i) -> float4 { get { return tex[uint2(i, 0)]; } }
    };
    ```
- **Too many storage textures:** split the outputs over several dispatches (GBufferRT re-traces once per group of
  8), or write into a buffer and copy into the textures afterwards.
- **A fifth bind group:** declare `ConstantBuffer<T>` instead of `ParameterBlock<T>`, or move the loose globals
  into an existing block.
- **Resource arrays:** unroll them into N named bindings and select with a `switch`, or pack the resources into a
  single one (a `Texture2DArray`, an atlas, or one buffer with offsets). For example,
  [MaterialSystem.slang](../packages/falcor/shaders/WebFalcor/Overrides/Scene/Material/MaterialSystem.slang)
  turns Falcor's bindless material textures into 16 `Texture2DArray` buckets.

### 3.3 Storage textures and formats

- **`read_write` storage textures exist only for `r32float`, `r32uint` and `r32sint`.** Other formats can be
  written but not read in the same shader. A texture can't be sampled and written in one pass either. For
  read-modify-write state (accumulation, history), use one of these:
  - a `RWStructuredBuffer<float4>` indexed by pixel ([Accumulate.cs.slang](../packages/falcor/shaders/WebFalcor/Overrides/RenderPasses/AccumulatePass/Accumulate.cs.slang));
  - a ping-pong pair: read last frame's texture, write a separate one;
  - a local variable, written once at the end.
- **A storage binding has one static format.** `ComputePass` retargets it to the bound texture's format.
  Elsewhere, or when the type class differs, annotate it: `[format("rgba16f")] RWTexture2D<float4> gOut;`.
- **Formats that can be storage textures:** `r32*`, `rg32*`, `rgba8` unorm/snorm/int/uint, `rgba16` float/int/uint,
  `rgba32*`, and `bgra8unorm` with an optional feature. **Formats that can't:** `r8`, `r16` (including
  `R16Float`), `rg8`, `rg16`, `rgb10a2`, `rg11b10` and all sRGB formats. Use a wider format for UAV outputs: the
  ports use `RGBA16Float` for `RGB10A2Unorm` and `R32Float` for `R16Float`. The render graph promotes `R8Uint`
  and `R16Uint` UAV fields to `R32Uint` automatically.
- **Formats that don't exist on WebGPU:** `RGB32Float` and other 3-channel textures, and `BGRX8`. Creating a
  texture in one of them throws. 3-channel formats exist only as buffer element types.
- **Formats that need optional GPU features:**
  - render targets: `R11G11B10Float` (`rg11b10ufloat-renderable`);
  - filtering: `R32Float` (`float32-filterable`);
  - 16-bit norm formats (`texture-formats-tier1`);
  - blending on float32 targets (`float32-blendable`).
- **Formats that are never renderable:** snorm formats and `RGB9E5`.
- **Filtering:** if a program samples any texture, every float texture bound to it needs a filterable format.
  Unsampled float textures are bound as unfilterable automatically.
- **Readback:** 24-bit depth can't be read back.
- **Multisampled and 1D textures** have a single mip level, and a 1D texture can't be a render target.
- **No texture atomics.** See [§3.6](#36-atomics).

### 3.4 Samplers and texture operations

- **No border addressing.** `Border` falls back to clamp-to-edge, with no border color. Emulate it in the shader
  (see `sampleBorder` in [SimplePostFX.cs.slang](../packages/falcor/shaders/WebFalcor/Overrides/RenderPasses/SimplePostFX/SimplePostFX.cs.slang)).
- **No min/max reduction samplers.** Creating one throws. For min/max mips, use `generateMips` with min/max.
- **`lodBias`** is not supported.
- **Texel offsets must be compile-time constants.** Slang turns an offset inside an unrolled loop into a runtime
  value, which WGSL rejects. Fold the offset into the coordinate:

  ```slang
  // native
  c += gTex.SampleLevel(gSampler, texC, 0, offset + i * dir);
  // web-falcor
  c += gTex.SampleLevel(gSampler, texC + float2(offset + i * dir) * invDim, 0);
  ```
- **`Texture3D.Load` with an offset** has no WGSL mapping. Add the offset to the coordinate.
- **No explicit registers.** `register(t0)`, `register(u0)`, `register(s0)` and `register(b0)` all become binding
  0 and collide. Remove them; bindings are made by name.
- **`RWTexture2D<unorm float4>`** has no WGSL form. Drop the `unorm` or `snorm`.

### 3.5 Types: 64-bit, 16-bit, `bool`, literals

- **No `double`, `int64_t` or `uint64_t`.** If fp64 reaches the WGSL backend, Slang emits nothing at all, and
  web-falcor reports
  `Slang emitted no entry point (silent WGSL backend failure — check for fp64/unsupported constructs)`.
  - Replace `double` with compensated float arithmetic: Kahan summation, or `fma` for difference-of-products.
  - Emulate 64-bit integers as `uint2(lo, hi)` and carry by hand:

    ```slang
    uint2 u64_add(uint2 a, uint2 b) { uint lo = a.x + b.x; return uint2(lo, a.y + b.y + (lo < a.x ? 1u : 0u)); }
    ```
    [SplitMix64.slang](../packages/falcor/shaders/WebFalcor/Overrides/Utils/Sampling/Pseudorandom/SplitMix64.slang)
    does this and stays bit-identical to native.
- **Unsuffixed large integer literals become i64.** `4294967295` does not fit in `int`, so Slang makes it 64-bit
  and the module fails. Write `0xffffffffu`. Literals above 2^63 are rejected outright.
- **16-bit types without `shader-f16`** (Chromium on Linux, for example, doesn't expose it):
  - `float16_t` becomes `float`, so precision rises.
  - `uint16_t` and `int16_t` always become 32-bit.
  - **Structs shared with the host change size and offsets**, and 16-bit index buffers can't be used. To keep a
    packed layout, store halves in `uint` fields and unpack them with `f16tof32`/`f32tof16` behind `property`
    accessors, as [BasicMaterialData.slang](../packages/falcor/shaders/WebFalcor/Overrides/Scene/Material/BasicMaterialData.slang)
    does.
  - `asfloat16` and `asuint16` are not available.
  - `half`, `int16_t2/4` and f16 matrices are not demoted.
- **`bool` in buffers** is handled for you ([§3.0](#30-what-the-compiler-handles-for-you)), with one exception.
  If a non-buffer struct has a member with the same name as the bool, the bool is left alone and Tint rejects the
  module. Rename one of them, or use `uint`.
- **`Buffer<T>` becomes `StructuredBuffer<T>`**, which changes its meaning:
  - **3-component elements** get a 16-byte stride. `createTypedBuffer` still sizes a `float3`/`RGB32Float`
    buffer at 12 bytes per element, so the data is misread.
  - **Narrow formats lose their conversion.** A `Buffer<float>` over `R16Float`, or an `RG8Snorm` buffer, is read
    as raw 32-bit words.
  - **`GetDimensions` takes different arguments.**

  Use 32-bit, 1-, 2- or 4-component element types, and pack narrower data by hand.

### 3.6 Atomics

WGSL atomics exist only on `atomic<u32>` and `atomic<i32>` elements of storage buffers.

- **`InterlockedAdd` on a `RWByteAddressBuffer` or a plain `uint` buffer:** declare the buffer as
  `RWStructuredBuffer<Atomic<uint>>` and index it by `byteOffset / 4`.
- **Float atomics** (`InterlockedAddF32`, including those in DiffRendering): use a compare-exchange loop on the
  bits:

  ```slang
  uint expected = buf[i].load();
  for (;;) {
      uint original = buf[i].compareExchange(expected, asuint(asfloat(expected) + value));
      if (original == expected) break;
      expected = original;
  }
  ```
- **Atomics on textures:** use a buffer, laid out by pixel.
- **fp16 and 64-bit atomics** don't exist. Use 32-bit atomics instead, or emulate 64-bit values as `uint2`.
- **Append/consume buffers and UAV counters:** replace them with an explicit `Atomic<uint>` counter in its own
  buffer.

### 3.7 Wave and subgroup operations

- **The `subgroups` feature is optional.** Provide a fallback (a groupshared reduction) or a define that compiles
  the wave path out.
- **The subgroup size is not 32.** It is 8 to 64, depending on the GPU. Code that assumes 32 lanes (warp-indexed
  shared memory, `WaveGetLaneIndex() < 32` masks, `uint` ballots) breaks. A shared-memory tree reduction is the
  portable replacement.
- **Unsupported ops:**
  - `WaveActiveCountBits(x)`: use `countbits(WaveActiveBallot(x).x)`.
  - `WaveMatch`: emulate with a loop over `WaveReadLaneFirst` + `WaveActiveBallot`.
  - NVAPI shuffles: use shared memory.
- **A `static groupshared` member inside a struct** is not supported. Declare a module-scope `groupshared` array.

### 3.8 Derivatives and implicit-LOD sampling

- **`ddx`, `ddy` and implicit-LOD `Sample()` are fragment-only.** Everything reachable from compute, which with
  software RT includes all hit shading, must use `SampleLevel` or `SampleGrad`.
- **Implicit derivatives need uniform control flow, even in fragment shaders.** WGSL rejects `Sample()` inside a
  non-uniform branch, and material code is full of them. Take the derivatives once at the top of the fragment
  shader and pass them down:

  ```slang
  let lod = ExplicitGradientTextureSampler(ddx(vsOut.texC), ddy(vsOut.texC));   // was ImplicitLodTextureSampler()
  ```

### 3.9 Raster stage

- **No `SV_PrimitiveID` or `SV_Barycentrics`.** Use vertex pulling: draw non-indexed, compute
  `triangleIndex = SV_VertexID / 3` in the vertex shader, pass it as a `nointerpolation` varying, and build
  barycentrics from per-corner unit weights.
  [GBufferRaster.3d.slang](../packages/falcor/shaders/WebFalcor/Overrides/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang)
  does this.
- **`SV_VertexID` includes the base vertex** on WebGPU; on D3D it doesn't. Subtract the mesh's `vbOffset` before
  indexing per-mesh data.
- **No `[earlydepthstencil]` and no rasterizer-ordered views.** Write extra channels to additional render
  targets.
- **Entry points can't have `out` or `inout` parameters.** Return a struct of varyings instead.
- **Don't create a local instance of a struct with semantics** (`VSOut o;` in a helper): WGSL then gets duplicate
  `@location`s. Build it only as the entry point's return value.
- **The HLSL `linear` qualifier** becomes screen-space linear interpolation in WGSL, not perspective-correct.
  Remove it ([§4](#4-differences-that-compile-but-change-results)).

### 3.10 Slang front-end changes (2024.1.34 → 2026.18.2)

These come from the newer compiler, not from WGSL. Native Falcor would hit them too after a Slang upgrade.

- **Brace and zero initialization is stricter.** These forms fail on structs that have an explicit or synthesized
  `__init`:
  - `T x = {};` and `return {};`
  - nested `{a, {b, c}}`
  - aggregate initialization of a struct that declares its own `__init`
  - `this = {};` inside `__init`, which makes the WGSL backend abort without a message

  Declare the variable, then assign the members:

  ```slang
  // native
  PBRTConductorBSDF bottom = {conductorD, conductorEta, conductorK};
  // web-falcor
  PBRTConductorBSDF bottom; bottom.D = conductorD; bottom.eta = conductorEta; bottom.k = conductorK;
  ```
- **New builtins collide with Falcor's names.** `packUnorm4x8`, `unpackUnorm2x16`, `packSnorm2x16` and similar
  are now builtins. Falcor's versions are renamed `falcor_packUnorm4x8` and so on; rename your own the same way.
- **`precise` reaches WGSL as invalid syntax.** Remove it. WGSL can't forbid FMA contraction, so expect small
  rounding differences.
- **Interface-typed values chosen at run time crash the WGSL backend.** For example, an `ITextureSampler` local
  assigned in two branches stops with
  `E99997: Unhandled info type in analyzeExtractExistentialType`. slang-wasm swallows that error, so you see an
  empty-emission error instead. Make the code after the branch generic and call it from each branch with a
  concrete type:

  ```slang
  void shadeHit<L : ITextureSampler>(const L lod, ...);
  if (useRayCones) shadeHit(ExplicitRayConesLodTextureSampler(l), ...);
  else             shadeHit(ExplicitGradientTextureSampler(dx, dy), ...);
  ```
- **Existential packing is larger on WGSL.** If you define a material or another type stored in an interface
  slot, it may need a larger `[anyValueSize]`. `IMaterial` went from 128 to 256.
- **Resources can't be passed as function arguments.** Pass the loaded values instead.
- **Autodiff (`[Differentiable]`, `fwd_diff`, `bwd_diff`):**
  - A derivative that is dropped implicitly now needs an explicit `detach()` (error E41031).
  - Inside a region reached through a nested `fwd_diff`, three patterns crash 2026.18: differentiable
    `out`/`inout` parameters (including `[mutating]` methods), `X.Differential d = {}`, and `no_diff` parameters
    that feed builtins such as `dot`. Return results instead of writing to `out` parameters, and `detach()` at the
    call site instead of using `no_diff`. Minimal repros are in [slang-repros/](slang-repros/).
  - Backward mode crashes in Slang 2026.7 and later for some kernels. Such programs can be compiled with a pinned
    older slang-wasm: set `ProgramDesc.slangRuntime` and load the runtime with
    `programManager.loadSlangRuntime(url)`.
  - Compile only the active `DIFF_MODE` branch (`#if DIFF_MODE == n`). Compiling them all instantiates every
    derivative and can exhaust the slang-wasm heap.
- **Comparisons inside `select()`.** `select(x < float2(...), a, b)` can be emitted as WGSL that parses the `<`
  as a template list, which Tint rejects. Compare into a named local first.

## 4. Differences that compile but change results

Nothing reports these, so check for them when an image doesn't match native:

| Difference | Effect | What to do |
| --- | --- | --- |
| HLSL `linear` interpolation → WGSL screen-space `linear` | `posW` and normals skew on tilted surfaces | Remove the qualifier |
| `SV_VertexID` includes the base vertex | Wrong per-vertex data on any mesh after the first | Subtract `vbOffset` |
| More than 16 storage buffers in a raster or full-screen pass | Black frame; one validation message in the console | Merge buffers ([§3.2](#32-binding-limits)) |
| Buffer sized with C++ `sizeof` | Black frame | Use WGSL layout sizes ([§2.3](#23-binding-values-from-typescript)) |
| NaN | WGSL compilers may assume no NaNs: `isnan()` guards can be folded away, and NaN comparisons are undefined (HLSL `NaN != x` is true) | Don't produce NaNs (guard 0/0 and `normalize(0)`); test `isnan` explicitly where NaN semantics matter |
| fp16 demotion and 16-bit integer widening | Higher precision; host-visible struct layouts change | Pack halves into `uint` ([§3.5](#35-types-64-bit-16-bit-bool-literals)) |
| fp64 replaced by compensated float | Last-bit differences | Accept, or emulate more precisely |
| Shared-memory or atomic reductions instead of wave ops | Float sums differ in the last bits (order changes) | Use image-test tolerances |
| No FMA control (`precise` removed) | Rounding differences | Use tolerances |
| Software BVH: 32-entry traversal stack, `instanceInclusionMask` ignored | Very deep BVHs can miss hits; masked instances are still hit | Keep scenes within the depth; filter instances in the hit code |
| `createDynamicObject` with an unregistered type ID | Falls back to the lowest-ID type (native: undefined behaviour) | Register every type |
| `Border` sampler | Clamps to the edge instead | Emulate the border in the shader |
| Graph field with `Unknown` format | Allocated as `RGBA32Float` (native uses the swapchain format, usually `BGRA8UnormSrgb`) | Give outputs an explicit format |
| Hashed alpha test | Uses a constant derivative instead of the screen-space one | — |
| Material `samplerID` | Ignored; all material textures share one sampler | — |

[parity-matrix.md §9](parity-matrix.md) lists the divergences of each upstream pass.

## 5. Browser and GPU differences

- **Storage-buffer limit:** Chrome and Edge on Windows and Linux, and Safari, give 16 storage buffers per stage.
  Chrome and Edge on macOS give 10, too few for the path tracer and the G-buffer passes. Use Safari on a Mac.
- **`shader-f16`** depends on the browser and platform, independently of what the driver supports. Assume it is
  missing ([§3.5](#35-types-64-bit-16-bit-bool-literals)).
- **`subgroups`** is optional, and the subgroup size depends on the GPU (8 to 64).
- **Optional texture features** (`float32-filterable`, `float32-blendable`, `rg11b10ufloat-renderable`,
  `texture-formats-tier1`, `bgra8unorm-storage`) are enabled when the adapter has them. The console lists the
  enabled features at startup (`Created WebGPU device (features: …)`), so check it when a format fails on another
  machine.
- **Workgroups:** web-falcor requests the adapter's maximum invocations per workgroup (1024 on desktop GPUs), but
  not a larger workgroup Z size. Dispatches are not split for you: keep each dimension within 65535 workgroups.
- **GPU timestamps** need the `timestamp-query` feature; without it the profiler records nothing. Timestamps
  inside a pass also need a Chromium-only feature.
- **Headless browsers** often fall back to SwiftShader (a CPU renderer). For automated runs, use a headed
  browser under Xvfb, as the GPU test suite does.

## 6. Reading the errors

| Message | Cause | See |
| --- | --- | --- |
| `Slang emitted no entry point (silent WGSL backend failure …)` | fp64, or an interface value chosen at run time | §3.5, §3.10 |
| `<pass> binds N storage buffers, but this GPU allows M per shader stage` | Too many storage buffers or textures | §3.2 |
| `The number of storage buffers (17) … exceeds the maximum` (WebGPU validation) | The same, in a raster or full-screen pass | §3.2 |
| `Shader parameter 'X' is not bound` | An unbound resource | §2.2 |
| `ParameterBlock: WGSL binding 'X' … has no reflection match` | A renamed or mangled field; often a `__` in the name | Rename it |
| `ResourceFormat X has no WebGPU texture format` | A 3-channel texture or `BGRX8` | §3.3 |
| `Expected N components for 'X'` | A vector or matrix of the wrong length, or a class instance without `.toArray()` | §2.3 |
| `Setting '…' uniforms not implemented yet` | A whole struct assigned to a `ShaderVar` | §2.3 |
| `cannot open file` | The shader is not in `shader-file-list.json` (run `npm run shaders:list`), or a plugin file that isn't registered | §2.4 |
| `Entry point 'X' not found` | Wrong entry-point name or module | — |
| Tint errors about `bool`, `read_write` access or `@interpolate` | A construct the WGSL rewrites don't cover | §3.5, §3.3, §3.9 |
| `E41031` | A derivative is dropped implicitly | §3.10 |
| Black frame with no error | Over the storage-buffer limit, or a buffer that is too small | §4 |

To see errors that slang-wasm hides, compile the shader with a native `slangc` of the same version
(2026.18.2; the [README](../README.md#full-setup-develop--run-the-gpuoracle-tests) shows how to install it):
`slangc -target wgsl -stage compute -entry main MyPass.cs.slang`. The CLI prints the diagnostics that slang-wasm
swallows. It applies none of the rewrites in §3.0, so some of its errors don't occur in the browser.

## 7. Checking a port

Render the same script and scene in native Mogwai and in the browser and compare the captured frames. The GPU test
suite does this for every upstream pass ([testing.md](testing.md)).

- Compare with a tolerance, not bit for bit. §4 lists the expected sources of difference.
- If an image is black, look for a WebGPU validation message in the console before you debug the shader
  (§6, last row).
