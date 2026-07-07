/**
 * WebFalcor shader overrides (DESIGN.md §4.3).
 *
 * Upstream shader files that use features WGSL cannot express are substituted
 * with web-owned implementations that keep the exact interface (entry points,
 * defines, cbuffers, binding names). Host code keeps referencing the upstream
 * path; the ProgramManager resolves content through this map. Each override
 * file documents its diff vs upstream.
 */

export const kShaderOverrides: Readonly<Record<string, string>> = {
    // RWBuffer texel buffers + warp-size-32 wave reduction -> structured buffers + portable shared-memory reduction.
    "Utils/Algorithm/ParallelReduction.cs.slang": "WebFalcor/Overrides/Utils/Algorithm/ParallelReduction.cs.slang",
    // ByteAddressBuffer atomics -> structured buffers with Atomic<uint> elements.
    "Utils/Algorithm/PrefixSum.cs.slang": "WebFalcor/Overrides/Utils/Algorithm/PrefixSum.cs.slang",
    // Read-write rgba32float storage textures (WGSL allows r32* only) -> structured buffers; Double mode omitted (no fp64/i64).
    "RenderPasses/AccumulatePass/Accumulate.cs.slang": "WebFalcor/Overrides/RenderPasses/AccumulatePass/Accumulate.cs.slang",
    // pack*/unpack* renamed (collide with builtins added to Slang post-2024); 'this = {};' replaced.
    "Utils/Math/FormatConversion.slang": "WebFalcor/Overrides/Utils/Math/FormatConversion.slang",
    "Utils/Math/PackedFormats.slang": "WebFalcor/Overrides/Utils/Math/PackedFormats.slang",
    "Scene/HitInfo.slang": "WebFalcor/Overrides/Scene/HitInfo.slang",
    // WGSL has no binding arrays (DESIGN.md §6.2): packed Texture2DArray material
    // textures, single sampler/buffer/3D bindings, single grid/SDF instances,
    // single-buffer Split*Buffers.
    "Scene/Material/BasicMaterialData.slang": "WebFalcor/Overrides/Scene/Material/BasicMaterialData.slang",
    "Scene/Material/AlphaTest.slang": "WebFalcor/Overrides/Scene/Material/AlphaTest.slang",
    "Scene/Material/MaterialData.slang": "WebFalcor/Overrides/Scene/Material/MaterialData.slang",
    "Scene/Material/MaterialSystem.slang": "WebFalcor/Overrides/Scene/Material/MaterialSystem.slang",
    "Scene/Material/TextureSampler.slang": "WebFalcor/Overrides/Scene/Material/TextureSampler.slang",
    "Scene/Scene.slang": "WebFalcor/Overrides/Scene/Scene.slang",
    "Scene/SceneTypes.slang": "WebFalcor/Overrides/Scene/SceneTypes.slang",
    // Newer-Slang '= {}' / brace-init fixes + WGSL raster-path gaps.
    "Scene/Material/MaterialFactory.slang": "WebFalcor/Overrides/Scene/Material/MaterialFactory.slang",
    "Scene/Raster.slang": "WebFalcor/Overrides/Scene/Raster.slang",
    // Software BVH traversal replaces DXR 1.1 RayQuery (DESIGN.md §5).
    "Scene/RaytracingInline.slang": "WebFalcor/Overrides/Scene/RaytracingInline.slang",
    "Rendering/Materials/StandardMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/StandardMaterial.slang",
    "Rendering/Materials/IMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/IMaterial.slang",
    // SV_PrimitiveID/SV_Barycentrics/[earlydepthstencil] absent from WGSL; static material dispatch.
    "RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang": "WebFalcor/Overrides/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang",
    // Write-only storage textures (WGSL r32*-only read_write rule).
    "RenderPasses/GBuffer/VBuffer/VBufferRT.slang": "WebFalcor/Overrides/RenderPasses/GBuffer/VBuffer/VBufferRT.slang",
    // Same + rgb10a2unorm storage unsupported in WGSL (normWRoughnessMaterialID -> rgba16f).
    "RenderPasses/GBuffer/GBuffer/GBufferRT.slang": "WebFalcor/Overrides/RenderPasses/GBuffer/GBuffer/GBufferRT.slang",
    // RT pipeline -> compute megakernel over SceneRayQuery (DESIGN.md §5).
    "RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang": "WebFalcor/Overrides/RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang",
    // Full PathTracer: '= {}' fixes + RT pipeline -> compute megakernel over SceneRayQuery.
    "RenderPasses/PathTracer/GeneratePaths.cs.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/GeneratePaths.cs.slang",
    "RenderPasses/PathTracer/PathTracer.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/PathTracer.slang",
    "RenderPasses/PathTracer/TracePass.rt.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/TracePass.rt.slang",
    "RenderPasses/PathTracer/ResolvePass.cs.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/ResolvePass.cs.slang",
    // Newer-Slang nested brace-init fix.
    "Rendering/Materials/PBRT/PBRTConductorMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/PBRT/PBRTConductorMaterial.slang",
    // Typed buffers (Buffer<T>) don't exist in WGSL -> structured buffer.
    "Rendering/Lights/EmissivePowerSampler.slang": "WebFalcor/Overrides/Rendering/Lights/EmissivePowerSampler.slang",
    // 64-bit traversal bitmask kept as uint2 (no 64-bit ints in WGSL).
    "Rendering/Lights/LightBVHSampler.slang": "WebFalcor/Overrides/Rendering/Lights/LightBVHSampler.slang",
    // WGSL has no 64-bit integers: SplitMix64 state emulated as uint2 (lo, hi), bit-identical sequences.
    "Utils/Sampling/Pseudorandom/SplitMix64.slang": "WebFalcor/Overrides/Utils/Sampling/Pseudorandom/SplitMix64.slang",
    "Utils/Sampling/UniformSampleGenerator.slang": "WebFalcor/Overrides/Utils/Sampling/UniformSampleGenerator.slang",
    // Write-only storage textures (WGSL r32*-only read_write rule).
    "RenderPasses/Utils/Composite/Composite.cs.slang": "WebFalcor/Overrides/RenderPasses/Utils/Composite/Composite.cs.slang",
    "RenderPasses/Utils/CrossFade/CrossFade.cs.slang": "WebFalcor/Overrides/RenderPasses/Utils/CrossFade/CrossFade.cs.slang",
    // Typed buffers (Buffer<T>) don't exist in WGSL -> structured buffer.
    "RenderPasses/Utils/GaussianBlur/GaussianBlur.ps.slang": "WebFalcor/Overrides/RenderPasses/Utils/GaussianBlur/GaussianBlur.ps.slang",
    // bool in uniform address space is non-host-shareable in WGSL -> uint.
    "RenderPasses/DebugPasses/Comparison.ps.slang": "WebFalcor/Overrides/RenderPasses/DebugPasses/Comparison.ps.slang",
    // Write-only storage textures (WGSL r32*-only read_write rule).
    "RenderPasses/ModulateIllumination/ModulateIllumination.cs.slang": "WebFalcor/Overrides/RenderPasses/ModulateIllumination/ModulateIllumination.cs.slang",
    // Write-only gDst + gDstPrev ping-pong, uint gInPlace, border sampling emulated (no border mode in WebGPU).
    "RenderPasses/SimplePostFX/SimplePostFX.cs.slang": "WebFalcor/Overrides/RenderPasses/SimplePostFX/SimplePostFX.cs.slang",
    // Write-only outputs + uniform bools -> uint.
    "RenderPasses/FLIPPass/FLIPPass.cs.slang": "WebFalcor/Overrides/RenderPasses/FLIPPass/FLIPPass.cs.slang",
    // RT pipeline -> compute megakernel over SceneRayQuery; existential lod
    // samplers restructured into generic helpers.
    "RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang": "WebFalcor/Overrides/RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang",
    // Write-only storage textures (WGSL r32*-only read_write rule).
    "RenderPasses/SceneDebugger/SceneDebugger.cs.slang": "WebFalcor/Overrides/RenderPasses/SceneDebugger/SceneDebugger.cs.slang",
    "RenderPasses/BSDFViewer/BSDFViewer.cs.slang": "WebFalcor/Overrides/RenderPasses/BSDFViewer/BSDFViewer.cs.slang",
    // WGSL has no binding arrays / storage-texture atomics -> per-type atomic buffers.
    "Rendering/Utils/PixelStats.slang": "WebFalcor/Overrides/Rendering/Utils/PixelStats.slang",
    // bool in cbuffer is non-host-shareable in WGSL -> uint flag.
    "RenderPasses/TAA/TAA.ps.slang": "WebFalcor/Overrides/RenderPasses/TAA/TAA.ps.slang",
    // RTXDI: no texel buffers in WGSL (structured swap), boiling filter
    // compiled out (WaveActiveCountBits unmapped), brace-init fixes.
    "Rendering/RTXDI/RTXDI.slang": "WebFalcor/Overrides/Rendering/RTXDI/RTXDI.slang",
    "Rendering/RTXDI/RTXDIApplicationBridge.slangh": "WebFalcor/Overrides/Rendering/RTXDI/RTXDIApplicationBridge.slangh",
    "Rendering/RTXDI/SurfaceData.slang": "WebFalcor/Overrides/Rendering/RTXDI/SurfaceData.slang",
    "Rendering/RTXDI/LightUpdater.cs.slang": "WebFalcor/Overrides/Rendering/RTXDI/LightUpdater.cs.slang",
    "RenderPasses/RTXDIPass/LoadShadingData.slang": "WebFalcor/Overrides/RenderPasses/RTXDIPass/LoadShadingData.slang",
};
