/**
 * WebFalcor shader overrides (docs §4.3).
 *
 * Upstream shader files that use features WGSL cannot express are substituted
 * with web-owned implementations that keep the exact interface (entry points,
 * defines, cbuffers, binding names). Host code keeps referencing the upstream
 * path; the ProgramManager resolves content through this map. Each override
 * file documents its diff vs upstream.
 */

export const kShaderOverrides: Readonly<Record<string, string>> = {
    // RWBuffer texel buffers + warp-size-32 wave reduction -> structured buffers + portable shared-memory reduction.
    "RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang": "WebFalcor/Overrides/RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang",
    "Utils/Debug/PixelDebug.slang": "WebFalcor/Overrides/Utils/Debug/PixelDebug.slang",
    "Utils/UI/TextRenderer.3d.slang": "WebFalcor/Overrides/Utils/UI/TextRenderer.3d.slang",
    "Rendering/Materials/MERLMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/MERLMaterial.slang",
    "Rendering/Materials/MERLMaterialInstance.slang": "WebFalcor/Overrides/Rendering/Materials/MERLMaterialInstance.slang",
    "Rendering/Materials/MERLMixMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/MERLMixMaterial.slang",
    "Rendering/Materials/MERLMixMaterialInstance.slang": "WebFalcor/Overrides/Rendering/Materials/MERLMixMaterialInstance.slang",
    "Scene/Material/RGLMaterialData.slang": "WebFalcor/Overrides/Scene/Material/RGLMaterialData.slang",
    "Rendering/Materials/RGLCommon.slang": "WebFalcor/Overrides/Rendering/Materials/RGLCommon.slang",
    "Rendering/Materials/RGLMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/RGLMaterial.slang",
    "Utils/Algorithm/BitonicSort.cs.slang": "WebFalcor/Overrides/Utils/Algorithm/BitonicSort.cs.slang",
    "Utils/Algorithm/ParallelReduction.cs.slang": "WebFalcor/Overrides/Utils/Algorithm/ParallelReduction.cs.slang",
    // ByteAddressBuffer atomics -> structured buffers with Atomic<uint> elements.
    "Utils/Algorithm/PrefixSum.cs.slang": "WebFalcor/Overrides/Utils/Algorithm/PrefixSum.cs.slang",
    // Read-write rgba32float storage textures (WGSL allows r32* only) -> structured buffers; Double mode omitted (no fp64/i64).
    "RenderPasses/AccumulatePass/Accumulate.cs.slang": "WebFalcor/Overrides/RenderPasses/AccumulatePass/Accumulate.cs.slang",
    // pack*/unpack* renamed (collide with builtins added to Slang post-2024); 'this = {};' replaced.
    "Utils/Math/FormatConversion.slang": "WebFalcor/Overrides/Utils/Math/FormatConversion.slang",
    "Utils/Math/PackedFormats.slang": "WebFalcor/Overrides/Utils/Math/PackedFormats.slang",
    "Utils/Geometry/IntersectionHelpers.slang": "WebFalcor/Overrides/Utils/Geometry/IntersectionHelpers.slang",
    "RenderPasses/ErrorMeasurePass/ErrorMeasurer.cs.slang": "WebFalcor/Overrides/RenderPasses/ErrorMeasurePass/ErrorMeasurer.cs.slang",
    // 'ShadingData sd = {}' has no zero-arg initializer under Slang 2026.12.2 -> loadShadingData inlined;
    // 'this = {};' in PixelData.__init aborts the WGSL backend -> explicit field init.
    "RenderPasses/PixelInspectorPass/PixelInspector.cs.slang": "WebFalcor/Overrides/RenderPasses/PixelInspectorPass/PixelInspector.cs.slang",
    "RenderPasses/PixelInspectorPass/PixelInspectorData.slang": "WebFalcor/Overrides/RenderPasses/PixelInspectorPass/PixelInspectorData.slang",
    "Scene/HitInfo.slang": "WebFalcor/Overrides/Scene/HitInfo.slang",
    // WGSL has no binding arrays (docs §6.2): packed Texture2DArray material
    // textures, single sampler/buffer/3D bindings, single grid/SDF instances,
    // single-buffer Split*Buffers.
    "Scene/Material/BasicMaterialData.slang": "WebFalcor/Overrides/Scene/Material/BasicMaterialData.slang",
    "Scene/Material/AlphaTest.slang": "WebFalcor/Overrides/Scene/Material/AlphaTest.slang",
    "Scene/Material/MaterialData.slang": "WebFalcor/Overrides/Scene/Material/MaterialData.slang",
    "Scene/Material/MaterialSystem.slang": "WebFalcor/Overrides/Scene/Material/MaterialSystem.slang",
    "Scene/Material/TextureSampler.slang": "WebFalcor/Overrides/Scene/Material/TextureSampler.slang",
    "Scene/Scene.slang": "WebFalcor/Overrides/Scene/Scene.slang",
    "Scene/Displacement/DisplacementMapping.slang": "WebFalcor/Overrides/Scene/Displacement/DisplacementMapping.slang",
    "Scene/SceneTypes.slang": "WebFalcor/Overrides/Scene/SceneTypes.slang",
    // Newer-Slang '= {}' / brace-init fixes + WGSL raster-path gaps.
    "Scene/Material/MaterialFactory.slang": "WebFalcor/Overrides/Scene/Material/MaterialFactory.slang",
    "Scene/Raster.slang": "WebFalcor/Overrides/Scene/Raster.slang",
    // Software BVH traversal replaces DXR 1.1 RayQuery (docs §5).
    "Scene/RaytracingInline.slang": "WebFalcor/Overrides/Scene/RaytracingInline.slang",
    "Rendering/Materials/StandardMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/StandardMaterial.slang",
    "Rendering/Materials/IMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/IMaterial.slang",
    // SV_PrimitiveID/SV_Barycentrics/[earlydepthstencil] absent from WGSL; static material dispatch.
    "RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang": "WebFalcor/Overrides/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang",
    // RT pipeline -> compute megakernel over SceneRayQuery (docs §5).
    "RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang": "WebFalcor/Overrides/RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang",
    // RT pipeline + shader table -> compute kernel with explicit hit-group selection.
    "RenderPasses/TestPasses/TestRtProgram.rt.slang": "WebFalcor/Overrides/RenderPasses/TestPasses/TestRtProgram.rt.slang",
    // Full PathTracer: '= {}' fixes + RT pipeline -> compute megakernel over SceneRayQuery.
    "RenderPasses/PathTracer/GeneratePaths.cs.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/GeneratePaths.cs.slang",
    "RenderPasses/PathTracer/PathTracer.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/PathTracer.slang",
    "RenderPasses/PathTracer/TracePass.rt.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/TracePass.rt.slang",
    // NRD outputs as views over one pixel buffer + one sample buffer (8 storage textures per stage).
    "RenderPasses/PathTracer/ResolvePass.cs.slang": "WebFalcor/Overrides/RenderPasses/PathTracer/ResolvePass.cs.slang",
    "RenderPasses/Shared/Denoising/NRDBuffers.slang": "WebFalcor/Overrides/RenderPasses/Shared/Denoising/NRDBuffers.slang",
    // Newer-Slang nested brace-init fix.
    "Rendering/Materials/PBRT/PBRTConductorMaterial.slang": "WebFalcor/Overrides/Rendering/Materials/PBRT/PBRTConductorMaterial.slang",
    // 64-bit traversal bitmask kept as uint2 (no 64-bit ints in WGSL).
    "Rendering/Lights/LightBVHSampler.slang": "WebFalcor/Overrides/Rendering/Lights/LightBVHSampler.slang",
    // WGSL has no 64-bit integers: SplitMix64 state emulated as uint2 (lo, hi), bit-identical sequences.
    "Utils/Sampling/Pseudorandom/SplitMix64.slang": "WebFalcor/Overrides/Utils/Sampling/Pseudorandom/SplitMix64.slang",
    // FalcorTest kernels (transplanted GPU unit tests): no typed buffers / 64-bit ints in WGSL.
    "Tests/Utils/HashUtilsTests.cs.slang": "WebFalcor/Overrides/Tests/Utils/HashUtilsTests.cs.slang",
    "Tests/Sampling/PseudorandomTests.cs.slang": "WebFalcor/Overrides/Tests/Sampling/PseudorandomTests.cs.slang",
    "Tests/Slang/Atomics.cs.slang": "WebFalcor/Overrides/Tests/Slang/Atomics.cs.slang",
    "Tests/Slang/SlangTests.cs.slang": "WebFalcor/Overrides/Tests/Slang/SlangTests.cs.slang",
    // Avoids a 0/0 that upstream masks with isnan() (WGSL may assume no NaNs).
    "Rendering/Materials/HairChiang16.slang": "WebFalcor/Overrides/Rendering/Materials/HairChiang16.slang",
    "Tests/Slang/WaveOps.cs.slang": "WebFalcor/Overrides/Tests/Slang/WaveOps.cs.slang",
    "Tests/Rendering/Materials/MicrofacetTests.cs.slang": "WebFalcor/Overrides/Tests/Rendering/Materials/MicrofacetTests.cs.slang",
    "Tests/Scene/Material/BSDFTests.cs.slang": "WebFalcor/Overrides/Tests/Scene/Material/BSDFTests.cs.slang",
    "Tests/Core/BufferTests.cs.slang": "WebFalcor/Overrides/Tests/Core/BufferTests.cs.slang",
    // Sample apps: SV_PrimitiveID has no core-WGSL equivalent.
    "Samples/MultiSampling/MultiSampling.3d.slang": "WebFalcor/Overrides/Samples/MultiSampling/MultiSampling.3d.slang",
    // HelloDXR: vertex-pulled raster (no SV_PrimitiveID), RT program lowered to compute.
    "Samples/HelloDXR/HelloDXR.3d.slang": "WebFalcor/Overrides/Samples/HelloDXR/HelloDXR.3d.slang",
    "Samples/HelloDXR/HelloDXR.rt.slang": "WebFalcor/Overrides/Samples/HelloDXR/HelloDXR.rt.slang",
    "Tests/Core/RootBufferParamBlockTests.cs.slang": "WebFalcor/Overrides/Tests/Core/RootBufferParamBlockTests.cs.slang",
    "Tests/Core/ParamBlockDefinition.slang": "WebFalcor/Overrides/Tests/Core/ParamBlockDefinition.slang",
    // Atomic<uint> histogram; WaveMatch emulated (no WGSL builtin).
    "Utils/Debug/WarpProfiler.slang": "WebFalcor/Overrides/Utils/Debug/WarpProfiler.slang",
    "Utils/Sampling/UniformSampleGenerator.slang": "WebFalcor/Overrides/Utils/Sampling/UniformSampleGenerator.slang",
    // Typed buffers (Buffer<T>) don't exist in WGSL -> structured buffer.
    // SampleLevel offsets folded into the coordinate (WGSL needs const-expression offsets).
    "RenderPasses/Utils/GaussianBlur/GaussianBlur.ps.slang": "WebFalcor/Overrides/RenderPasses/Utils/GaussianBlur/GaussianBlur.ps.slang",
    // Write-only gDst + gDstPrev ping-pong, uint gInPlace, border sampling emulated (no border mode in WebGPU).
    "RenderPasses/SimplePostFX/SimplePostFX.cs.slang": "WebFalcor/Overrides/RenderPasses/SimplePostFX/SimplePostFX.cs.slang",
    // Aggregate init of a struct with an explicit __init is rejected by Slang 2026.12.2 -> member-wise.
    "Rendering/Materials/PBRT/PBRTCoatedConductorMaterialInstance.slang":
        "WebFalcor/Overrides/Rendering/Materials/PBRT/PBRTCoatedConductorMaterialInstance.slang",
    // RT pipeline -> compute megakernel over SceneRayQuery; existential lod
    // samplers restructured into generic helpers.
    "RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang": "WebFalcor/Overrides/RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang",
    // WARDiffPathTracer: compute kernel + reshaping around Slang 2026.18 nested-autodiff crashes.
    "RenderPasses/WARDiffPathTracer/WARDiffPathTracer.rt.slang": "WebFalcor/Overrides/RenderPasses/WARDiffPathTracer/WARDiffPathTracer.rt.slang",
    "RenderPasses/WARDiffPathTracer/PTUtils.slang": "WebFalcor/Overrides/RenderPasses/WARDiffPathTracer/PTUtils.slang",
    "RenderPasses/WARDiffPathTracer/WarpedAreaReparam.slang": "WebFalcor/Overrides/RenderPasses/WARDiffPathTracer/WarpedAreaReparam.slang",
    "DiffRendering/DiffSceneIO.slang": "WebFalcor/Overrides/DiffRendering/DiffSceneIO.slang",
    "DiffRendering/DiffSceneQuery.slang": "WebFalcor/Overrides/DiffRendering/DiffSceneQuery.slang",
    "DiffRendering/DiffDebugParams.slang": "WebFalcor/Overrides/DiffRendering/DiffDebugParams.slang",
    "DiffRendering/InverseOptimizationParams.slang": "WebFalcor/Overrides/DiffRendering/InverseOptimizationParams.slang",
    "DiffRendering/SceneGradients.slang": "WebFalcor/Overrides/DiffRendering/SceneGradients.slang",
    // Write-only storage textures (WGSL r32*-only read_write rule).
    "RenderPasses/SceneDebugger/SceneDebugger.cs.slang": "WebFalcor/Overrides/RenderPasses/SceneDebugger/SceneDebugger.cs.slang",
    "RenderPasses/BSDFViewer/BSDFViewer.cs.slang": "WebFalcor/Overrides/RenderPasses/BSDFViewer/BSDFViewer.cs.slang",
    // WGSL has no binding arrays / storage-texture atomics -> per-type atomic buffers.
    "Rendering/Utils/PixelStats.slang": "WebFalcor/Overrides/Rendering/Utils/PixelStats.slang",
    // Warp-size-32 wave reduction -> portable shared-memory reduction; no `= {}` ShadingData init.
    "Rendering/Materials/BSDFIntegrator.cs.slang": "WebFalcor/Overrides/Rendering/Materials/BSDFIntegrator.cs.slang",
    // RTXDI: no texel buffers in WGSL (structured swap), boiling filter
    // compiled out (WaveActiveCountBits unmapped), brace-init fixes.
    "Rendering/RTXDI/RTXDI.slang": "WebFalcor/Overrides/Rendering/RTXDI/RTXDI.slang",
    "Rendering/RTXDI/RTXDIApplicationBridge.slangh": "WebFalcor/Overrides/Rendering/RTXDI/RTXDIApplicationBridge.slangh",
    "Rendering/RTXDI/SurfaceData.slang": "WebFalcor/Overrides/Rendering/RTXDI/SurfaceData.slang",
    "Rendering/RTXDI/LightUpdater.cs.slang": "WebFalcor/Overrides/Rendering/RTXDI/LightUpdater.cs.slang",
    "RenderPasses/RTXDIPass/LoadShadingData.slang": "WebFalcor/Overrides/RenderPasses/RTXDIPass/LoadShadingData.slang",
    // Outputs moved into the FinalShading block (max 4 bind groups) + write-only.
    "RenderPasses/RTXDIPass/FinalShading.cs.slang": "WebFalcor/Overrides/RenderPasses/RTXDIPass/FinalShading.cs.slang",
    // Per-LOD Texture3D binding array -> one R8Snorm atlas stacked along Z.
    "Scene/SDFs/NormalizedDenseSDFGrid/NDSDFGrid.slang": "WebFalcor/Overrides/Scene/SDFs/NormalizedDenseSDFGrid/NDSDFGrid.slang",
    // brickID != UINT32_MAX promotes to i64 (unsupported in WGSL) -> 0xffffffffu; packed multi-grid indirection.
    "Scene/SDFs/SparseBrickSet/SDFSBS.slang": "WebFalcor/Overrides/Scene/SDFs/SparseBrickSet/SDFSBS.slang",
    // Per-voxel AABBs moved into the shared SDF BVH buffer (16-storage-buffer cap).
    "Scene/SDFs/SparseVoxelSet/SDFSVS.slang": "WebFalcor/Overrides/Scene/SDFs/SparseVoxelSet/SDFSVS.slang",
    // decodeHit avoids decodeLocation's >2^63 Morton unshift masks (Tint abstract-int).
    "Scene/SDFs/SparseVoxelOctree/SDFSVO.slang": "WebFalcor/Overrides/Scene/SDFs/SparseVoxelOctree/SDFSVO.slang",
    // `SurfaceData data = {}` -> plain declaration (synthesized __init).
    "RenderPasses/BSDFOptimizer/BSDFViewer.cs.slang": "WebFalcor/Overrides/RenderPasses/BSDFOptimizer/BSDFViewer.cs.slang",
    "RenderPasses/BSDFOptimizer/BSDFOptimizer.cs.slang": "WebFalcor/Overrides/RenderPasses/BSDFOptimizer/BSDFOptimizer.cs.slang",
    // grads as Atomic<uint> with a compare-exchange float add (no float atomics in WGSL).
    "DiffRendering/AggregateGradients.cs.slang": "WebFalcor/Overrides/DiffRendering/AggregateGradients.cs.slang",
    // UINT32_MAX as 0xffffffffu (the un-suffixed literal is i64, which WGSL lacks).
    "Utils/Math/MathConstants.slangh": "WebFalcor/Overrides/Utils/Math/MathConstants.slangh",
};
