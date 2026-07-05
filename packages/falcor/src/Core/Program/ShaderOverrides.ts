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
};
