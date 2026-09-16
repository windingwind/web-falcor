/**
 * WebFalcor override of Rendering/Materials/BSDFIntegrator.cs.slang (docs §4.3).
 *
 * Diffs vs upstream:
 * - WaveActiveSum/WaveIsFirstLane (warp-size-32 assumption) -> portable
 *   shared-memory tree reduction (WGSL subgroup size varies by adapter).
 * - `static groupshared` struct member -> module-scope groupshared array.
 * - `ShadingData sd = {}` (rejected by newer Slang) -> default-initialized local.
 * Interface (entry points, parameter block, binding names) is unchanged.
 */
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Sampling.UniformSampleGenerator;

// 1024 integration threads fold into 256 slots, then tree-reduce; the final
// pass (256 threads) reuses the same array.
static const uint kCacheSize = 256;
groupshared float3 gSharedResult[kCacheSize];

/** Portable group sum over kGroupSize threads (value from every thread); result valid in gSharedResult[0]. */
void reduceGroupSum(float3 value, const uint groupThreadIdx, const uint kGroupSize)
{
    if (groupThreadIdx < kCacheSize) gSharedResult[groupThreadIdx] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint base = kCacheSize; base < kGroupSize; base += kCacheSize)
    {
        if (groupThreadIdx >= base && groupThreadIdx < base + kCacheSize) gSharedResult[groupThreadIdx - base] += value;
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint s = kCacheSize / 2; s > 0; s >>= 1)
    {
        if (groupThreadIdx < s) gSharedResult[groupThreadIdx] += gSharedResult[groupThreadIdx + s];
        GroupMemoryBarrierWithGroupSync();
    }
}

/** Utility for integrating an isotropic BSDF over the hemisphere (see upstream for the layout). */
struct BSDFIntegrator
{
    uint2 gridSize;                             ///< Size of integration grid over the hemisphere.
    uint gridCount;                             ///< Integration grid count. One incident direction per grid.
    uint resultCount;                           ///< Number of intermediate results per grid.
    uint materialID;                            ///< Material ID in the scene.

    StructuredBuffer<float> cosThetas;          ///< Cosine theta of incident directions.
    RWStructuredBuffer<float3> results;         ///< Intermediate results. One value per thread group.
    RWStructuredBuffer<float3> finalResults;    ///< Final results. One value per grid.

    static const uint2 kSampleCount = { 8, 8 }; ///< Number of stratified samples per grid cell.

    void executeIntegration(const uint2 threadID, const uint groupThreadIdx, const uint3 groupID)
    {
        // No early return: every thread must reach the group barriers below.
        const bool active = groupID.z < gridCount;

        // Setup shading data struct (WGSL zero-initializes locals).
        ShadingData sd;
        sd.frame = ShadingFrame::createIdentity();
        sd.faceN = sd.frame.N;
        sd.frontFacing = true;
        sd.mtl = gScene.materials.getMaterialHeader(materialID);
        sd.materialID = materialID;
        sd.IoR = 1.f;

        float cosTheta = saturate(cosThetas[active ? groupID.z : 0]);
        float sinTheta = sqrt(1.f - cosTheta * cosTheta);
        sd.V = float3(sinTheta, 0.f, cosTheta);

        // Create MaterialInstance.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Approximate integral of the BSDF over grid cell with stratified samples at strata centers.
        float3 value = float3(0.f);
        UniformSampleGenerator sg = UniformSampleGenerator(threadID, 0);
        const float2 invSampleCount = float2(1.f / kSampleCount.x, 1.f / kSampleCount.y);
        for (uint sy = 0; sy < kSampleCount.y; sy++)
        {
            for (uint sx = 0; sx < kSampleCount.x; sx++)
            {
                // Compute sample position in [0,1]^2 and map to hemisphere.
                float2 offset = (float2(sx, sy) + 0.5f) * invSampleCount;
                float2 u = (float2(threadID) + offset) / float2(gridSize);
                float pdf = 0.f;
                float3 wo = sample_cosine_hemisphere_concentric(u, pdf);

                // Evaluate BSDF divided by pdf.
                value += pdf > 0.f ? mi.eval(sd, wo, sg) / pdf : float3(0.f);
            }
        }
        value *= 1.f / (kSampleCount.x * kSampleCount.y);

        // Reduce value over thread group (1024 threads) and write to memory.
        reduceGroupSum(value, groupThreadIdx, 1024);
        if (active && groupThreadIdx == 0)
        {
            uint groupIdx = (gridSize.x / 32 /* group width */) * groupID.y + groupID.x;
            uint dstIdx = resultCount * groupID.z + groupIdx;
            results[dstIdx] = gSharedResult[0] * (1.f / 1024);
        }
    }

    /** Reduce results buffer to a single final value per grid. */
    void executeFinal(const uint groupThreadIdx, const uint gridIdx)
    {
        const bool active = gridIdx < gridCount;
        float3 value = active ? results[resultCount * gridIdx + groupThreadIdx] : float3(0.f);
        reduceGroupSum(value, groupThreadIdx, 256);
        if (active && groupThreadIdx == 0)
        {
            finalResults[gridIdx] = gSharedResult[0] * (1.f / 256);
        }
    }
};

ParameterBlock<BSDFIntegrator> gIntegrator;

[numthreads(32, 32, 1)]
void mainIntegration(uint3 threadID : SV_DispatchThreadID, uint groupThreadIdx : SV_GroupIndex, uint3 groupID : SV_GroupID)
{
    gIntegrator.executeIntegration(threadID.xy, groupThreadIdx, groupID);
}

[numthreads(256, 1, 1)]
void mainFinal(uint3 threadID : SV_DispatchThreadID)
{
    gIntegrator.executeFinal(threadID.x, threadID.y);
}
