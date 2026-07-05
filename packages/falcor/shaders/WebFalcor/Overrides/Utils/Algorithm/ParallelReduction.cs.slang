/**
 * WebFalcor override of Utils/Algorithm/ParallelReduction.cs.slang.
 *
 * Diffs vs upstream (see DESIGN.md §4.3):
 * - RWBuffer/Buffer (typed texel buffers) -> RW/StructuredBuffer: WGSL has no
 *   texel buffers.
 * - Wave-intrinsic warp reduction (upstream assumes warp size 32) -> portable
 *   shared-memory tree reduction: WGSL subgroup size varies by adapter (8..64).
 *
 * Interface (entry points, defines, cbuffer, binding names) is identical to
 * upstream; host code is unchanged.
 */
#include "ParallelReductionType.slangh"
#include "Utils/Math/MathConstants.slangh"

cbuffer PerFrameCB
{
    uint2 gResolution; // Pixel dimensions of input texture.
    uint2 gNumTiles;   // Number of tiles in input texture.
    uint gElems;       // Number of elements in input intermediate buffer.
};

#if FORMAT_TYPE == FORMAT_TYPE_FLOAT
typedef vector<float, 4> DataType;
static const float kMaxValue = FLT_MAX;
static const float kMinValue = -FLT_MAX;
#elif FORMAT_TYPE == FORMAT_TYPE_SINT
typedef vector<int, 4> DataType;
static const int kMinValue = INT32_MIN;
static const int kMaxValue = INT32_MAX;
#elif FORMAT_TYPE == FORMAT_TYPE_UINT
typedef vector<uint, 4> DataType;
static const uint kMinValue = 0;
static const uint kMaxValue = UINT32_MAX;
#endif

Texture2D<DataType> gInput;

StructuredBuffer<DataType> gInputBuffer;
RWStructuredBuffer<DataType> gResult;

// 1024 threads fold into 256 shared slots, then tree-reduce (8 KB per array).
#define CACHE_SIZE 256

#if REDUCTION_TYPE == REDUCTION_TYPE_SUM
groupshared DataType gCache[CACHE_SIZE];
#elif REDUCTION_TYPE == REDUCTION_TYPE_MINMAX
groupshared DataType gCacheMin[CACHE_SIZE];
groupshared DataType gCacheMax[CACHE_SIZE];
#endif

DataType loadTexture(uint2 pixelCoords)
{
    DataType value = gInput[pixelCoords];
#if FORMAT_CHANNELS < 4
    // The default value for missing components is (0,0,0,1). Reset last component to zero for consistency.
    value.w = 0;
#endif
    return value;
}

#if REDUCTION_TYPE == REDUCTION_TYPE_SUM
/** Portable group reduction; writes single result to gResult[dstIdx]. */
void reduceSum(DataType value, uint dstIdx, uint groupThreadIdx)
{
    // Fold 1024 values into 256 shared slots, one quarter at a time.
    if (groupThreadIdx < CACHE_SIZE)
        gCache[groupThreadIdx] = value;
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIdx >= CACHE_SIZE && groupThreadIdx < 2 * CACHE_SIZE)
        gCache[groupThreadIdx - CACHE_SIZE] += value;
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIdx >= 2 * CACHE_SIZE && groupThreadIdx < 3 * CACHE_SIZE)
        gCache[groupThreadIdx - 2 * CACHE_SIZE] += value;
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIdx >= 3 * CACHE_SIZE)
        gCache[groupThreadIdx - 3 * CACHE_SIZE] += value;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction 256 -> 1.
    for (uint s = CACHE_SIZE / 2; s > 0; s >>= 1)
    {
        if (groupThreadIdx < s)
            gCache[groupThreadIdx] += gCache[groupThreadIdx + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupThreadIdx == 0)
        gResult[dstIdx] = gCache[0];
}
#endif

#if REDUCTION_TYPE == REDUCTION_TYPE_MINMAX
/** Portable group min/max; writes results to gResult[dstIdx*2], gResult[dstIdx*2+1]. */
void reduceMinMax(DataType minValue, DataType maxValue, uint dstIdx, uint groupThreadIdx)
{
    if (groupThreadIdx < CACHE_SIZE)
    {
        gCacheMin[groupThreadIdx] = minValue;
        gCacheMax[groupThreadIdx] = maxValue;
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIdx >= CACHE_SIZE && groupThreadIdx < 2 * CACHE_SIZE)
    {
        gCacheMin[groupThreadIdx - CACHE_SIZE] = min(gCacheMin[groupThreadIdx - CACHE_SIZE], minValue);
        gCacheMax[groupThreadIdx - CACHE_SIZE] = max(gCacheMax[groupThreadIdx - CACHE_SIZE], maxValue);
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIdx >= 2 * CACHE_SIZE && groupThreadIdx < 3 * CACHE_SIZE)
    {
        gCacheMin[groupThreadIdx - 2 * CACHE_SIZE] = min(gCacheMin[groupThreadIdx - 2 * CACHE_SIZE], minValue);
        gCacheMax[groupThreadIdx - 2 * CACHE_SIZE] = max(gCacheMax[groupThreadIdx - 2 * CACHE_SIZE], maxValue);
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIdx >= 3 * CACHE_SIZE)
    {
        gCacheMin[groupThreadIdx - 3 * CACHE_SIZE] = min(gCacheMin[groupThreadIdx - 3 * CACHE_SIZE], minValue);
        gCacheMax[groupThreadIdx - 3 * CACHE_SIZE] = max(gCacheMax[groupThreadIdx - 3 * CACHE_SIZE], maxValue);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = CACHE_SIZE / 2; s > 0; s >>= 1)
    {
        if (groupThreadIdx < s)
        {
            gCacheMin[groupThreadIdx] = min(gCacheMin[groupThreadIdx], gCacheMin[groupThreadIdx + s]);
            gCacheMax[groupThreadIdx] = max(gCacheMax[groupThreadIdx], gCacheMax[groupThreadIdx + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupThreadIdx == 0)
    {
        gResult[dstIdx * 2] = gCacheMin[0];
        gResult[dstIdx * 2 + 1] = gCacheMax[0];
    }
}
#endif

[numthreads(32, 32, 1)]
void initialPass(uint3 globalThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupId: SV_GroupID)
{
    const uint2 pixelCoords = globalThreadId.xy;
    const uint tileIdx = groupId.y * gNumTiles.x + groupId.x;

#if REDUCTION_TYPE == REDUCTION_TYPE_SUM
    DataType value = DataType(0);
    if (all(pixelCoords < gResolution))
        value = loadTexture(pixelCoords);

    reduceSum(value, tileIdx, groupThreadIdx);
#elif REDUCTION_TYPE == REDUCTION_TYPE_MINMAX
    DataType minValue = DataType(kMaxValue);
    DataType maxValue = DataType(kMinValue);
    if (all(pixelCoords < gResolution))
        minValue = maxValue = loadTexture(pixelCoords);

    reduceMinMax(minValue, maxValue, tileIdx, groupThreadIdx);
#endif
}

[numthreads(1024, 1, 1)]
void finalPass(uint3 globalThreadId: SV_DispatchThreadID, uint groupThreadIdx: SV_GroupIndex, uint3 groupId: SV_GroupID)
{
#if REDUCTION_TYPE == REDUCTION_TYPE_SUM
    DataType value = DataType(0);
    if (globalThreadId.x < gElems)
        value = gInputBuffer[globalThreadId.x];

    reduceSum(value, groupId.x, groupThreadIdx);
#elif REDUCTION_TYPE == REDUCTION_TYPE_MINMAX
    DataType minValue = DataType(kMaxValue);
    DataType maxValue = DataType(kMinValue);
    if (globalThreadId.x < gElems)
    {
        minValue = gInputBuffer[globalThreadId.x * 2];
        maxValue = gInputBuffer[globalThreadId.x * 2 + 1];
    }

    reduceMinMax(minValue, maxValue, groupId.x, groupThreadIdx);
#endif
}
