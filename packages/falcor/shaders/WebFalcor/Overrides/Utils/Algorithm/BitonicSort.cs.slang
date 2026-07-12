/**
 * WebFalcor override of Utils/Algorithm/BitonicSort.cs.slang (docs §9).
 * The native kernel shuffles within warps via NVAPI (warp-32 only); WGSL has
 * no portable shuffle, so the same bitonic network runs entirely in shared
 * memory (identical comparator predicate, identical chunk results).
 *
 * Host defines: CHUNK_SIZE (power-of-two) and GROUP_SIZE (power-of-two
 * <= 1024, >= CHUNK_SIZE). One thread per element.
 */

#if (CHUNK_SIZE > GROUP_SIZE)
#error CHUNK_SIZE > GROUP_SIZE not supported
#endif

cbuffer CB
{
    uint gTotalSize; ///< Total number of elements.
    uint gDispatchX; ///< Number of thread groups in dispatch dimension X.
};

RWStructuredBuffer<uint> gData; ///< The data buffer we're sorting in-place.

groupshared uint gSharedData[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint groupIdx: SV_GroupIndex)
{
    const uint group = groupID.y * gDispatchX + groupID.x;
    const uint thid = groupIdx;
    const uint globalIdx = group * GROUP_SIZE + thid;

    const uint N = CHUNK_SIZE;
    const uint i = globalIdx & (N - 1); // local index in chunk

    // Out-of-bounds elements sort as UINT_MAX (placed last), as in native.
    uint value = 0xffffffffu;
    if (globalIdx < gTotalSize)
        value = gData[globalIdx];
    gSharedData[thid] = value;
    GroupMemoryBarrierWithGroupSync();

    for (uint k = 2; k <= N; k <<= 1)
    {
        const bool dir = ((i & k) == 0); // ascending (true) or descending
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            // Partner stays in-chunk: j < N and chunks tile thread groups.
            const uint a = gSharedData[thid];
            const uint b = gSharedData[thid ^ j];
            GroupMemoryBarrierWithGroupSync();
            // Matches the native predicate: the lower element of an
            // ascending pair keeps the min (ties keep either; values equal).
            const bool keepMin = (((i & j) == 0) == dir);
            gSharedData[thid] = (keepMin == (a <= b)) ? a : b;
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (globalIdx < gTotalSize)
        gData[globalIdx] = gSharedData[thid];
}
