/**
 * WebFalcor override of Utils/Algorithm/PrefixSum.cs.slang.
 *
 * Diffs vs upstream (see docs §4.3):
 * - RW/ByteAddressBuffer -> structured buffers: WGSL atomics require
 *   atomic<u32>-typed elements (Slang Atomic<uint>), and byte-address loads
 *   become element indices (all accesses were 4-byte aligned uints upstream).
 *
 * Interface (entry points, defines, cbuffer, binding names, semantics) is
 * identical to upstream; host code is unchanged apart from buffer struct sizes.
 */

cbuffer CB
{
    uint gNumGroups;      ///< Number of groups we'll process, each group is 2N elements.
    uint gTotalNumElems;  ///< Total number of elements. Does not have to be a multiple of the group size.
    uint gIter;           ///< Current iteration when the input exceeds 2*GROUP_SIZE*GROUP_SIZE elements.
};

RWStructuredBuffer<uint> gData;                     ///< Data buffer (in-place prefix sum).
RWStructuredBuffer<Atomic<uint>> gPrefixGroupSums;  ///< One uint per group: sum of all elements in the group and to the left.
RWStructuredBuffer<Atomic<uint>> gTotalSum;         ///< One uint: total sum of a prefix sum iteration.
StructuredBuffer<uint> gPrevTotalSum;               ///< One uint: total sum of the previous iteration.

groupshared uint gSharedData[2 * GROUP_SIZE]; ///< Temporary working buffer in shared memory for 2N elements.

/**
 * Parallel prefix sum in shared memory over consecutive groups of 2N elements
 * (Blelloch scan). Same algorithm as upstream.
 */
[numthreads(GROUP_SIZE, 1, 1)]
void groupScan(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    const uint thid = groupThreadID.x;
    const uint groupIdx = groupID.x;

    const uint prevSum = gPrevTotalSum[0];
    const uint idx = 2 * GROUP_SIZE * GROUP_SIZE * gIter + groupIdx * (2 * GROUP_SIZE) + thid;
    gSharedData[thid] = (idx < gTotalNumElems ? gData[idx] : 0);
    gSharedData[thid + GROUP_SIZE] = ((idx + GROUP_SIZE) < gTotalNumElems ? gData[idx + GROUP_SIZE] : 0);

    // Reduction phase.
    uint offset = 1;
    for (uint d = GROUP_SIZE; d > 0; d >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();

        if (thid < d)
        {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = ai + offset;

            gSharedData[bi] += gSharedData[ai];
        }
        offset *= 2;
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute prefix sum over groups (atomics, since groups run out-of-order).
    if (thid >= groupIdx && thid < gNumGroups)
    {
        uint sum = gSharedData[2 * GROUP_SIZE - 1];
        gPrefixGroupSums[thid].add(sum);

        if (thid == gNumGroups - 1)
        {
            gTotalSum[0].add(sum);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Zero out top element for the down-sweep phase.
    if (thid == 0)
        gSharedData[2 * GROUP_SIZE - 1] = 0;

    // Down-sweep phase.
    for (uint d = 1; d <= GROUP_SIZE; d *= 2)
    {
        offset >>= 1;

        GroupMemoryBarrierWithGroupSync();

        if (thid < d)
        {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = ai + offset;

            uint tmp = gSharedData[ai];
            gSharedData[ai] = gSharedData[bi];
            gSharedData[bi] += tmp;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Write results and add previous iteration's sum.
    if (idx < gTotalNumElems)
        gData[idx] = gSharedData[thid] + prevSum;
    if ((idx + GROUP_SIZE) < gTotalNumElems)
        gData[idx + GROUP_SIZE] = gSharedData[thid + GROUP_SIZE] + prevSum;
}

/**
 * Finalize pass: add group sums to the left to each group.
 * Skips the first 2N elements (their group's prefix sum is zero).
 */
[numthreads(GROUP_SIZE, 1, 1)]
void finalizeGroups(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    const uint thid = groupThreadID.x;
    const uint groupIdx = groupID.x;

    uint sum = gPrefixGroupSums[groupIdx >> 1].load();
    uint globalIdx = (groupIdx * GROUP_SIZE) + thid + 2 * GROUP_SIZE * (1 + GROUP_SIZE * gIter);

    if (globalIdx < gTotalNumElems)
    {
        gData[globalIdx] = gData[globalIdx] + sum;
    }
}
