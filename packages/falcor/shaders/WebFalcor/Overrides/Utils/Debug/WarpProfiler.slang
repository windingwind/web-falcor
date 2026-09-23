/**
 * WebFalcor override of Utils/Debug/WarpProfiler.slang. WGSL atomics need an
 * atomic-typed buffer (Atomic<uint>), and WGSL has no WaveMatch: the matching
 * lanes are found by a WaveReadLaneFirst/WaveActiveBallot loop instead, and
 * WaveActiveCountBits (no WGSL lowering) is countbits of the ballot.
 */
/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * GPU side of the warp profiler (mirrors upstream).
 * WebFalcor: requires the WebGPU 'subgroups' feature and a 32-wide subgroup.
 */
RWStructuredBuffer<Atomic<uint>> gWarpHistogram; // WebFalcor: was RWStructuredBuffer<uint>

/** Records the number of active lanes of the current warp in bin `binIndex`.
    Must be called from all lanes of the warp that are active.
*/
void warpUtilization(uint binIndex)
{
    uint activeCount = countbits(WaveActiveBallot(true).x); // WebFalcor: WaveActiveCountBits has no WGSL lowering
    if (WaveIsFirstLane())
    {
        binIndex = binIndex * 32 + activeCount - 1;
        gWarpHistogram[binIndex].add(1);
    }
}

/** WaveMatch(value).x for 32-wide subgroups: lanes peel off in groups of equal value. */
uint webfalcorWaveMatch(uint value)
{
    uint mask = 0;
    for (;;)
    {
        if (WaveReadLaneFirst(value) == value)
        {
            mask = WaveActiveBallot(true).x;
            break;
        }
    }
    return mask;
}

/** Records the number of distinct `value`s among the active lanes in bin `binIndex`. */
void warpDivergence(uint binIndex, uint value)
{
    uint groupMask = webfalcorWaveMatch(value);
    uint firstLaneInGroup = firstbitlow(groupMask);
    if (WaveGetLaneIndex() == firstLaneInGroup)
    {
        warpUtilization(binIndex);
    }
}
