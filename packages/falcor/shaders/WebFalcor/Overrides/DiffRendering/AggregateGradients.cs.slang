/**
 * WebFalcor override of DiffRendering/AggregateGradients.cs.slang. WGSL has no float atomics:
 * `grads` is an Atomic<uint> buffer and InterlockedAddF32 a compare-exchange loop.
 */
/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/NVAPI.slangh"

import DiffRendering.SharedTypes;

struct GradientsAggregator
{
    uint gradDim;
    uint hashSize;

    ByteAddressBuffer tmpGrads;
    RWStructuredBuffer<Atomic<uint>> grads; // WebFalcor: was RWByteAddressBuffer

    void aggregateDirect(uint2 threadID)
    {
        if (threadID.x >= gradDim || threadID.y >= hashSize)
            return;

        uint index = threadID.y * gradDim + threadID.x;
        float value = asfloat(tmpGrads.Load(index * 4));
        grads[threadID.x].store(asuint(value));
    }

    void aggregateHashGrid(uint2 threadID)
    {
        if (threadID.x >= gradDim || threadID.y >= hashSize)
            return;

        uint index = threadID.y * gradDim + threadID.x;
        float value = asfloat(tmpGrads.Load(index * 4));
        uint expected = grads[threadID.x].load();
        for (;;)
        {
            const uint original = grads[threadID.x].compareExchange(expected, asuint(asfloat(expected) + value));
            if (original == expected)
                break;
            expected = original;
        }
    }
}

ParameterBlock<GradientsAggregator> gAggregator;

[numthreads(256, 1, 1)]
void mainDirect(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gAggregator.aggregateDirect(dispatchThreadID.xy);
}

[numthreads(64, 8, 1)]
void mainHashGrid(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gAggregator.aggregateHashGrid(dispatchThreadID.xy);
}
