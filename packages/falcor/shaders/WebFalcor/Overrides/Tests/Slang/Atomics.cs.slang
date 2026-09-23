/**
 * WebFalcor override of Tests/Slang/Atomics.cs.slang. WGSL has only 32-bit
 * integer atomics: InterlockedAddF32 on the buffer is emulated by a
 * compare-exchange loop on an Atomic<uint> buffer. The fp16 variants and the
 * RWTexture2D<float> atomics have no WGSL equivalent and are dropped.
 */
/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
StructuredBuffer<float> data;
RWStructuredBuffer<Atomic<uint>> resultBuf; // WebFalcor: was RWByteAddressBuffer

/** InterlockedAddF32 at dword `index` via compare-exchange. */
void interlockedAddF32(uint index, float value)
{
    uint expected = resultBuf[index].load();
    for (;;)
    {
        const uint original = resultBuf[index].compareExchange(expected, asuint(asfloat(expected) + value));
        if (original == expected)
            break;
        expected = original;
    }
}

[numthreads(256, 1, 1)]
void testBufferAddF32(uint3 threadID: SV_DispatchThreadID)
{
    float x = data[threadID.x];
    interlockedAddF32(0, x);
    interlockedAddF32(1, -x);
}
