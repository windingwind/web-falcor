/**
 * WebFalcor override of Tests/Slang/SlangTests.cs.slang, reduced to testEnum:
 * the other entry points use 16/64-bit scalar types (double, uint64_t,
 * float16_t storage) that WGSL lacks, and fail the whole module.
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
import SlangShared;

RWStructuredBuffer<uint> result;

[numthreads(1, 1, 1)]
void testEnum()
{
    // Store the enum values cast to uint.
    result[0] = (uint)Type1::A;
    result[1] = (uint)Type1::B;
    result[2] = (uint)Type1::C;
    result[3] = (uint)Type1::D;

    result[4] = (uint)Type2::A;
    result[5] = (uint)Type2::B;
    result[6] = (uint)Type2::C;
    result[7] = (uint)Type2::D;

    result[8] = (uint)Type3::A;
    result[9] = (uint)Type3::B;
    result[10] = (uint)Type3::C;
    result[11] = (uint)Type3::D;
}
