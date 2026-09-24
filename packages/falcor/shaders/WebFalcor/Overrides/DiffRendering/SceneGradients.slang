/**
 * WebFalcor override of DiffRendering/SceneGradients.slang: WARDiffPathTracer defines
 * WEBFALCOR_DIFF_PARAMS_AS_CB to declare the block as a ConstantBuffer, which fits it in
 * WebGPU's 4 bind groups. Other programs keep the ParameterBlock. WGSL has neither resource
 * arrays nor float atomics: tmpGrads[4] becomes tmpGrads0..3 of Atomic<uint>, and
 * InterlockedAddF32 a compare-exchange loop.
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

__exported import DiffRendering.SharedTypes;

struct SceneGradients
{
    uint gradDim[(uint)GradientType::Count];
    uint hashSize[(uint)GradientType::Count];

    // Temporary buffers for keeping gradients before aggregating them (one per GradientType).
    RWStructuredBuffer<Atomic<uint>> tmpGrads0;
    RWStructuredBuffer<Atomic<uint>> tmpGrads1;
    RWStructuredBuffer<Atomic<uint>> tmpGrads2;
    RWStructuredBuffer<Atomic<uint>> tmpGrads3;

    uint getGradDim(GradientType gradType) { return gradDim[(uint)gradType]; }

    uint getHashSize(GradientType gradType) { return hashSize[(uint)gradType]; }

    uint loadTmp(GradientType gradType, uint index)
    {
        switch (gradType)
        {
        case GradientType::Material: return tmpGrads0[index].load();
        case GradientType::MeshPosition: return tmpGrads1[index].load();
        case GradientType::MeshNormal: return tmpGrads2[index].load();
        default: return tmpGrads3[index].load();
        }
    }

    uint compareExchangeTmp(GradientType gradType, uint index, uint expected, uint desired)
    {
        switch (gradType)
        {
        case GradientType::Material: return tmpGrads0[index].compareExchange(expected, desired);
        case GradientType::MeshPosition: return tmpGrads1[index].compareExchange(expected, desired);
        case GradientType::MeshNormal: return tmpGrads2[index].compareExchange(expected, desired);
        default: return tmpGrads3[index].compareExchange(expected, desired);
        }
    }

    float getGrad(GradientType gradType, uint gradIndex, uint hashIndex)
    {
        uint index = hashIndex * gradDim[(uint)gradType] + gradIndex;
        return asfloat(loadTmp(gradType, index));
    }

    void atomicAddGrad(GradientType gradType, uint gradIndex, uint hashIndex, float value)
    {
        const uint dim = getGradDim(gradType);
        if (gradIndex < dim)
        {
            const uint index = hashIndex * dim + gradIndex;
            uint expected = loadTmp(gradType, index);
            for (;;)
            {
                const uint original = compareExchangeTmp(gradType, index, expected, asuint(asfloat(expected) + value));
                if (original == expected)
                    break;
                expected = original;
            }
        }
    }
};

#if WEBFALCOR_DIFF_PARAMS_AS_CB
ConstantBuffer<SceneGradients> gSceneGradients;
#else
ParameterBlock<SceneGradients> gSceneGradients;
#endif

#ifdef SCENE_GRADIENTS_BLOCK
void main() {}
#endif
