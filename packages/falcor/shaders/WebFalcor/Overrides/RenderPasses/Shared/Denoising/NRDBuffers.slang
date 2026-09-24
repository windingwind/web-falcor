/**
 * WebFalcor override of RenderPasses/Shared/Denoising/NRDBuffers.slang. WebGPU allows 8 storage
 * textures per stage and the path tracer's NRD outputs are 13 (18 with the resolve's), so every
 * member is a view with native's subscript syntax: per-pixel members address regions of one
 * float4 buffer (gNRDPixelData, copied into the output textures afterwards), per-sample members
 * fields of one struct buffer (gNRDSampleData). A region of kNRDNoRegion drops writes, like
 * native's unbound UAVs. The views hold no state, so their setters are nonmutating.
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
import NRDData;

#if !defined(OUTPUT_NRD_DATA) || !defined(OUTPUT_NRD_ADDITIONAL_DATA)
#error NRDBuffers.slang is missing defines
#endif

static const bool kOutputNRDData = OUTPUT_NRD_DATA;
static const bool kOutputNRDAdditionalData = OUTPUT_NRD_ADDITIONAL_DATA;

static const uint kNRDNoRegion = 0xffffffffu;

/// Per-sample NRD data, one element per sample (was five structured buffers).
struct NRDSampleData
{
    NRDRadiance radiance;
    float4 primaryHitNEEOnDelta;
    float4 emission;
    float4 reflectance;
    float hitDist;
    // 96 bytes in WGSL (NRDRadiance's float3 aligns to 16); scalar padding keeps it there.
    float _pad0;
    float _pad1;
    float _pad2;
};

// Declared only with NRD output: loose globals add a bind group, and RTXDI kernels already use all 4.
#if OUTPUT_NRD_DATA || OUTPUT_NRD_ADDITIONAL_DATA
RWStructuredBuffer<NRDSampleData> gNRDSampleData;
/// Per-pixel NRD outputs, one region of frameDim.x * frameDim.y texels per connected output.
RWStructuredBuffer<float4> gNRDPixelData;
#define NRD_PIXEL(i) gNRDPixelData[i]
#define NRD_SAMPLE(i) gNRDSampleData[i]
#else
static NRDSampleData gNRDSampleDataNone;
static float4 gNRDPixelDataNone;
#define NRD_PIXEL(i) gNRDPixelDataNone
#define NRD_SAMPLE(i) gNRDSampleDataNone
#endif

struct NRDPixelView4
{
    uint region;
    uint width;
    uint pixelCount;

    __subscript(uint2 pixel) -> float4
    {
        get { return region == kNRDNoRegion ? float4(0.f) : NRD_PIXEL(region * pixelCount + pixel.y * width + pixel.x); }
        [nonmutating] set { if (region != kNRDNoRegion) NRD_PIXEL(region * pixelCount + pixel.y * width + pixel.x) = newValue; }
    }
};

struct NRDPixelView1
{
    uint region;
    uint width;
    uint pixelCount;

    __subscript(uint2 pixel) -> float
    {
        get { return region == kNRDNoRegion ? 0.f : NRD_PIXEL(region * pixelCount + pixel.y * width + pixel.x).x; }
        [nonmutating] set { if (region != kNRDNoRegion) NRD_PIXEL(region * pixelCount + pixel.y * width + pixel.x) = float4(newValue, 0.f, 0.f, 0.f); }
    }
};

struct NRDSampleRadianceView
{
    __subscript(uint i) -> NRDRadiance { get { return NRD_SAMPLE(i).radiance; } [nonmutating] set { NRD_SAMPLE(i).radiance = newValue; } }
};
struct NRDSampleNEEOnDeltaView
{
    __subscript(uint i) -> float4 { get { return NRD_SAMPLE(i).primaryHitNEEOnDelta; } [nonmutating] set { NRD_SAMPLE(i).primaryHitNEEOnDelta = newValue; } }
};
struct NRDSampleHitDistView
{
    __subscript(uint i) -> float { get { return NRD_SAMPLE(i).hitDist; } [nonmutating] set { NRD_SAMPLE(i).hitDist = newValue; } }
};
struct NRDSampleEmissionView
{
    __subscript(uint i) -> float4 { get { return NRD_SAMPLE(i).emission; } [nonmutating] set { NRD_SAMPLE(i).emission = newValue; } }
};
struct NRDSampleReflectanceView
{
    __subscript(uint i) -> float4 { get { return NRD_SAMPLE(i).reflectance; } [nonmutating] set { NRD_SAMPLE(i).reflectance = newValue; } }
};

struct NRDBuffers
{
    NRDSampleRadianceView sampleRadiance;
    NRDSampleNEEOnDeltaView samplePrimaryHitNEEOnDelta;
    NRDSampleHitDistView sampleHitDist;
    NRDSampleEmissionView sampleEmission;
    NRDSampleReflectanceView sampleReflectance;

    NRDPixelView4 primaryHitEmission;
    NRDPixelView4 primaryHitDiffuseReflectance;
    NRDPixelView4 primaryHitSpecularReflectance;

    NRDPixelView4 deltaReflectionReflectance;
    NRDPixelView4 deltaReflectionEmission;
    NRDPixelView4 deltaReflectionNormWRoughMaterialID;
    NRDPixelView1 deltaReflectionPathLength;
    NRDPixelView1 deltaReflectionHitDist;

    NRDPixelView4 deltaTransmissionReflectance;
    NRDPixelView4 deltaTransmissionEmission;
    NRDPixelView4 deltaTransmissionNormWRoughMaterialID;
    NRDPixelView1 deltaTransmissionPathLength;
    NRDPixelView4 deltaTransmissionPosW;
};
