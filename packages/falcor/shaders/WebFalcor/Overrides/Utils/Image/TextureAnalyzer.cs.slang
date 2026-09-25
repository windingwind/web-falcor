/**
 * WebFalcor override of Utils/Image/TextureAnalyzer.cs.slang.
 *
 * Diff vs upstream: the wave reduction (WaveActiveBitOr/Min/Max + first-lane atomics) becomes
 * per-texel atomics; the atomic or/min/max produce the same result. WGSL subgroups are optional.
 * NaN texels are compared and reduced as HLSL and the wave ops did (see below).
 * The result is an Atomic<uint> structured buffer (WGSL has no ByteAddressBuffer atomics); offsets stay in bytes.
 */
#include "Utils/Math/MathConstants.slangh"

struct TextureAnalyzer
{
    Texture2D<float4> input;
    RWStructuredBuffer<Atomic<uint>> result; // WebFalcor: was RWByteAddressBuffer
    uint32_t resultOffset;
    uint2 inputDim;

    void clear(const uint index)
    {
        if (index >= inputDim.x)
            return;

        uint offset = (resultOffset + index * 64) / 4;
        for (uint i = 0; i < 4; i++)
        {
            result[offset + i].store(0);                    // mask
            result[offset + 4 + i].store(asuint(0.f));      // value
            result[offset + 8 + i].store(asuint(FLT_MAX));  // minValue
            result[offset + 12 + i].store(asuint(0.f));     // maxValue
        }
    }

    void analyze(const uint2 texel)
    {
        uint mask = 0;
        float4 minVal = FLT_MAX;
        float4 maxVal = -FLT_MAX;
        bool valid = false;

        if (all(texel < inputDim))
        {
            // Read reference value from top-left texel.
            float4 ref = input[uint2(0, 0)];

            if (all(texel == uint2(0, 0)))
            {
                for (uint i = 0; i < 4; i++) result[resultOffset / 4 + 4 + i].store(asuint(ref[i]));
            }

            // Compare current texel to the reference value.
            // Produce bit mask in bits 0-3 indicating which color channels differ.
            float4 val = input[texel];

            // WebFalcor: WGSL leaves NaN comparisons undefined; HLSL's NaN != x is true.
            for (uint c = 0; c < 4; c++)
                mask |= (val[c] != ref[c] || isnan(val[c]) || isnan(ref[c])) ? (1u << c) : 0;

            // Compute range flags for each color channel.
            // Produce flags in bits 4-19 for | 0...0 | A | B | G | R | 0000 | (each group is 4 bits).
            uint range = 0;
            for (int i = 3; i >= 0; i--)
            {
                if (val[i] > 0.f)
                    range |= 1;
                if (val[i] < 0.f)
                    range |= 2;
                if (isinf(val[i]))
                    range |= 4;
                if (isnan(val[i]))
                    range |= 8;
                range <<= 4;
            }

            // Store range and const flags together.
            mask |= range;

            // WebFalcor: the wave min/max skipped NaN lanes; without it, NaN texels don't contribute.
            for (uint c = 0; c < 4; c++)
            {
                minVal[c] = isnan(val[c]) ? FLT_MAX : val[c];
                maxVal[c] = isnan(val[c]) ? -FLT_MAX : val[c];
            }
            valid = true;
        }

        if (!valid)
            return;

        // WebFalcor: every texel writes its own atomics (no wave reduction; same result).
        {
            if (mask != 0)
                result[resultOffset / 4].or(mask);

            // Clamp to zero since we're using integer atomics.
            minVal = max(minVal, float4(0));
            maxVal = max(maxVal, float4(0));

            result[(resultOffset + 32) / 4].min(asuint(minVal.x));
            result[(resultOffset + 36) / 4].min(asuint(minVal.y));
            result[(resultOffset + 40) / 4].min(asuint(minVal.z));
            result[(resultOffset + 44) / 4].min(asuint(minVal.w));

            result[(resultOffset + 48) / 4].max(asuint(maxVal.x));
            result[(resultOffset + 52) / 4].max(asuint(maxVal.y));
            result[(resultOffset + 56) / 4].max(asuint(maxVal.z));
            result[(resultOffset + 60) / 4].max(asuint(maxVal.w));
        }
    }
};

ConstantBuffer<TextureAnalyzer> gTextureAnalyzer;

[numthreads(256, 1, 1)]
void clear(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gTextureAnalyzer.clear(dispatchThreadId.x);
}

[numthreads(16, 16, 1)]
void analyze(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gTextureAnalyzer.analyze(dispatchThreadId.xy);
}
