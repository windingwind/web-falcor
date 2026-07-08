/**
 * WebFalcor override of RenderPasses/AccumulatePass/Accumulate.cs.slang.
 *
 * Diff vs upstream (see docs §4.3): per-pixel accumulation state moves
 * from RWTexture2D<float4> to RWStructuredBuffer<float4> — WGSL restricts
 * read_write storage textures to r32float/r32uint/r32sint, and the state
 * textures are read-modify-written each frame. Double mode (64-bit) is not
 * representable in WGSL (parity matrix §8.1) and is omitted; the host maps
 * Double to SingleCompensated with a warning.
 *
 * Entry points, defines, cbuffer and logic match upstream.
 */

// bool is non-host-shareable in WGSL uniform buffers -> uint flags.
cbuffer PerFrameCB
{
    uint2 gResolution;
    uint gAccumCount;
    uint gAccumulate;
    uint gMovingAverageMode;
}

#define INPUT_FORMAT_FLOAT 0
#define INPUT_FORMAT_UINT 1
#define INPUT_FORMAT_SINT 2

#if _INPUT_FORMAT == INPUT_FORMAT_FLOAT
typedef float4 InputDataType;
#elif _INPUT_FORMAT == INPUT_FORMAT_UINT
typedef uint4 InputDataType;
#elif _INPUT_FORMAT == INPUT_FORMAT_SINT
typedef int4 InputDataType;
#else
#error "Invalid format"
#endif

Texture2D<InputDataType> gCurFrame;
// WTexture2D (write-only): RWTexture2D would emit read_write access, which WGSL
// restricts to r32* formats.
WTexture2D<float4> gOutputFrame;

// Accumulation state, linearized (index = y * gResolution.x + x).
RWStructuredBuffer<float4> gLastFrameSum;  // If mode is Single or SingleKahan
RWStructuredBuffer<float4> gLastFrameCorr; // If mode is SingleKahan

[numthreads(16, 16, 1)]
void accumulateSingle(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const uint idx = pixelPos.y * gResolution.x + pixelPos.x;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate != 0)
    {
        float curWeight = 1.0 / (gAccumCount + 1);

        if (gMovingAverageMode != 0)
        {
            output = lerp(gLastFrameSum[idx], curColor, curWeight);
            gLastFrameSum[idx] = output;
        }
        else
        {
            float4 sum = gLastFrameSum[idx] + curColor;
            output = sum * curWeight;
            gLastFrameSum[idx] = sum;
        }
    }
    else
    {
        output = curColor;
    }

    gOutputFrame.Store(pixelPos, output);
}

[numthreads(16, 16, 1)]
void accumulateSingleCompensated(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const uint idx = pixelPos.y * gResolution.x + pixelPos.x;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate != 0)
    {
        // Kahan compensated summation (mirrors upstream).
        const float4 sum = gLastFrameSum[idx];
        float4 corr = gLastFrameCorr[idx];
        float4 y = curColor - corr;
        float4 t = sum + y;
        corr = (t - sum) - y;

        float curWeight = 1.0 / (gAccumCount + 1);
        output = t * curWeight;

        gLastFrameSum[idx] = t;
        gLastFrameCorr[idx] = corr;
    }
    else
    {
        output = curColor;
    }

    gOutputFrame.Store(pixelPos, output);
}
