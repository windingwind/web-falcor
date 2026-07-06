/**
 * WebFalcor override of RenderPasses/Utils/Composite/Composite.cs.slang
 * (DESIGN.md §4.3). Diff vs upstream: the output texture is write-only
 * (WTexture2D + Store) -- WGSL restricts read_write storage textures to r32*.
 */
#include "CompositeMode.slangh"

/**
 * Simple composite pass that blends two buffers together.
 */

cbuffer CB
{
    uint2 frameDim;
    float scaleA;
    float scaleB;
}

// Inputs
Texture2D<float4> A;
Texture2D<float4> B;

// Output
#if !defined(OUTPUT_FORMAT)
#error OUTPUT_FORMAT is undefined
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_FLOAT
WTexture2D<float4> output;
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_UINT
WTexture2D<uint4> output;
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_SINT
WTexture2D<int4> output;
#else
#error OUTPUT_FORMAT unknown
#endif

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= frameDim))
        return;

    float4 result = float4(0.f);
#if !defined(COMPOSITE_MODE)
#error COMPOSITE_MODE is undefined
#elif COMPOSITE_MODE == COMPOSITE_MODE_ADD
    result = (scaleA * A[pixel]) + (scaleB * B[pixel]);
#elif COMPOSITE_MODE == COMPOSITE_MODE_MULTIPLY
    result = (scaleA * A[pixel]) * (scaleB * B[pixel]);
#else
#error COMPOSITE_MODE unknown
#endif

#if OUTPUT_FORMAT == OUTPUT_FORMAT_FLOAT
    output.Store(pixel, result);
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_UINT
    output.Store(pixel, uint4(round(result)));
#else
    output.Store(pixel, int4(round(result)));
#endif
}
