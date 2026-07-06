/**
 * WebFalcor override of RenderPasses/Utils/CrossFade/CrossFade.cs.slang
 * (DESIGN.md §4.3). Diff vs upstream: the output texture is write-only
 * (WTexture2D + Store) -- WGSL restricts read_write storage textures to r32*.
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
WTexture2D<float4> output;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= frameDim))
        return;

    float4 result = (scaleA * A[pixel]) + (scaleB * B[pixel]);

    output.Store(pixel, result);
}
