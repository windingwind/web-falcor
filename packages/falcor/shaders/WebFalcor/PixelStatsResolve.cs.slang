// Resolves the WebFalcor pixel-stats buffer into the PathTracer's optional
// rayCount / pathLength outputs (native copies its stats textures instead).
// Region layout matches the PixelStats override: 0 visibility rays,
// 1 closestHit rays, 2 path length (N = gFrameDim.x * gFrameDim.y each).
#define is_valid(name) (is_valid_##name != 0)

cbuffer CB
{
    uint2 gFrameDim;
}

StructuredBuffer<uint> gStatsBuffer;

WTexture2D<uint> gRayCount;
WTexture2D<uint> gPathLength;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim))
        return;
    const uint n = gFrameDim.x * gFrameDim.y;
    const uint idx = pixel.y * gFrameDim.x + pixel.x;
    if (is_valid(gRayCount))
        gRayCount.Store(pixel, gStatsBuffer[idx] + gStatsBuffer[n + idx]);
    if (is_valid(gPathLength))
        gPathLength.Store(pixel, gStatsBuffer[2 * n + idx]);
}
