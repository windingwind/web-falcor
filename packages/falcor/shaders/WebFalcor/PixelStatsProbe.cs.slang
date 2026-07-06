// Regression probe for the PixelStats override: define propagation into the
// imported module, Atomic<uint> buffer binding, and region indexing.
import Rendering.Utils.PixelStats;

cbuffer ProbeCB
{
    uint2 gDim;
}

[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gDim))
        return;
    logSetPixel(tid.xy);
    logTraceRay(PixelStatsRayType::Visibility);
    if ((tid.x & 1) == 1)
        logTraceRay(PixelStatsRayType::ClosestHit);
    logPathLength(tid.x + tid.y);
    logPathVertex();
}
