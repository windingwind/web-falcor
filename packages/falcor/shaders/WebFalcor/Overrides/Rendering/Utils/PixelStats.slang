/**
 * WebFalcor override of Rendering/Utils/PixelStats.slang (docs §4.3).
 * Diffs vs upstream: WGSL has no binding arrays and no storage-texture
 * atomics, and storage buffers are capped per-stage (16 on the dev GPU) —
 * the five per-type RWTexture2D<uint> counters become regions of ONE
 * Atomic<uint> buffer (region r spans [r*N, (r+1)*N), N = dim.x*dim.y,
 * pixel index y*dim.x + x). A PixelStatsCB with the frame dim is added
 * for the indexing.
 */
__exported import PixelStatsShared;

#ifdef _PIXEL_STATS_ENABLED
cbuffer PixelStatsCB
{
    uint2 gPixelStatsDim;
}

/// Regions: 0 visibility rays, 1 closestHit rays, 2 path length,
/// 3 path vertex count, 4 volume lookups.
RWStructuredBuffer<Atomic<uint>> gStatsBuffer;

static uint2 gPixelStatsPixel;

uint pixelStatsIndex(uint region)
{
    return region * gPixelStatsDim.x * gPixelStatsDim.y + gPixelStatsPixel.y * gPixelStatsDim.x + gPixelStatsPixel.x;
}
#endif

void logSetPixel(uint2 pixel)
{
#ifdef _PIXEL_STATS_ENABLED
    gPixelStatsPixel = pixel;
#endif
}

void logTraceRay(PixelStatsRayType rayType)
{
#ifdef _PIXEL_STATS_ENABLED
    gStatsBuffer[pixelStatsIndex((uint)rayType)].add(1);
#endif
}

void logPathLength(uint pathLength)
{
#ifdef _PIXEL_STATS_ENABLED
    gStatsBuffer[pixelStatsIndex(2)].store(pathLength);
#endif
}

void logPathVertex()
{
#ifdef _PIXEL_STATS_ENABLED
    gStatsBuffer[pixelStatsIndex(3)].add(1);
#endif
}

void logVolumeLookup()
{
#ifdef _PIXEL_STATS_ENABLED
    gStatsBuffer[pixelStatsIndex(4)].add(1);
#endif
}
