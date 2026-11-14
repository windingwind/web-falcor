/**
 * MotionStats: deterministic GPU reduction of screen-space motion vectors (see MotionStats.ts).
 *
 * Motion of a pixel m = -mvec * frameDim (pixels moved since the previous frame; mvec = previous - current position
 * in UV units, as GBufferRaster/GBufferRT write it). Angles are atan2(m.y, m.x) in pixel coordinates (x right, y down),
 * in [-pi, pi).
 *  clearMain  : zero the histogram counters.
 *  reduceMain : per 16x16 tile, a fixed-order shared-memory tree reduction of (sum m, sum |m|, sum |m|^2) and
 *               (max |m|, count, moving count) into two partials per tile; histogram counts with integer atomics.
 *  finalMain  : one workgroup reduces the partials in a fixed order and writes the stats and histogram textures.
 * All float sums run in a fixed order, so the result is bitwise reproducible; histogram counts are exact integers.
 */
#include "Utils/Math/MathConstants.slangh"

static const uint kGroupSize = 256;

cbuffer CB
{
    uint2 gFrameDim;
    uint gGroupsX;
    uint gNumGroups;
    uint gMagBins;
    uint gDirBins;
    float gMagRange;   ///< Magnitude histogram covers [0, gMagRange) pixels in gMagBins bins; the last bin also takes overflow.
    float gMinMotion;  ///< Pixels moving at most this far count as static (excluded from the direction histogram).
    uint gUseMask;
};

Texture2D<float4> gMvec;
Texture2D<float4> gMask;
RWStructuredBuffer<float4> gPartials;
RWStructuredBuffer<Atomic<uint>> gHist;
RWTexture2D<float4> gStats;
RWTexture2D<float4> gHistOut;

groupshared float4 sSum[kGroupSize];
groupshared float4 sAux[kGroupSize];

float4 combineAux(float4 a, float4 b)
{
    return float4(max(a.x, b.x), a.y + b.y, a.z + b.z, a.w + b.w);
}

void treeReduce(uint gi)
{
    for (uint stride = kGroupSize / 2; stride > 0; stride >>= 1)
    {
        if (gi < stride)
        {
            sSum[gi] = sSum[gi] + sSum[gi + stride];
            sAux[gi] = combineAux(sAux[gi], sAux[gi + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(64, 1, 1)]
void clearMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x < gMagBins + gDirBins) gHist[tid.x].store(0u);
}

[numthreads(16, 16, 1)]
void reduceMain(uint3 groupId: SV_GroupID, uint3 tid: SV_DispatchThreadID, uint gi: SV_GroupIndex)
{
    float4 s = float4(0.f);
    float4 a = float4(0.f);
    const uint2 pixel = tid.xy;
    if (all(pixel < gFrameDim) && (gUseMask == 0 || gMask[pixel].w != 0.f))
    {
        const float2 m = -gMvec[pixel].xy * float2(gFrameDim);
        const float mag = length(m);
        const bool moving = mag > gMinMotion;
        s = float4(m, mag, mag * mag);
        a = float4(mag, 1.f, moving ? 1.f : 0.f, 0.f);
        const uint mb = min(uint(max(mag / gMagRange, 0.f) * float(gMagBins)), gMagBins - 1);
        gHist[mb].add(1u);
        if (moving)
        {
            const float u = (atan2(m.y, m.x) + M_PI) / (2.f * M_PI);
            const uint db = min(uint(u * float(gDirBins)), gDirBins - 1);
            gHist[gMagBins + db].add(1u);
        }
    }
    sSum[gi] = s;
    sAux[gi] = a;
    GroupMemoryBarrierWithGroupSync();
    treeReduce(gi);
    if (gi == 0)
    {
        const uint g = groupId.y * gGroupsX + groupId.x;
        gPartials[2 * g] = sSum[0];
        gPartials[2 * g + 1] = sAux[0];
    }
}

[numthreads(256, 1, 1)]
void finalMain(uint gi: SV_GroupIndex)
{
    float4 s = float4(0.f);
    float4 a = float4(0.f);
    for (uint g = gi; g < gNumGroups; g += kGroupSize)
    {
        s = s + gPartials[2 * g];
        a = combineAux(a, gPartials[2 * g + 1]);
    }
    sSum[gi] = s;
    sAux[gi] = a;
    GroupMemoryBarrierWithGroupSync();
    treeReduce(gi);

    if (gi == 0)
    {
        s = sSum[0];
        a = sAux[0];
        const float n = a.y;
        const float inv = n > 0.f ? 1.f / n : 0.f;
        const float2 mean = s.xy * inv;
        uint best = 0, bestCount = 0, total = 0;
        for (uint b = 0; b < gDirBins; b++)
        {
            const uint c = gHist[gMagBins + b].load();
            total += c;
            if (c > bestCount) { bestCount = c; best = b; }
        }
        const float binWidth = 2.f * M_PI / float(gDirBins);
        const float domAngle = total > 0 ? -M_PI + (float(best) + 0.5f) * binWidth : 0.f;
        const float domShare = total > 0 ? float(bestCount) / float(total) : 0.f;
        const float meanLen = length(mean);
        gStats[uint2(0, 0)] = float4(mean, s.z * inv, a.x);
        gStats[uint2(1, 0)] = float4(sqrt(s.w * inv), a.z * inv, domAngle, domShare);
        gStats[uint2(2, 0)] = float4(meanLen > 0.f ? atan2(mean.y, mean.x) : 0.f, n, meanLen, float(total));
        gStats[uint2(3, 0)] = float4(gMagRange, gMinMotion, float(gMagBins), float(gDirBins));
    }
    const float n = sAux[0].y;
    const float inv = n > 0.f ? 1.f / n : 0.f;
    if (gi < gMagBins)
    {
        const uint c = gHist[gi].load();
        gHistOut[uint2(gi, 0)] = float4(float(c) * inv, float(c), 0.f, 0.f);
    }
    if (gi < gDirBins)
    {
        const uint c = gHist[gMagBins + gi].load();
        gHistOut[uint2(gi, 1)] = float4(float(c) * inv, float(c), 0.f, 0.f);
    }
}
