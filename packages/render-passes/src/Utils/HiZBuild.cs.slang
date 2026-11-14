/**
 * HiZBuild: min/max depth pyramid (see HiZBuild.ts). Output texels are float2(min, max).
 *  copyMip0 : mip 0 = (d, d) with d = the selected channel of the input.
 *  reduce   : mip k from mip k-1. Texel (x, y) covers source texels [2x, 2x+1] x [2y, 2y+1], extended by the last
 *             source column/row when the source size is odd and the texel is the last one (so every source texel is
 *             covered by exactly the texels that contain it and min/max are conservative for any size).
 */
cbuffer CB
{
    uint2 gDstDim;
    uint2 gSrcDim;
    uint gChannel;
};

Texture2D<float4> gDepth;
Texture2D<float2> gSrcMip;
RWTexture2D<float2> gDstMip;

[numthreads(16, 16, 1)]
void copyMip0(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gDstDim)) return;
    const float d = gDepth[p][gChannel];
    gDstMip[p] = float2(d, d);
}

[numthreads(16, 16, 1)]
void reduce(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gDstDim)) return;
    const uint2 base = p * 2;
    const uint2 last = gSrcDim - 1;
    // Footprint end: 2x+1, or 2x+2 for the last texel of an odd source.
    const uint2 end = min(base + uint2(p.x == gDstDim.x - 1 && (gSrcDim.x & 1) != 0 ? 2 : 1, p.y == gDstDim.y - 1 && (gSrcDim.y & 1) != 0 ? 2 : 1), last);
    float lo = gSrcMip[min(base, last)].x;
    float hi = gSrcMip[min(base, last)].y;
    for (uint y = base.y; y <= end.y; y++)
    {
        for (uint x = base.x; x <= end.x; x++)
        {
            const float2 v = gSrcMip[uint2(x, y)];
            lo = min(lo, v.x);
            hi = max(hi, v.y);
        }
    }
    gDstMip[p] = float2(lo, hi);
}
