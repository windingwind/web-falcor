/**
 * TemporalBlend: exponential blend of the current colour with the reprojected previous output (see TemporalBlend.ts).
 * out = frame has valid history and the reprojected pixel is on screen ? lerp(src, history, alpha) : src.
 * The history is fetched at the nearest pixel (no filtering), so the result is deterministic per pixel.
 */
cbuffer CB
{
    uint2 gFrameDim;
    float gAlpha;
    uint gHistoryValid;
    uint gHasMvec;
};

Texture2D<float4> gSrc;
Texture2D<float4> gMvec;
Texture2D<float4> gHistory;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gHistoryOut;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 c = gSrc[p];
    float4 o = c;
    if (gHistoryValid != 0 && gAlpha > 0.f)
    {
        float2 uv = (float2(p) + 0.5f) / float2(gFrameDim);
        if (gHasMvec != 0) uv += gMvec[p].xy; // mvec = previous minus current position in UV units
        if (all(uv >= 0.f) && all(uv < 1.f))
        {
            const uint2 q = min(uint2(uv * float2(gFrameDim)), gFrameDim - 1);
            o = lerp(c, gHistory[q], gAlpha);
        }
    }
    gDst[p] = o;
    gHistoryOut[p] = o;
}
