/**
 * GradientFog (see GradientFog.ts). Distance d = |posW - camera|; optical depth
 *   tau = density * integral_{start}^{d} exp(-k (h(t) - h0)) dt,   h(t) = up . (cam + t dir)   (k = heightFalloff; k = 0: d - start)
 * f = maxOpacity * (1 - exp(-tau)); fog colour = piecewise-linear ramp over distance through 2-4 stops;
 * out = lerp(src, colour(d), f). Background (posW.w = 0): skyMode 0 = fog with the last stop colour at maxOpacity, 1 = keep.
 */

static const uint kMaxStops = 4;

cbuffer CB
{
    uint2 gFrameDim;
    uint gStopCount;
    uint gSkyMode;
    float4 gStops[kMaxStops];   ///< rgb = colour (linear), a = distance (ascending)
    float3 gCamPos;
    float gDensity;
    float3 gUp;
    float gStart;
    float gHeightFalloff;
    float gHeightRef;
    float gMaxOpacity;
};

Texture2D<float4> gSrc;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;

float3 rampColour(float d)
{
    if (d <= gStops[0].a) return gStops[0].rgb;
    for (uint i = 1; i < gStopCount; i++)
        if (d <= gStops[i].a)
            return lerp(gStops[i - 1].rgb, gStops[i].rgb, (d - gStops[i - 1].a) / max(gStops[i].a - gStops[i - 1].a, 1e-6f));
    return gStops[gStopCount - 1].rgb;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    const float4 pos = gPosW[p];
    if (gDensity == 0.f || gMaxOpacity == 0.f) { gDst[p] = src; return; }
    if (pos.w == 0.f)
    {
        gDst[p] = gSkyMode == 1 ? src : float4(lerp(src.rgb, gStops[gStopCount - 1].rgb, gMaxOpacity), src.a);
        return;
    }
    const float3 v = pos.xyz - gCamPos;
    const float d = length(v);
    const float L = max(d - gStart, 0.f);
    float tau = gDensity * L;
    if (gHeightFalloff > 0.f && L > 0.f)
    {
        const float k = gHeightFalloff;
        const float u = dot(v / max(d, 1e-6f), gUp);         // vertical rate along the ray
        const float hs = dot(gCamPos, gUp) + u * gStart - gHeightRef; // height above h0 where fog starts
        const float ku = k * u * L;
        // integral_0^L exp(-k (hs + u t)) dt = exp(-k hs) * (1 - exp(-k u L)) / (k u); series for small k u L
        const float g = abs(ku) < 1e-3f ? L * (1.f - 0.5f * ku) : (1.f - exp(-ku)) / (k * u);
        tau = gDensity * exp(-k * hs) * g;
    }
    const float f = gMaxOpacity * (1.f - exp(-tau));
    gDst[p] = float4(lerp(src.rgb, rampColour(d), f), src.a);
}
