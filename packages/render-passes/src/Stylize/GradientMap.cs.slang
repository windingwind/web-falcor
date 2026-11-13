/**
 * GradientMap (see GradientMap.ts). t = sRGB-encoded Rec. 709 luma of src; out = lerp(src, ramp(t), strength),
 * ramp = piecewise-linear through 2-6 colour stops (rgb linear, a = position).
 */
static const uint kMaxStops = 6;
cbuffer CB
{
    uint2 gFrameDim;
    uint gStopCount;
    float gStrength;
    float4 gStops[kMaxStops];
};
Texture2D<float4> gSrc;
RWTexture2D<float4> gDst;

float encodeSrgb(float x) { x = saturate(x); return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.f / 2.4f) - 0.055f; }

float3 ramp(float x)
{
    if (x <= gStops[0].a) return gStops[0].rgb;
    for (uint i = 1; i < gStopCount; i++)
        if (x <= gStops[i].a)
            return lerp(gStops[i - 1].rgb, gStops[i].rgb, (x - gStops[i - 1].a) / max(gStops[i].a - gStops[i - 1].a, 1e-6f));
    return gStops[gStopCount - 1].rgb;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    if (gStrength == 0.f) { gDst[p] = src; return; }
    const float t = encodeSrgb(dot(src.rgb, float3(0.2126f, 0.7152f, 0.0722f)));
    gDst[p] = float4(lerp(src.rgb, ramp(t), gStrength), src.a);
}
