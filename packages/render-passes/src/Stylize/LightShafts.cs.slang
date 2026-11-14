/**
 * LightShafts (see LightShafts.ts). Screen-space radial light scattering (Mitchell, "Volumetric Light Scattering as a
 * Post-Process", GPU Gems 3, ch. 13): march `samples` steps from the pixel towards the sun's screen position, step =
 * (sun - uv) * density / samples, accumulating the occlusion mask (bright sky: luminance above threshold, geometry = 0)
 * with exponential decay. out = src + exposure * sunFacing * tint * accum.
 */

cbuffer CB
{
    uint2 gFrameDim;
    float2 gSunUV;
    float3 gTint;
    float gSunFacing;
    uint gSamples;
    float gDensity;
    float gDecay;
    float gWeight;
    float gExposure;
    float gThreshold;
};

Texture2D<float4> gSrc;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;

float occluder(float2 uv)
{
    const int2 q = clamp(int2(uv * float2(gFrameDim)), int2(0), int2(gFrameDim) - 1);
    if (gPosW[q].w != 0.f) return 0.f;
    const float lum = dot(gSrc[q].rgb, float3(0.2126f, 0.7152f, 0.0722f));
    return max(lum - gThreshold, 0.f);
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    if (gExposure == 0.f || gSunFacing <= 0.f) { gDst[p] = src; return; }
    float2 uv = (float2(p) + 0.5f) / float2(gFrameDim);
    const float2 step = (gSunUV - uv) * (gDensity / float(gSamples));
    float illum = 1.f, acc = 0.f;
    for (uint i = 0; i < gSamples; i++)
    {
        uv += step;
        if (any(uv < 0.f) || any(uv > 1.f)) break;
        acc += occluder(uv) * illum * gWeight;
        illum *= gDecay;
    }
    gDst[p] = float4(src.rgb + gExposure * gSunFacing * acc * gTint, src.a);
}
