/**
 * RampShade (see RampShade.ts). x = half-Lambert (0.5 N.L + 0.5) or Lambert (saturate(N.L)), times the optional
 * visibility; x = x^gamma; ramp(x) = piecewise-linear RGB ramp through up to 6 stops, each segment sharpened by `hardness`;
 * shaded = albedo * (ramp(x) * overbright + ambient); out = lerp(src, shaded, mix) on geometry pixels.
 */

static const uint kMaxStops = 6;

cbuffer CB
{
    uint2 gFrameDim;
    uint gMode;           ///< 0 = half_lambert, 1 = lambert
    uint gStopCount;
    float4 gStops[kMaxStops]; ///< rgb = colour (linear), a = position in [0, 1], ascending
    float3 gLightDir;
    float gGamma;
    float gHardness;
    float gOverbright;
    float gAmbient;
    float gMix;
    uint gHasVisibility;
};

Texture2D<float4> gSrc;
Texture2D<float4> gAlbedo;
Texture2D<float4> gNormW;
Texture2D<float4> gPosW;
Texture2D<float4> gVisibility;
RWTexture2D<float4> gDst;

float3 evalRamp(float x)
{
    if (x <= gStops[0].a) return gStops[0].rgb;
    for (uint i = 1; i < gStopCount; i++)
    {
        if (x <= gStops[i].a)
        {
            const float w = max(gStops[i].a - gStops[i - 1].a, 1e-6f);
            float t = (x - gStops[i - 1].a) / w;
            // hardness 0: linear; 1: step at the segment midpoint
            t = saturate((t - 0.5f) / max(1.f - gHardness, 1e-4f) + 0.5f);
            return lerp(gStops[i - 1].rgb, gStops[i].rgb, t);
        }
    }
    return gStops[gStopCount - 1].rgb;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    if (gPosW[p].w == 0.f || gMix == 0.f)
    {
        gDst[p] = src;
        return;
    }
    const float ndl = dot(normalize(gNormW[p].xyz), gLightDir);
    float x = gMode == 0 ? 0.5f * ndl + 0.5f : saturate(ndl);
    if (gHasVisibility != 0) x *= saturate(gVisibility[p].r);
    x = pow(saturate(x), gGamma);
    const float3 albedo = gAlbedo[p].rgb;
    const float3 shaded = albedo * (evalRamp(x) * gOverbright + gAmbient);
    gDst[p] = float4(lerp(src.rgb, shaded, gMix), src.a);
}
