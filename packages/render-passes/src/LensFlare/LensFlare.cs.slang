/**
 * LensFlare: screen-space lens flare (ghosts, halo, anamorphic streak) from the bright pixels of the frame
 * (see LensFlare.ts). Entry points, in frame order:
 *  prefilter (half res) : 2x2 box downsample of the source (or the `bright` input) + soft-knee threshold
 *  flare     (half res) : ghosts + halo (mirrored through the image centre, chromatic offset) + horizontal streak
 *  composite (full res) : dst = src + intensity x bilinear(flare)
 *
 * Approach after John Chapman's "pseudo lens flare" (as used e.g. by Filament's flare material, Apache-2.0) and
 * KinoStreak's horizontal streak (MIT, (c) 2017 Keijiro Takahashi); this code is written for the pass and copies neither.
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint2 gHalfDim;
    uint gHasBright;
    float gThreshold;
    float gKnee;
    float gIntensity;
    uint gGhostCount;
    float gGhostSpacing;
    float gGhostIntensity;
    float gHaloRadius;
    float3 gGhostTint;
    float gHaloFalloff;
    float3 gHaloTint;
    float gHaloIntensity;
    float3 gStreakTint;
    float gStreakIntensity;
    float gStreakLength;
    float gChromatic;
    uint gHasFlareOut;
};

static const uint kStreakTaps = 24;

Texture2D<float4> gSrc;
Texture2D<float4> gBright;
Texture2D<float4> gBrightHalf;  ///< prefilter output (flare pass input)
Texture2D<float4> gFlareHalf;   ///< flare output (composite input)
RWTexture2D<float4> gHalfOut;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gFlareOut;

struct Bil
{
    uint2 lo, hi;
    float2 f;
};

/** Bilinear taps with clamp-to-edge addressing (explicit, so the result does not depend on sampler state). */
Bil bilTaps(uint2 dim, float2 uv)
{
    const float2 x = uv * float2(dim) - 0.5f;
    const float2 i = floor(x);
    Bil b;
    b.f = x - i;
    b.lo = uint2(clamp(int2(i), int2(0), int2(dim) - 1));
    b.hi = uint2(clamp(int2(i) + 1, int2(0), int2(dim) - 1));
    return b;
}

float3 bilinearBright(float2 uv)
{
    const Bil b = bilTaps(gHalfDim, uv);
    const float3 x0 = lerp(gBrightHalf[uint2(b.lo.x, b.lo.y)].rgb, gBrightHalf[uint2(b.hi.x, b.lo.y)].rgb, b.f.x);
    const float3 x1 = lerp(gBrightHalf[uint2(b.lo.x, b.hi.y)].rgb, gBrightHalf[uint2(b.hi.x, b.hi.y)].rgb, b.f.x);
    return lerp(x0, x1, b.f.y);
}

float3 bilinearFlare(float2 uv)
{
    const Bil b = bilTaps(gHalfDim, uv);
    const float3 x0 = lerp(gFlareHalf[uint2(b.lo.x, b.lo.y)].rgb, gFlareHalf[uint2(b.hi.x, b.lo.y)].rgb, b.f.x);
    const float3 x1 = lerp(gFlareHalf[uint2(b.lo.x, b.hi.y)].rgb, gFlareHalf[uint2(b.hi.x, b.hi.y)].rgb, b.f.x);
    return lerp(x0, x1, b.f.y);
}

/** Linear lookup of the bright image along row y at continuous x; zero outside the image. */
float3 rowSample(float x, uint y)
{
    const float xc = x - 0.5f;
    const float i = floor(xc);
    const float f = xc - i;
    const int i0 = int(i), i1 = i0 + 1;
    float3 a = float3(0.f), b = float3(0.f);
    if (i0 >= 0 && i0 < int(gHalfDim.x)) a = gBrightHalf[uint2(i0, y)].rgb;
    if (i1 >= 0 && i1 < int(gHalfDim.x)) b = gBrightHalf[uint2(i1, y)].rgb;
    return lerp(a, b, f);
}

float3 sampleCA(float2 uv, float2 off)
{
    return float3(bilinearBright(uv + off).r, bilinearBright(uv).g, bilinearBright(uv - off).b);
}

[numthreads(16, 16, 1)]
void prefilter(uint3 tid: SV_DispatchThreadID)
{
    const uint2 q = tid.xy;
    if (any(q >= gHalfDim)) return;
    float3 c = float3(0.f);
    for (uint j = 0; j < 2; j++)
    {
        for (uint i = 0; i < 2; i++)
        {
            const uint2 p = min(2 * q + uint2(i, j), gFrameDim - 1);
            c += gHasBright != 0 ? gBright[p].rgb : gSrc[p].rgb;
        }
    }
    c = max(c * 0.25f, float3(0.f));
    const float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - gThreshold + gKnee, 0.f, 2.f * gKnee);
    soft = soft * soft / (4.f * gKnee + 1e-5f);
    const float w = max(soft, br - gThreshold) / max(br, 1e-5f);
    gHalfOut[q] = float4(c * w, 1.f);
}

[numthreads(16, 16, 1)]
void flare(uint3 tid: SV_DispatchThreadID)
{
    const uint2 q = tid.xy;
    if (any(q >= gHalfDim)) return;
    const float2 uv = (float2(q) + 0.5f) / float2(gHalfDim);
    const float2 fuv = 1.f - uv; // mirrored through the image centre
    const float2 gv = (0.5f - fuv) * gGhostSpacing;
    const float gl = length(gv);
    const float2 gdir = gl > 1e-6f ? gv / gl : float2(0.f);
    const float2 ca = gdir * gChromatic;

    float3 ghosts = float3(0.f);
    for (uint i = 0; i < gGhostCount; i++)
    {
        const float2 suv = fuv + gv * float(i);
        if (any(suv < 0.f) || any(suv > 1.f)) continue;
        const float w = saturate(1.f - length(0.5f - suv) / 0.70710678f);
        ghosts += sampleCA(suv, ca) * (w * w);
    }

    const float2 huv = fuv + gdir * gHaloRadius;
    float3 halo = float3(0.f);
    if (gl > 1e-6f && all(huv >= 0.f) && all(huv <= 1.f))
        halo = sampleCA(huv, ca) * pow(saturate(1.f - length(0.5f - huv) / 0.70710678f), gHaloFalloff);

    float3 streak = float3(0.f);
    if (gStreakIntensity != 0.f)
    {
        const float stride = max(gStreakLength * float(gHalfDim.x) / float(kStreakTaps), 1.f);
        float wsum = 0.f;
        for (int j = -int(kStreakTaps); j <= int(kStreakTaps); j++)
        {
            const float w = exp(-4.f * abs(float(j)) / float(kStreakTaps));
            streak += w * rowSample(float(q.x) + 0.5f + float(j) * stride, q.y);
            wsum += w;
        }
        streak /= wsum;
    }
    gHalfOut[q] = float4(ghosts * gGhostIntensity * gGhostTint + halo * gHaloIntensity * gHaloTint + streak * gStreakIntensity * gStreakTint, 1.f);
}

[numthreads(16, 16, 1)]
void composite(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    if (gIntensity == 0.f)
    {
        gDst[p] = src;
        if (gHasFlareOut != 0) gFlareOut[p] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const float3 f = gIntensity * bilinearFlare((float2(p) + 0.5f) / float2(gFrameDim));
    gDst[p] = float4(src.rgb + f, src.a);
    if (gHasFlareOut != 0) gFlareOut[p] = float4(f, 1.f);
}
