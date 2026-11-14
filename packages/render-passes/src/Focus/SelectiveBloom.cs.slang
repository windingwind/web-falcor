/**
 * SelectiveBloom: bloom restricted to selected objects (see SelectiveBloom.ts). Entry points, in frame order:
 *  prefilter (half res) : 2x2 box downsample of src x selection, then a soft-knee threshold
 *  blurH / blurV        : separable Gaussian (65 taps, stride grows with sigma, linear row/column lookups)
 *  composite (full res) : dst = src + intensity x tint x bilinear(glow)
 *
 * The idea of masking the bloom input by an object selection follows pmndrs/postprocessing's SelectiveBloomEffect
 * (https://github.com/pmndrs/postprocessing, zlib licence, Copyright (c) 2015 Raoul van Rüschen); the code here is
 * written for this pass and does not copy it.
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint2 gHalfDim;
    uint4 gIds[8];      ///< selected ids, 4 per entry (gIdCount used)
    uint gIdCount;
    uint gHasId;
    uint gHasMask;
    uint gInvert;
    float gThreshold;
    float gKnee;
    float gIntensity;
    float gSigma;       ///< blur sigma in half-resolution pixels
    float3 gTint;
    uint gHasGlowOut;
};

static const int kTaps = 32;

Texture2D<float4> gSrc;
Texture2D<float4> gId;      ///< StyleAovs.id (raw id in g)
Texture2D<float4> gMask;    ///< optional selection weight in r
Texture2D<float4> gIn;      ///< blur input / composite glow
RWTexture2D<float4> gOut;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gGlowOut;

bool idSelected(uint id)
{
    for (uint i = 0; i < gIdCount; i++)
        if (gIds[i >> 2][i & 3] == id) return true;
    return false;
}

/** Selection weight of a full-resolution pixel: id membership x mask; nothing selected without id and mask inputs. */
float selection(uint2 p)
{
    if (gHasId == 0 && gHasMask == 0) return 0.f;
    float s = 1.f;
    if (gHasId != 0)
    {
        const float4 v = gId[p];
        const bool hit = v.a != 0.f && idSelected(uint(v.g + 0.5f));
        s = hit ? 1.f : 0.f;
    }
    if (gHasMask != 0) s *= saturate(gMask[p].r);
    return gInvert != 0 ? 1.f - s : s;
}

[numthreads(16, 16, 1)]
void prefilter(uint3 tid: SV_DispatchThreadID)
{
    const uint2 q = tid.xy;
    if (any(q >= gHalfDim)) return;
    float3 c = 0.f;
    for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 2; i++)
        {
            const uint2 p = min(2 * q + uint2(i, j), gFrameDim - 1);
            c += gSrc[p].rgb * selection(p);
        }
    c = max(c * 0.25f, 0.f);
    const float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - gThreshold + gKnee, 0.f, 2.f * gKnee);
    soft = soft * soft / (4.f * gKnee + 1e-5f);
    const float w = max(soft, br - gThreshold) / max(br, 1e-5f);
    gOut[q] = float4(c * w, 1.f);
}

/** Linear lookup at continuous coordinate x along the axis (clamp to edge). */
float3 axisSample(float x, uint2 q, bool horiz)
{
    const uint n = horiz ? gHalfDim.x : gHalfDim.y;
    const float xc = x - 0.5f;
    const float f0 = floor(xc);
    const float f = xc - f0;
    const int i0 = clamp(int(f0), 0, int(n) - 1), i1 = clamp(int(f0) + 1, 0, int(n) - 1);
    const uint2 a = horiz ? uint2(uint(i0), q.y) : uint2(q.x, uint(i0));
    const uint2 b = horiz ? uint2(uint(i1), q.y) : uint2(q.x, uint(i1));
    return lerp(gIn[a].rgb, gIn[b].rgb, f);
}

float3 blur(uint2 q, bool horiz)
{
    const float stride = max(3.f * gSigma / float(kTaps), 1.f);
    const float x0 = (horiz ? float(q.x) : float(q.y)) + 0.5f;
    float3 acc = 0.f;
    float ws = 0.f;
    for (int j = -kTaps; j <= kTaps; j++)
    {
        const float d = float(j) * stride;
        const float w = exp(-0.5f * d * d / (gSigma * gSigma));
        acc += w * axisSample(x0 + d, q, horiz);
        ws += w;
    }
    return acc / ws;
}

[numthreads(16, 16, 1)]
void blurH(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gHalfDim)) return;
    gOut[tid.xy] = float4(blur(tid.xy, true), 1.f);
}

[numthreads(16, 16, 1)]
void blurV(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gHalfDim)) return;
    gOut[tid.xy] = float4(blur(tid.xy, false), 1.f);
}

float3 bilinearGlow(float2 uv)
{
    const float2 x = uv * float2(gHalfDim) - 0.5f;
    const float2 i = floor(x);
    const float2 f = x - i;
    const uint2 lo = uint2(clamp(int2(i), int2(0), int2(gHalfDim) - 1));
    const uint2 hi = uint2(clamp(int2(i) + 1, int2(0), int2(gHalfDim) - 1));
    const float3 x0 = lerp(gIn[uint2(lo.x, lo.y)].rgb, gIn[uint2(hi.x, lo.y)].rgb, f.x);
    const float3 x1 = lerp(gIn[uint2(lo.x, hi.y)].rgb, gIn[uint2(hi.x, hi.y)].rgb, f.x);
    return lerp(x0, x1, f.y);
}

[numthreads(16, 16, 1)]
void composite(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 s = gSrc[p];
    if (gIntensity == 0.f)
    {
        gDst[p] = s;
        if (gHasGlowOut != 0) gGlowOut[p] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const float3 g = gIntensity * gTint * bilinearGlow((float2(p) + 0.5f) / float2(gFrameDim));
    gDst[p] = float4(s.rgb + g, s.a);
    if (gHasGlowOut != 0) gGlowOut[p] = float4(g, 1.f);
}
