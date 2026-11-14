/**
 * StylizedFocus: keep a focus region sharp and abstract the rest by blur and desaturation (see StylizedFocus.ts).
 * Entry points: blurH / blurV (separable Gaussian, full resolution, run for two sigmas) and composite.
 *
 * Idea after Cole et al. 2006, "Directing Gaze in 3D Models with Stylized Focus" (EGSR): an emphasis weight from the
 * distance to a focus drives the amount of abstraction. Only the blur and desaturation channels are implemented here;
 * the code is written for this pass.
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gMode;             ///< 0 depth, 1 point, 2 screen, 3 ids, 4 mask
    uint gHasMask;
    float3 gCamPos;
    float gFocusDistance;
    float3 gFocusPoint;
    float gFocusRange;
    float2 gScreenCenter;
    float gFalloff;
    float gBackgroundFocus;
    uint4 gIds[8];
    uint gIdCount;
    float gStrength;
    float gDesaturate;
    uint gHasBlur;
    float gSigma;           ///< blur sigma of this dispatch (pixels)
    uint gHasFocusOut;
};

static const int kTaps = 32;

Texture2D<float4> gSrc;
Texture2D<float4> gPosW;
Texture2D<float4> gId;
Texture2D<float4> gMask;
Texture2D<float4> gIn;      ///< blur input
Texture2D<float4> gBlur1;   ///< small blur (sigma / 3)
Texture2D<float4> gBlur2;   ///< full blur (sigma)
RWTexture2D<float4> gOut;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gFocusOut;

bool idSelected(uint id)
{
    for (uint i = 0; i < gIdCount; i++)
        if (gIds[i >> 2][i & 3] == id) return true;
    return false;
}

/** 1 = in focus, 0 = fully abstracted. */
float focusWeight(uint2 p)
{
    if (gMode == 4) return gHasMask != 0 ? saturate(gMask[p].r) : 1.f;
    if (gMode == 3)
    {
        const float4 v = gId[p];
        return (v.a != 0.f && idSelected(uint(v.g + 0.5f))) ? 1.f : 0.f;
    }
    if (gMode == 2)
    {
        const float2 uv = (float2(p) + 0.5f) / float2(gFrameDim);
        const float2 d = (uv - gScreenCenter) * float2(float(gFrameDim.x) / float(gFrameDim.y), 1.f);
        return 1.f - smoothstep(gFocusRange, gFocusRange + gFalloff, length(d));
    }
    const float4 pw = gPosW[p];
    if (pw.a == 0.f) return gBackgroundFocus;
    const float dist = gMode == 0 ? abs(length(pw.xyz - gCamPos) - gFocusDistance) : length(pw.xyz - gFocusPoint);
    return 1.f - smoothstep(gFocusRange, gFocusRange + gFalloff, dist);
}

float3 axisSample(float x, uint2 q, bool horiz)
{
    const uint n = horiz ? gFrameDim.x : gFrameDim.y;
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
    if (any(tid.xy >= gFrameDim)) return;
    gOut[tid.xy] = float4(blur(tid.xy, true), 1.f);
}

[numthreads(16, 16, 1)]
void blurV(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gFrameDim)) return;
    gOut[tid.xy] = float4(blur(tid.xy, false), 1.f);
}

[numthreads(16, 16, 1)]
void composite(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 s = gSrc[p];
    const float f = focusWeight(p);
    if (gHasFocusOut != 0) gFocusOut[p] = float4(f, f, f, 1.f);
    const float a = gStrength * (1.f - f);
    if (!(a > 0.f))
    {
        gDst[p] = s;
        return;
    }
    float3 c = s.rgb;
    if (gHasBlur != 0)
    {
        const float t = saturate(a);
        c = t < 0.5f ? lerp(s.rgb, gBlur1[p].rgb, 2.f * t) : lerp(gBlur1[p].rgb, gBlur2[p].rgb, 2.f * t - 1.f);
    }
    const float L = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    c = lerp(c, float3(L), saturate(a * gDesaturate));
    gDst[p] = float4(c, s.a);
}
