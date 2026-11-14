/**
 * CRT (see CRT.ts). Barrel curvature, Gaussian scanline beams over `sourceLines` virtual lines (beam width from beamMax for
 * dark to beamMin for bright pixels, energy-normalised), a phosphor mask at output resolution (aperture grille, slot or dot
 * triads), a halation glow (box blur of the source over a few lines), vignette; out = lerp(src, crt, strength).
 */
cbuffer CB
{
    uint2 gFrameDim;
    float gSourceLines;
    float gBeamMin;
    float gBeamMax;
    uint gMask;          ///< 0 none, 1 aperture, 2 slot, 3 dot
    float gMaskStrength;
    float gCurvature;
    float gHalation;
    float gVignette;
    float gBrightness;
    float gStrength;
};
Texture2D<float4> gSrc;
RWTexture2D<float4> gDst;

float3 fetch(float2 uv)
{
    const int2 q = clamp(int2(uv * float2(gFrameDim)), int2(0), int2(gFrameDim) - 1);
    return gSrc[q].rgb;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    const float4 src = gSrc[p];
    if (gStrength == 0.f) { gDst[p] = src; return; }
    float2 uv = (float2(p) + 0.5f) / float2(gFrameDim);
    float2 c = uv * 2.f - 1.f;
    c *= 1.f + gCurvature * dot(c, c) * 0.25f;
    uv = c * 0.5f + 0.5f;
    float3 col = 0.f;
    if (all(uv >= 0.f) && all(uv <= 1.f))
    {
        // scanlines: the two nearest source lines contribute Gaussian beams
        const float y = uv.y * gSourceLines - 0.5f;
        const float y0 = floor(y);
        float3 acc = 0.f;
        for (int k = 0; k <= 1; k++)
        {
            const float line = y0 + float(k);
            const float3 c3 = fetch(float2(uv.x, (line + 0.5f) / gSourceLines));
            const float lum = saturate(dot(c3, float3(0.2126f, 0.7152f, 0.0722f)));
            const float w = lerp(gBeamMax, gBeamMin, sqrt(lum));
            const float d = (y - line) / w;
            acc += c3 * exp(-0.5f * d * d) / (w * 2.5066283f);
        }
        col = acc;
        // phosphor mask
        if (gMask != 0)
        {
            uint ph = uint(p.x) % 3u;
            if (gMask == 2 && ((uint(p.y) / 2u + uint(p.x) / 3u) & 1u) != 0) ph = (ph + 1u) % 3u;   // slot: staggered rows
            if (gMask == 3) ph = (uint(p.x) + uint(p.y)) % 3u;                                       // dot: diagonal triads
            float3 m = float3(ph == 0u, ph == 1u, ph == 2u) * 3.f;
            col *= lerp(float3(1.f), m, gMaskStrength);
        }
        // halation: wide glow from a few source lines around
        if (gHalation > 0.f)
        {
            float3 g = 0.f;
            for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++) g += fetch(uv + float2(i, j) * (2.f / float(gFrameDim.y)));
            col += gHalation * g / 49.f;
        }
        const float2 v = uv * (1.f - uv);
        col *= lerp(1.f, saturate(pow(v.x * v.y * 16.f, 0.3f)), gVignette) * gBrightness;
    }
    gDst[p] = float4(lerp(src.rgb, col, gStrength), src.a);
}
