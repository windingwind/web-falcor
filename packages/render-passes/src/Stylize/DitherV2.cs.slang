/**
 * DitherV2 (see DitherV2.ts). Ordered dithering of the sRGB-encoded colour to `levels` levels per channel:
 * e' = floor(e * (levels - 1) + 0.5 + strength * t) / (levels - 1), t in (-0.5, 0.5) from the threshold pattern;
 * PS1 mode: e8 = 255 e + strength * M[y][x] (M = the 4x4 PS1 offset matrix, -4..3), then truncated to 5 bits.
 * Pattern coordinates: screen pixels / scale, view-direction sphere (yaw/pitch, stable under camera translation along
 * the view, sticks to the world under rotation) or world space (dominant-axis projection of posW).
 * out = lerp(src, dithered, mix).
 */

cbuffer CB
{
    uint2 gFrameDim;
    uint gPattern;      ///< 0 bayer2, 1 bayer4, 2 bayer8, 3 blue_noise, 4 ps1, 5 ign
    uint gSpace;        ///< 0 screen, 1 view_sphere, 2 world
    float gLevels;
    float gScale;       ///< screen: pixels per pattern cell
    float gAngScale;    ///< view_sphere: pattern cells per radian
    float gWorldScale;  ///< world: pattern cells per world unit
    float3 gCamPos;
    float gStrength;
    float gMix;
    uint gBlueSize;
    uint gHasPosW;
    uint gHasNormW;
};

Texture2D<float4> gSrc;
Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gBlueNoise;   ///< r = rank / N in [0, 1)
RWTexture2D<float4> gDst;

float encodeSrgb(float x) { x = saturate(x); return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.f / 2.4f) - 0.055f; }
float decodeSrgb(float x) { return x <= 0.04045f ? x / 12.92f : pow((x + 0.055f) / 1.055f, 2.4f); }

/** Bayer matrix of size 2^n (n <= 3): the finest coordinate bit is the most significant digit of the index. */
float bayer(uint2 c, uint n)
{
    uint v = 0;
    for (uint i = 0; i < n; i++)
    {
        const uint bx = (c.x >> i) & 1u, by = (c.y >> i) & 1u;
        v = (v << 2) | ((bx ^ by) << 1) | by;
    }
    const float N = float(1u << (2 * n));
    return (float(v) + 0.5f) / N - 0.5f;
}

float threshold(int2 c)
{
    const uint2 u = uint2(c & 0xffff);
    switch (gPattern)
    {
    case 0: return bayer(u, 1);
    case 1: return bayer(u, 2);
    case 2: return bayer(u, 3);
    case 3: return gBlueNoise[u % gBlueSize].r + 0.5f / float(gBlueSize * gBlueSize) - 0.5f;
    case 5: return frac(52.9829189f * frac(0.06711056f * float(c.x) + 0.00583715f * float(c.y))) - 0.5f; // Jimenez 2014 IGN
    default: return 0.f;
    }
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    const float4 src = gSrc[p];
    if (gMix == 0.f) { gDst[p] = src; return; }

    // pattern cell coordinates
    float2 cf = float2(p) / max(gScale, 1e-3f);
    const float4 pos = gHasPosW != 0 ? gPosW[p] : float4(0.f);
    if (gSpace == 1 && pos.w != 0.f)
    {
        const float3 dir = normalize(pos.xyz - gCamPos);
        cf = float2(atan2(dir.z, dir.x), asin(clamp(dir.y, -1.f, 1.f))) * gAngScale;
    }
    else if (gSpace == 2 && pos.w != 0.f)
    {
        const float3 n = gHasNormW != 0 ? abs(gNormW[p].xyz) : float3(0.f, 1.f, 0.f);
        const float2 uv = (n.x >= n.y && n.x >= n.z) ? pos.zy : ((n.y >= n.z) ? pos.xz : pos.xy);
        cf = uv * gWorldScale;
    }
    const int2 c = int2(floor(cf));

    float3 e = float3(encodeSrgb(src.r), encodeSrgb(src.g), encodeSrgb(src.b));
    if (gPattern == 4)
    {
        const int M[16] = { -4, 0, -3, 1, 2, -2, 3, -1, -3, 1, -4, 0, 3, -1, 2, -2 };
        const uint2 u = uint2(c & 3);
        const float off = gStrength * float(M[u.y * 4u + u.x]);
        e = floor(clamp(floor(e * 255.f + 0.5f) + off, 0.f, 255.f) / 8.f) * 8.f / 248.f;
        e = saturate(e);
    }
    else
    {
        const float L = max(gLevels - 1.f, 1.f);
        e = clamp(floor(e * L + 0.5f + gStrength * threshold(c)), 0.f, L) / L;
    }
    const float3 d = float3(decodeSrgb(e.r), decodeSrgb(e.g), decodeSrgb(e.b));
    gDst[p] = float4(lerp(src.rgb, d, gMix), src.a);
}
