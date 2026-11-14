/**
 * LightCookiePattern: procedural light cookies (see LightCookiePattern.ts): drifting cloud shadows, window frames and
 * dappled light through foliage. One entry point; every texel is a pure function of its uv and the parameters.
 *
 * Cloud shadows after the scrolling-noise idea of PlayCanvas's cloud-shadows example (MIT,
 * https://github.com/playcanvas/engine); hash: PCG (Jarzynski & Olano 2020). Written for this pass.
 */
cbuffer CB
{
    uint2 gDim;
    uint gMode;          ///< 0 clouds, 1 window, 2 leaves
    uint gSeed;
    float3 gShadowColor;
    float gStrength;
    float2 gWind;        ///< uv per second (clouds)
    float gTime;
    float gCoverage;
    float gSoftness;
    uint gScale;         ///< clouds: base cells per tile; leaves: cells per tile
    uint gOctaves;
    float gSpotSize;     ///< leaves: spot radius in cells
    float gSway;         ///< leaves: sway amplitude in cells
    float gSwaySpeed;    ///< leaves: sway cycles per second
    float4 gRect;        ///< window: x0, y0, x1, y1 (uv)
    uint2 gPanes;        ///< window: columns, rows
    float gFrame;        ///< window: frame width (fraction of the window)
    float gMullion;      ///< window: mullion width (fraction of the window)
};

RWTexture2D<float4> gDst;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float hash2(uint2 c, uint k)
{
    return float(pcg(c.x + pcg(c.y + pcg(gSeed * 4u + k))) >> 8) * (1.f / 16777216.f);
}

uint2 wrapCell(int2 c, uint period)
{
    const int p = int(period);
    return uint2(((c % p) + p) % p);
}

/** Periodic value noise in [0, 1] (period in cells), quintic interpolation. */
float valueNoise(float2 p, uint period, uint octave)
{
    const float2 f0 = floor(p);
    const int2 i = int2(f0);
    const float2 f = p - f0;
    const float2 u = f * f * f * (f * (f * 6.f - 15.f) + 10.f);
    const float a = hash2(wrapCell(i, period), octave), b = hash2(wrapCell(i + int2(1, 0), period), octave);
    const float c = hash2(wrapCell(i + int2(0, 1), period), octave), d = hash2(wrapCell(i + int2(1, 1), period), octave);
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float clouds(float2 uv)
{
    float2 q = uv - gWind * gTime;
    q -= floor(q);
    float n = 0.f, amp = 1.f, sum = 0.f;
    uint period = max(gScale, 1u);
    for (uint o = 0; o < gOctaves; o++)
    {
        n += amp * valueNoise(q * float(period), period, o);
        sum += amp;
        amp *= 0.5f;
        period *= 2u;
    }
    n /= max(sum, 1e-6f);
    const float t = 1.f - gCoverage;
    return smoothstep(t - gSoftness, t + gSoftness, n);
}

float window(float2 uv)
{
    const float2 r = (uv - gRect.xy) / max(gRect.zw - gRect.xy, float2(1e-6f));
    if (any(r < 0.f) || any(r > 1.f)) return 1.f;
    const float s = gSoftness;
    const float edge = min(min(r.x, 1.f - r.x), min(r.y, 1.f - r.y));
    float open = smoothstep(gFrame - s, gFrame + s, edge);
    for (uint ax = 0; ax < 2; ax++)
    {
        const uint n = gPanes[ax];
        if (n < 2) continue;
        const float t = r[ax] * float(n);
        const float k = clamp(round(t), 1.f, float(n - 1));
        const float d = abs(t - k) / float(n);
        open *= smoothstep(0.5f * gMullion - s, 0.5f * gMullion + s, d);
    }
    return 1.f - open;
}

float leaves(float2 uv)
{
    const uint N = max(gScale, 1u);
    const float2 p = uv * float(N);
    const int2 c = int2(floor(p));
    float light = 0.f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            const int2 cc = c + int2(dx, dy);
            const uint2 w = wrapCell(cc, N);
            const float h1 = hash2(w, 11u), h2 = hash2(w, 12u), h3 = hash2(w, 13u), h4 = hash2(w, 14u);
            const float ph = 6.2831853f * (gTime * gSwaySpeed + h3);
            const float2 centre = float2(cc) + 0.5f + 0.8f * (float2(h1, h2) - 0.5f) + gSway * float2(sin(ph), cos(1.3f * ph));
            const float rad = gSpotSize * (0.5f + 0.5f * h4);
            const float d = length(p - centre);
            light = max(light, 1.f - smoothstep(rad * (1.f - gSoftness), rad, d));
        }
    return 1.f - light;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gDim)) return;
    const float2 uv = (float2(p) + 0.5f) / float2(gDim);
    float occ = 0.f;
    if (gStrength != 0.f)
    {
        if (gMode == 0) occ = clouds(uv);
        else if (gMode == 1) occ = window(uv);
        else occ = leaves(uv);
    }
    const float a = gStrength * occ;
    gDst[p] = a == 0.f ? float4(1.f) : float4(lerp(float3(1.f), gShadowColor, a), 1.f);
}
