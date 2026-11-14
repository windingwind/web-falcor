/**
 * RainWetness: wet surfaces and puddles on the lit G-buffer (see RainWetness.ts). One entry point, one dispatch.
 *
 * Per pixel with geometry (posW.w != 0), with N = normalize(normW), u = dot(N, up):
 *  wet    = wetness x facing(u), facing = lerp(sideWetness, 1, smoothstep(0, 1, u)) for u >= 0, sideWetness x (1 + u) below;
 *  puddle = smoothstep(puddleUpMin, 1, u) x smoothstep(1 - coverage, 1 - coverage + puddleEdge, fbm(planar x puddleScale));
 *  ripples (puddles only): rings on a planar grid (cell rippleScale, one ring per cell and layer, 2 layers), a pure
 *           function of `time` and the seed; height h = A sin(k (d - r)) exp(-((d - r)/w)^2) (1 - tau), slope from dh/dd.
 *  w      = max(wet, puddle)
 *  dst.rgb   = color - darkening x w x diffuse - puddleSpecular x puddle x specular   (clamped at 0; alpha kept)
 *  specRough = (lerp(F0, 0.02, puddle), lerp(roughness x (1 - gloss x wet), puddleRoughness, puddle))
 *  normW     = normalize(lerp(N, up, puddle) - rippleStrength x puddle x slope) (w kept)
 *  wetness   = (wet, puddle, ripple height, 1)
 * Pixels with w == 0 (and the background) copy their inputs bit for bit.
 *
 * Written for this pass. Ideas: wet-surface darkening and gloss after S. Lagarde, "Water drop 2b - Dynamic rain and its
 * effects" (2013 blog series); no code copied.
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gHasDiffuse;
    uint gHasSpecular;
    float3 gUp;
    float gWetness;
    float3 gT1;              ///< planar basis (orthonormal to gUp)
    float gSideWetness;
    float3 gT2;
    float gDarkening;
    float gGloss;
    float gPuddleCoverage;
    float gPuddleEdge;
    float gPuddleScale;
    float gPuddleUpMin;
    float gPuddleRoughness;
    float gPuddleSpecular;
    float gRippleStrength;
    float gRippleScale;
    float gRippleRate;
    float gTime;
    uint gSeed;
};

Texture2D<float4> gColor;
Texture2D<float4> gDiffuse;
Texture2D<float4> gSpecular;
Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gSpecRough;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gSpecRoughOut;
RWTexture2D<float4> gNormWOut;
RWTexture2D<float4> gWetnessOut;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/** Uniform [0, 1) from an integer lattice point and a salt. */
float hash2(int2 c, uint salt)
{
    return float(pcg(uint(c.x) * 73856093u ^ pcg(uint(c.y) * 19349663u ^ pcg(gSeed * 83492791u + salt))) >> 8) * (1.f / 16777216.f);
}

float valueNoise(float2 x, uint salt)
{
    const float2 i = floor(x);
    const float2 f = x - i;
    const float2 s = f * f * (3.f - 2.f * f);
    const int2 c = int2(i);
    const float a = hash2(c, salt), b = hash2(c + int2(1, 0), salt);
    const float cc = hash2(c + int2(0, 1), salt), d = hash2(c + int2(1, 1), salt);
    return lerp(lerp(a, b, s.x), lerp(cc, d, s.x), s.y);
}

/** 4-octave fBm in [0, 1). */
float fbm(float2 x)
{
    float v = 0.f, a = 0.5f, n = 0.f;
    for (uint o = 0; o < 4u; o++)
    {
        v += a * valueNoise(x, o);
        n += a;
        a *= 0.5f;
        x = x * 2.03f + float2(17.1f, 9.7f);
    }
    return v / n;
}

float smoothstep01(float e0, float e1, float x)
{
    const float t = saturate((x - e0) / max(e1 - e0, 1e-6f));
    return t * t * (3.f - 2.f * t);
}

/** Ripple rings: height (x) and planar slope (yz) at planar position q (world units). */
float3 ripples(float2 q)
{
    float3 acc = float3(0.f);
    const float cell = gRippleScale;
    for (uint layer = 0; layer < 2u; layer++)
    {
        const float2 g = q / cell + float2(0.37f, 0.71f) * float(layer);
        const int2 c0 = int2(floor(g));
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                const int2 c = c0 + int2(dx, dy);
                const uint salt = 100u + 10u * layer;
                const float2 centre = float2(c) + float2(hash2(c, salt), hash2(c, salt + 1u));
                const float tau = frac(gTime * gRippleRate + hash2(c, salt + 2u));
                const float2 o = (g - centre) * cell;
                const float d = length(o);
                const float r = tau * cell;               // ring radius grows to one cell over the period
                const float wdt = 0.08f * cell;
                const float k = 6.2831853f / (0.12f * cell);
                const float x = d - r;
                const float env = exp(-(x * x) / (wdt * wdt)) * (1.f - tau);
                const float h = sin(k * x) * env;
                const float dh = (k * cos(k * x) - 2.f * x / (wdt * wdt) * sin(k * x)) * env;
                const float2 dir = d > 1e-6f ? o / d : float2(0.f);
                acc += float3(h, dh * dir) * 0.08f * cell;
            }
        }
    }
    return acc;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 col = gColor[p];
    const float4 nw = gNormW[p];
    const float4 sr = gSpecRough[p];
    const float4 pos = gPosW[p];
    float wet = 0.f, puddle = 0.f;
    float3 rip = float3(0.f);
    float3 N = float3(0.f, 0.f, 1.f);
    float2 q = float2(0.f);
    if (pos.w != 0.f && dot(nw.xyz, nw.xyz) > 0.f)
    {
        N = normalize(nw.xyz);
        const float u = dot(N, gUp);
        const float facing = u >= 0.f ? lerp(gSideWetness, 1.f, smoothstep01(0.f, 1.f, u)) : gSideWetness * saturate(1.f + u);
        wet = saturate(gWetness * facing);
        q = float2(dot(pos.xyz, gT1), dot(pos.xyz, gT2));
        if (gPuddleCoverage > 0.f)
        {
            const float n = fbm(q * gPuddleScale);
            const float lo = 1.f - gPuddleCoverage;
            puddle = smoothstep01(gPuddleUpMin, 1.f, u) * smoothstep01(lo, lo + gPuddleEdge, n);
        }
    }
    const float w = max(wet, puddle);
    if (w == 0.f)
    {
        gDst[p] = col;
        gSpecRoughOut[p] = sr;
        gNormWOut[p] = nw;
        gWetnessOut[p] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const float3 diff = gHasDiffuse != 0 ? gDiffuse[p].rgb : col.rgb;
    const float3 spec = gHasSpecular != 0 ? gSpecular[p].rgb : float3(0.f);
    const float3 rgb = max(col.rgb - gDarkening * w * diff - gPuddleSpecular * puddle * spec, 0.f);
    const float rough = lerp(sr.a * (1.f - gGloss * wet), gPuddleRoughness, puddle);
    const float3 f0 = lerp(sr.rgb, float3(0.02f), puddle);
    float3 n2 = lerp(N, gUp, puddle);
    if (puddle > 0.f && gRippleStrength > 0.f)
    {
        rip = ripples(q);
        n2 -= gRippleStrength * puddle * (rip.y * gT1 + rip.z * gT2);
    }
    gDst[p] = float4(rgb, col.a);
    gSpecRoughOut[p] = float4(f0, rough);
    gNormWOut[p] = float4(normalize(n2), nw.w);
    gWetnessOut[p] = float4(wet, puddle, rip.x, 1.f);
}
