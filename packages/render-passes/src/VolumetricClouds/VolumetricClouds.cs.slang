/**
 * VolumetricClouds kernels (see VolumetricClouds.ts; its marchCloudRay() is the CPU reference of march()).
 *
 *   march   (low-res grid): per ray, the cloud slab interval (curved planet shell or flat slab), clamped to the scene
 *           surface and maxDistance, is marched in `steps` jittered steps; each sample with density > 0 marches
 *           `lightSteps` steps towards the light (Beer transmittance), adds the dual-lobe Henyey-Greenstein sun term
 *           with a powder factor and the height-blended sky ambient, and integrates energy-conserving.
 *   resolve (full res): upsamples the march, removes cloud in front of nearer scene surfaces, reprojects the history
 *           (cloud front point, wind-compensated, previous camera), clamps it to the current neighbourhood, blends,
 *           writes the new history and the outputs.
 *
 * Adapted in part from bevy-volumetric-clouds (https://github.com/evroon/bevy-volumetric-clouds, MIT License,
 * Copyright (c) 2025 evroon): the planet-shell slab intersection, the dual-lobe phase mix, the light march with a growing
 * step and the energy-conserving integration (after Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering
 * in Frostbite", 2016). Noise, weather map, height profiles, temporal resolve and compositing are written for
 * web-falcor. Permission notice of the adapted work:
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 *   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 *   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 *   permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
 *   the Software.
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 *   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

cbuffer CB
{
    uint4 gDims;        ///< W, H, lowW, lowH
    uint4 gFlags;       ///< factor, flags (1 posW, 2 history valid, 4 jitter by frame, 8 colour), steps, lightSteps
    uint4 gSeed;        ///< salt, frameIndex, octaves, detailOctaves
    float4 gCamPos;     ///< camera position, w = |W|
    float4 gCamU;       ///< xyz = camera U, w = |U|^2
    float4 gCamV;       ///< xyz = camera V, w = |V|^2
    float4 gCamW;       ///< xyz = camera W, w = |W|^2
    float4 gPrevPos;
    float4 gPrevU;
    float4 gPrevV;
    float4 gPrevW;
    float4 gUp;         ///< xyz = up (unit), w = camera altitude above the ground
    float4 gE1;         ///< xyz = horizontal axis 1 (weather map u)
    float4 gE2;         ///< xyz = horizontal axis 2 (weather map v)
    float4 gSlab;       ///< cloud bottom, cloud top, planet radius (0 = flat), maxDistance
    float4 gShape;      ///< shapeScale, detailScale, detailStrength, weatherScale
    float4 gWeather;    ///< coverage, cloudType, weatherAmount, typeVariation
    float4 gMedia;      ///< density (extinction per unit at d = 1), albedo, powder, minTransmittance
    float4 gPhase;      ///< g forward, g backward, lobe mix, jitter
    float4 gLightDir;   ///< xyz = towards the light, w = first light step
    float4 gLightCol;   ///< rgb = light colour x intensity, w = light step growth
    float4 gAmbTop;     ///< rgb, w = fadeDistance (0 = off)
    float4 gAmbBot;     ///< rgb, w = temporal blend
    float4 gOffset;     ///< xyz = noise offset (wind x time), w = opacity
    float4 gWindDelta;  ///< xyz = cloud displacement since the previous frame
};

Texture2D<float4> gColor;
Texture2D<float4> gPosW;
Texture2D<float4> gMarchA;      ///< march: S.rgb, T
Texture2D<float4> gMarchB;      ///< march: front distance along the ray (-1 = none)
Texture2D<float4> gHistA;       ///< history read: S.rgb, T
Texture2D<float4> gHistB;       ///< history read: front, valid
RWTexture2D<float4> gOutA;      ///< march output A / history write A
RWTexture2D<float4> gOutB;      ///< march output B / history write B
RWTexture2D<float4> gDst;
RWTexture2D<float4> gLayer;
RWTexture2D<float4> gLayerDepth;
RWTexture2D<float4> gTrans;

static const float kNoDepth = 2.9e38f;

// ---- integer hash, value noise (integer part bit-identical to the CPU reference) ----

uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

float u01(uint u) { return float(u >> 8u) * (1.f / 16777216.f); }

float hashCorner(int3 i, uint salt)
{
    return u01(pcg3d(uint3(uint(i.x), uint(i.y), uint(i.z) ^ salt)).x);
}

float vnoise3(float3 p, uint salt)
{
    const float3 fi = floor(p);
    const int3 i = int3(fi);
    const float3 f = p - fi;
    const float3 w = f * f * (3.f - 2.f * f);
    const float x00 = lerp(hashCorner(i + int3(0, 0, 0), salt), hashCorner(i + int3(1, 0, 0), salt), w.x);
    const float x10 = lerp(hashCorner(i + int3(0, 1, 0), salt), hashCorner(i + int3(1, 1, 0), salt), w.x);
    const float x01 = lerp(hashCorner(i + int3(0, 0, 1), salt), hashCorner(i + int3(1, 0, 1), salt), w.x);
    const float x11 = lerp(hashCorner(i + int3(0, 1, 1), salt), hashCorner(i + int3(1, 1, 1), salt), w.x);
    return lerp(lerp(x00, x10, w.y), lerp(x01, x11, w.y), w.z);
}

/** fBm of vnoise3, lacunarity 2.03, gain 0.5, per-octave salt, normalised to [0, 1). */
float fbm3(float3 p, uint oct, uint salt)
{
    float s = 0.f, a = 0.5f, n = 0.f;
    for (uint o = 0; o < oct; o++)
    {
        s += a * vnoise3(p, salt + o * 7919u);
        n += a;
        p = p * 2.03f + float3(17.1f, 5.3f, 11.7f);
        a *= 0.5f;
    }
    return s / n;
}

float sstep(float e0, float e1, float x)
{
    const float t = saturate((x - e0) / (e1 - e0));
    return t * t * (3.f - 2.f * t);
}

/** Vertical density profile at normalised height h for cloud type t (0 stratus, 0.5 stratocumulus, 1 cumulus). */
float heightProfile(float h, float t)
{
    const float st = sstep(0.f, 0.07f, h) * (1.f - sstep(0.12f, 0.28f, h));
    const float sc = sstep(0.f, 0.1f, h) * (1.f - sstep(0.3f, 0.6f, h));
    const float cu = sstep(0.f, 0.12f, h) * (1.f - sstep(0.55f, 0.98f, h));
    return t < 0.5f ? lerp(st, sc, 2.f * t) : lerp(sc, cu, 2.f * t - 1.f);
}

/** Altitude above the ground of camera + r (curved shell of radius R, or flat for R = 0). */
float altitudeOf(float3 r)
{
    const float a = gUp.w, R = gSlab.z;
    const float ru = dot(r, gUp.xyz);
    if (R <= 0.f) return a + ru;
    const float Ra = R + a;
    const float k = 2.f * Ra * ru + dot(r, r);
    return a + k / (sqrt(max(Ra * Ra + k, 0.f)) + Ra);
}

/** Cloud density in [0, 1] at camera + r with altitude alt. */
float cloudDensity(float3 r, float alt)
{
    const float bot = gSlab.x, top = gSlab.y;
    const float h = (alt - bot) / (top - bot);
    if (h < 0.f || h > 1.f) return 0.f;
    const uint salt = gSeed.x;
    const float3 q = gCamPos.xyz + r - gOffset.xyz;
    const float2 wq = float2(dot(q, gE1.xyz), dot(q, gE2.xyz)) * gShape.w;
    const float wc = fbm3(float3(wq, 0.5f), 3u, salt + 101u);
    const float cov = saturate(gWeather.x * (1.f + gWeather.z * (2.f * wc - 1.f)));
    if (cov <= 0.f) return 0.f;
    const float wt = fbm3(float3(wq, 3.5f), 2u, salt + 202u);
    const float type = saturate(gWeather.y + gWeather.w * (2.f * wt - 1.f));
    const float prof = heightProfile(h, type);
    if (prof <= 0.f) return 0.f;
    const float base = fbm3(q * gShape.x, gSeed.z, salt);
    float d = saturate((base * prof - (1.f - cov)) / max(cov, 1e-3f));
    if (d > 0.f && gShape.z > 0.f)
    {
        const float det = fbm3(q * gShape.y, gSeed.w, salt + 303u);
        d = saturate(d - gShape.z * det * (1.f - d));
    }
    return d;
}

float hg(float c, float g)
{
    const float g2 = g * g;
    const float den = 1.f + g2 - 2.f * g * c;
    return 0.0795774715459f * (1.f - g2) / (den * sqrt(den));
}

/** Distances t >= 0 along unit dir where the ray is inside the slab: (t0, t1); t1 <= t0 = no interval. */
float2 slabInterval(float3 dir)
{
    const float a = gUp.w, R = gSlab.z, bot = gSlab.x, top = gSlab.y;
    const float mu = dot(dir, gUp.xyz);
    float t0 = 0.f, t1 = -1.f;
    if (R <= 0.f)
    {
        if (a < bot) { if (mu > 0.f) { t0 = (bot - a) / mu; t1 = (top - a) / mu; } }
        else if (a > top) { if (mu < 0.f) { t0 = (top - a) / mu; t1 = (bot - a) / mu; } }
        else { t0 = 0.f; t1 = mu > 0.f ? (top - a) / mu : (mu < 0.f ? (bot - a) / mu : 3.0e38f); }
        return float2(t0, t1);
    }
    const float Ra = R + a, b = Ra * mu;
    const float db = b * b + (bot - a) * (2.f * R + bot + a);
    const float dt = b * b + (top - a) * (2.f * R + top + a);
    if (a < bot)
    {
        t0 = -b + sqrt(max(db, 0.f));
        t1 = -b + sqrt(max(dt, 0.f));
    }
    else if (a > top)
    {
        if (mu < 0.f && dt >= 0.f)
        {
            t0 = -b - sqrt(dt);
            t1 = db >= 0.f ? -b - sqrt(db) : -b + sqrt(dt);
        }
    }
    else
    {
        t0 = 0.f;
        t1 = (mu < 0.f && db >= 0.f) ? -b - sqrt(db) : -b + sqrt(max(dt, 0.f));
    }
    return float2(t0, t1);
}

float lightTransmittance(float3 r)
{
    float s = 0.f, ds = gLightDir.w, od = 0.f;
    for (uint k = 0; k < gFlags.w; k++)
    {
        const float3 rl = r + gLightDir.xyz * (s + 0.5f * ds);
        const float alt = altitudeOf(rl);
        if (alt > gSlab.y) break;
        od += cloudDensity(rl, alt) * ds;
        s += ds;
        ds *= gLightCol.w;
    }
    return exp(-gMedia.x * od);
}

float3 pixelRay(float2 uv, float3 U, float3 V, float3 W)
{
    const float2 ndc = float2(2.f * uv.x - 1.f, 1.f - 2.f * uv.y);
    return ndc.x * U + ndc.y * V + W;
}

/** S.rgb, T and the front distance for unit dir, march limited to maxT. jitter in [0, 1). */
void marchRay(float3 dir, float maxT, float jit, out float4 sT, out float front)
{
    sT = float4(0.f, 0.f, 0.f, 1.f);
    front = -1.f;
    const float2 iv = slabInterval(dir);
    const float t0 = iv.x, t1 = min(iv.y, min(maxT, gSlab.w));
    if (!(t1 > t0)) return;
    const uint n = max(gFlags.z, 1u);
    const float dt = (t1 - t0) / float(n);
    const float c = dot(dir, gLightDir.xyz);
    const float phase = lerp(hg(c, gPhase.x), hg(c, gPhase.y), gPhase.z);
    const float3 sun = gLightCol.rgb * phase;
    float3 S = float3(0.f, 0.f, 0.f);
    float T = 1.f;
    for (uint i = 0; i < n; i++)
    {
        const float t = t0 + (float(i) + jit) * dt;
        const float3 r = dir * t;
        const float alt = altitudeOf(r);
        const float d = cloudDensity(r, alt);
        if (d > 0.f)
        {
            if (front < 0.f) front = t;
            const float sigma = d * gMedia.x;
            const float h = saturate((alt - gSlab.x) / (gSlab.y - gSlab.x));
            const float Tl = lightTransmittance(r);
            const float powder = lerp(1.f, 1.f - exp(-2.f * sigma * gLightDir.w), gMedia.z);
            const float3 amb = lerp(gAmbBot.rgb, gAmbTop.rgb, h);
            const float3 Sc = gMedia.y * (sun * (Tl * powder) + amb);
            const float dT = exp(-sigma * dt);
            S += T * Sc * (1.f - dT);
            T *= dT;
            if (T < gMedia.w) break;
        }
    }
    if (front >= 0.f && (gAmbTop.w > 0.f || gOffset.w < 1.f))
    {
        const float f = (gAmbTop.w > 0.f ? exp(-front / gAmbTop.w) : 1.f) * gOffset.w;
        S *= f;
        T = 1.f - (1.f - T) * f;
    }
    sT = float4(S, T);
}

float jitterAt(uint2 p)
{
    if (gPhase.w <= 0.f) return 0.5f;
    const uint fr = (gFlags.y & 4u) != 0u ? gSeed.y : 0u;
    const float h = u01(pcg3d(uint3(p.x, p.y, fr * 0x9E3779B9u + gSeed.x)).x);
    return 0.5f + gPhase.w * (h - 0.5f);
}

/** Ray distance to the scene surface at full-res pixel p (3e38 on background or without posW). */
float sceneDist(uint2 p)
{
    if ((gFlags.y & 1u) == 0u) return 3.0e38f;
    const float4 pw = gPosW[p];
    if (pw.w == 0.f) return 3.0e38f;
    return length(pw.xyz - gCamPos.xyz);
}

[numthreads(8, 8, 1)]
void march(uint3 tid: SV_DispatchThreadID)
{
    const uint2 lp = tid.xy;
    if (any(lp >= gDims.zw)) return;
    const uint f = gFlags.x;
    // The farthest scene surface over the footprint: clouds behind any background pixel of the footprint are kept.
    float maxT = 0.f;
    for (uint y = 0; y < f; y++)
        for (uint x = 0; x < f; x++)
        {
            const uint2 p = min(lp * f + uint2(x, y), gDims.xy - 1u);
            maxT = max(maxT, sceneDist(p));
        }
    const float2 uv = (float2(lp) + 0.5f) / float2(gDims.zw);
    const float3 dir = normalize(pixelRay(uv, gCamU.xyz, gCamV.xyz, gCamW.xyz));
    float4 sT;
    float front;
    marchRay(dir, maxT, jitterAt(lp), sT, front);
    gOutA[lp] = sT;
    gOutB[lp] = float4(front, 0.f, 0.f, 0.f);
}

[numthreads(8, 8, 1)]
void resolve(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gDims.xy)) return;
    const uint f = gFlags.x;
    const int2 lmax = int2(gDims.zw) - 1;
    float4 cur;
    float front;
    int2 lc;
    if (f == 1u)
    {
        cur = gMarchA[p];
        front = gMarchB[p].x;
        lc = int2(p);
    }
    else
    {
        const float2 q = (float2(p) + 0.5f) / float(f) - 0.5f;
        const float2 q0 = floor(q);
        const float2 w = q - q0;
        const int2 i0 = int2(q0);
        cur = float4(0.f, 0.f, 0.f, 0.f);
        float fs = 0.f, fw = 0.f;
        for (int k = 0; k < 4; k++)
        {
            const int2 o = int2(k & 1, k >> 1);
            const int2 i = clamp(i0 + o, int2(0, 0), lmax);
            const float wk = (o.x == 1 ? w.x : 1.f - w.x) * (o.y == 1 ? w.y : 1.f - w.y);
            cur += wk * gMarchA[i];
            const float fr = gMarchB[i].x;
            if (fr >= 0.f) { fs += wk * fr; fw += wk; }
        }
        front = fw > 0.f ? fs / fw : -1.f;
        lc = clamp(int2(floor((float2(p) + 0.5f) / float(f))), int2(0, 0), lmax);
    }
    const float sd = sceneDist(p);
    const bool occluded = sd < 3.0e38f && (front < 0.f || sd <= front);
    if (occluded)
    {
        cur = float4(0.f, 0.f, 0.f, 1.f);
        front = -1.f;
    }
    const float3 dir = normalize(pixelRay((float2(p) + 0.5f) / float2(gDims.xy), gCamU.xyz, gCamV.xyz, gCamW.xyz));
    float4 res = cur;
    if ((gFlags.y & 2u) != 0u && !occluded)
    {
        // Reproject the cloud front point (or the direction at maxDistance) into the previous frame.
        const float3 X = gCamPos.xyz + dir * (front >= 0.f ? front : gSlab.w) - gWindDelta.xyz;
        const float3 v = X - gPrevPos.xyz;
        const float z = dot(v, gPrevW.xyz) / gPrevW.w;
        if (z > 0.f)
        {
            const float2 ndc = float2(dot(v, gPrevU.xyz) / gPrevU.w, dot(v, gPrevV.xyz) / gPrevV.w) / z;
            const float2 pp = float2(0.5f * (ndc.x + 1.f), 0.5f * (1.f - ndc.y)) * float2(gDims.xy) - 0.5f;
            if (all(pp >= 0.f) && all(pp <= float2(gDims.xy) - 1.f))
            {
                const float2 p0 = floor(pp);
                const float2 w = pp - p0;
                const int2 i0 = int2(p0);
                const int2 hmax = int2(gDims.xy) - 1;
                float4 h = float4(0.f, 0.f, 0.f, 0.f);
                float valid = 1.f;
                for (int k = 0; k < 4; k++)
                {
                    const int2 o = int2(k & 1, k >> 1);
                    const int2 i = min(i0 + o, hmax);
                    const float wk = (o.x == 1 ? w.x : 1.f - w.x) * (o.y == 1 ? w.y : 1.f - w.y);
                    h += wk * gHistA[i];
                    if (wk > 0.f) valid = min(valid, gHistB[i].y);
                }
                if (valid > 0.f)
                {
                    float4 mn = float4(3.0e38f, 3.0e38f, 3.0e38f, 3.0e38f), mx = -mn;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            const float4 s = gMarchA[clamp(lc + int2(dx, dy), int2(0, 0), lmax)];
                            mn = min(mn, s);
                            mx = max(mx, s);
                        }
                    res = lerp(cur, clamp(h, mn, mx), gAmbBot.w);
                }
            }
        }
    }
    gOutA[p] = res;
    gOutB[p] = float4(front, occluded ? 0.f : 1.f, 0.f, 0.f);
    const float4 c = (gFlags.y & 8u) != 0u ? gColor[p] : float4(0.f, 0.f, 0.f, 1.f);
    gDst[p] = occluded ? c : float4(c.rgb * res.a + res.rgb, c.a);
    gLayer[p] = float4(res.rgb, 1.f - res.a);
    gLayerDepth[p] = float4(front >= 0.f ? front * dot(dir, gCamW.xyz) / gCamPos.w : kNoDepth, 0.f, 0.f, 0.f);
    gTrans[p] = float4(res.a, res.a, res.a, 1.f);
}
