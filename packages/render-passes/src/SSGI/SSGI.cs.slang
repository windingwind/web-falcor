/**
 * SSGI: screen-space one-bounce diffuse indirect lighting with a visibility bitmask (see SSGI.ts). Entry points:
 *  trace      : per trace pixel, `gSliceCount` slices x 2 sides x `gStepCount` samples; each sample occludes the angular
 *               sectors between its front surface and front + thickness (extended back to the previous sample when both
 *               lie on one smooth surface); sectors occluded first receive the sample's
 *               radiance (if its surface faces the receiver), weighted by the sector's cosine-weighted slice measure.
 *               Writes (incident radiance / pi, unoccluded share).
 *  denoise    : one a-trous iteration (5x5 B3 kernel, step gAtrousStep) with normal/depth edge stops.
 *  accumulate : full-resolution bilateral upsample of the denoised trace + exponential blend with the reprojected
 *               history (neighbourhood clamp); without valid history the upsampled value is written as is.
 *  composite  : dst = color + intensity * albedo * bounce (physical), or albedo * (paint(bounce) + hemisphere ambient)
 *               (painted); intensity 0 copies `color` bit for bit.
 *
 * Painted mode is continuous in all its parameters: a hue rotation and chroma scale in YIQ, a tint, and a sky/ground
 * blend by the normal; there is no floor/round/step of luminance or N.L. (The 32 angular sectors of the trace are the
 * visibility bitmask of the method; slice rotation and step jitter vary per pixel and frame, so they do not band.)
 *
 * Adapted from three.js SSGINode (MIT, Copyright (c) 2010-2026 three.js authors, https://github.com/mrdoob/three.js)
 * and SSRT3 (MIT, Copyright (c) 2024 CDRIN, https://github.com/cdrinmatane/SSRT3); full notice in SSGI.ts.
 * Method: Therrien, Levesque, Gilet, "Screen Space Indirect Lighting with Visibility Bitmask" (2023); slice measure:
 * Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion" (2016).
 */
#include "Utils/Math/MathConstants.slangh"

cbuffer CB
{
    uint2 gFrameDim;
    uint2 gTraceDim;
    uint gScale;
    uint gSliceCount;
    uint gStepCount;
    uint gFrameSeed;
    float4x4 gInvViewProj;
    float3 gCamPos;
    float gRadius;
    float3 gCamFwd;
    float gThickness;
    float gProjScale;      ///< Pixels per world unit at view depth 1.
    float gMaxRadiusPx;
    uint gHizMips;         ///< 0 = no Hi-Z input.
    float gHizStartPx;
    float gRadianceClamp;  ///< 0 = off.
    uint gAtrousStep;
    float gSigmaN;
    float gSigmaZ;
    float gTemporalAlpha;
    uint gHistoryValid;
    float gIntensity;
    uint gPainted;
    float gHueShift;       ///< Radians.
    float gSaturation;
    float3 gTint;
    float gHemiIntensity;
    float3 gSkyColor;
    float gHemiOcclusion;
    float3 gGroundColor;
    float _pad0;
    float3 gUp;
    float _pad1;
};

Texture2D<float4> gDepth;
Texture2D<float4> gNormW;
Texture2D<float4> gRadiance;
Texture2D<float2> gHiZ;
Texture2D<float4> gDenoiseIn;
Texture2D<float4> gDenoised;
Texture2D<float4> gMvec;
Texture2D<float4> gHistory;
Texture2D<float4> gColor;
Texture2D<float4> gAccum;
Texture2D<float4> gAlbedo;

RWTexture2D<float4> gTraceOut;
RWTexture2D<float4> gDenoiseOut;
RWTexture2D<float4> gAccumOut;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gIndirectOut;
RWTexture2D<float4> gGiOut;

static const uint kSectors = 32;

// ---------------------------------------------------------------------------
// Helpers

float3 worldPos(float2 uv, float z)
{
    const float4 w = mul(gInvViewProj, float4(uv.x * 2.f - 1.f, 1.f - uv.y * 2.f, z, 1.f));
    return w.xyz / w.w;
}

float viewDepth(float3 p) { return dot(p - gCamPos, gCamFwd); }

uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

float2 rand2(uint2 p, uint frame)
{
    const uint3 h = pcg3d(uint3(p, frame));
    return float2(h.xy >> 8) * (1.f / 16777216.f);
}

/// Antiderivative of cos(t - n) sin(t).
float G(float t, float n) { return -0.25f * cos(2.f * t - n) + 0.5f * t * sin(n); }

/// Integral of cos(t - n) |sin t| over [a, b] (a <= b, both in [-pi, pi]).
float sectorWeight(float a, float b, float n)
{
    if (a >= 0.f) return G(b, n) - G(a, n);
    if (b <= 0.f) return G(a, n) - G(b, n);
    return (G(0.f, n) - G(a, n)) + (G(b, n) - G(0.f, n));
}

/// Signed angle of d from V towards T in the slice plane.
float sliceAngle(float3 d, float3 V, float3 T) { return atan2(dot(d, T), dot(d, V)); }

float sampleDepth(uint2 sp, float t)
{
    if (gHizMips > 1 && t > gHizStartPx)
    {
        const uint mip = min(uint(log2(t / gHizStartPx)) + 1u, gHizMips - 1u);
        return gHiZ.Load(int3(int2(sp >> mip), int(mip))).x;
    }
    return gDepth[sp].x;
}

// ---------------------------------------------------------------------------

[numthreads(8, 8, 1)]
void trace(uint3 tid: SV_DispatchThreadID)
{
    const uint2 q = tid.xy;
    if (any(q >= gTraceDim)) return;
    const uint2 p = min(q * gScale, gFrameDim - 1);
    const float z = gDepth[p].x;
    if (z >= 1.f)
    {
        gTraceOut[q] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const float2 dim = float2(gFrameDim);
    const float2 uv = (float2(p) + 0.5f) / dim;
    const float3 P = worldPos(uv, z);
    const float3 N = normalize(gNormW[p].xyz);
    const float3 V = normalize(gCamPos - P);
    const float vz = max(viewDepth(P), 1e-4f);
    const float radiusPx = min(gRadius * gProjScale / vz, gMaxRadiusPx);
    if (radiusPx < 1.f)
    {
        gTraceOut[q] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const float2 rnd = rand2(p, gFrameSeed);
    const float sectorAngle = M_PI / float(kSectors);
    float3 gi = float3(0.f);
    float ao = 0.f;
    float norm = 0.f;
    for (uint i = 0; i < gSliceCount; i++)
    {
        const float phi = (float(i) + rnd.x) * M_PI / float(gSliceCount);
        const float2 dir = float2(cos(phi), sin(phi));
        // World direction of a one-pixel screen step along `dir` at this depth; T = its part orthogonal to V.
        const float3 D = worldPos(uv + dir / dim, z) - P;
        float3 T = D - V * dot(D, V);
        const float lt = length(T);
        if (lt < 1e-12f) continue;
        T /= lt;
        const float3 A = cross(T, V); // slice plane normal
        const float3 projN = N - A * dot(N, A);
        const float lp = length(projN);
        norm += 1.f;
        if (lp < 1e-6f) continue;
        const float n = sliceAngle(projN, V, T);
        const float lo = n - 0.5f * M_PI;
        float w[kSectors];
        for (uint k = 0; k < kSectors; k++)
        {
            const float a = lo + float(k) * sectorAngle;
            w[k] = sectorWeight(a, a + sectorAngle, n);
        }
        uint occ = 0;
        float3 L = float3(0.f);
        for (int side = 0; side < 2; side++)
        {
            const float2 sdir = side == 0 ? dir : -dir;
            float3 prevS = P, prevN = N;
            float prevA = 0.f;
            bool prevValid = false;
            for (uint j = 0; j < gStepCount; j++)
            {
                const float u = (float(j) + rnd.y) / float(gStepCount);
                const float t = max(float(j + 1), radiusPx * u * u);
                const float2 s2 = float2(p) + 0.5f + sdir * t;
                if (any(s2 < 0.f) || any(s2 >= dim)) break;
                const uint2 sp = uint2(s2);
                const float zs = sampleDepth(sp, t);
                if (zs >= 1.f) { prevValid = false; continue; } // background never occludes
                const float3 S = worldPos((float2(sp) + 0.5f) / dim, zs);
                const float3 Ns = normalize(gNormW[sp].xyz);
                const float3 d = S - P;
                const float3 db = d + normalize(S - gCamPos) * gThickness;
                float tf = sliceAngle(d, V, T);
                float tb = sliceAngle(db, V, T);
                if (tb - tf > M_PI) tb -= 2.f * M_PI;
                else if (tf - tb > M_PI) tb += 2.f * M_PI;
                float a0 = min(tf, tb), a1 = max(tf, tb);
                // Bridge: when this sample and the previous one on this side lie on one smooth surface (similar normals,
                // the previous point on this sample's tangent plane), the surface also covers the angles in between, so
                // sparse steps leave no gaps in the bitmask. Depth discontinuities never pass the plane test.
                const float3 e = S - prevS;
                if (prevValid && dot(Ns, prevN) > 0.9f && abs(dot(e, Ns)) <= 0.05f * length(e) + 1e-6f)
                {
                    float pa = prevA;
                    if (pa - tf > M_PI) pa -= 2.f * M_PI;
                    else if (tf - pa > M_PI) pa += 2.f * M_PI;
                    a0 = min(a0, pa);
                    a1 = max(a1, pa);
                }
                prevS = S; prevN = Ns; prevA = tf; prevValid = true;
                const float x0 = saturate((a0 - lo) / M_PI);
                const float x1 = saturate((a1 - lo) / M_PI);
                const uint start = min(uint(x0 * float(kSectors)), kSectors - 1);
                const uint cnt = min(uint(ceil((x1 - x0) * float(kSectors))), kSectors);
                const uint bits = cnt == 0 ? 0u : ((0xFFFFFFFFu >> (kSectors - cnt)) << start);
                uint fresh = bits & ~occ;
                occ |= bits;
                if (fresh == 0u) continue;
                if (dot(Ns, d) >= 0.f) continue; // the sample's surface faces away from the receiver: occludes, no light
                float3 Ls = gRadiance[sp].rgb;
                if (gRadianceClamp > 0.f) Ls = min(Ls, float3(gRadianceClamp));
                float ws = 0.f;
                while (fresh != 0u)
                {
                    ws += w[firstbitlow(fresh)];
                    fresh &= fresh - 1u;
                }
                L += ws * Ls;
            }
        }
        float vis = 0.f;
        uint open = ~occ;
        while (open != 0u)
        {
            vis += w[firstbitlow(open)];
            open &= open - 1u;
        }
        gi += lp * L;
        ao += lp * vis;
    }
    const float inv = norm > 0.f ? 1.f / norm : 0.f;
    gTraceOut[q] = float4(max(gi * inv, float3(0.f)), saturate(ao * inv));
}

/// View depth and normal of the full-resolution texel under trace texel q.
void guide(uint2 q, out float d, out float3 n, out bool bg)
{
    const uint2 p = min(q * gScale, gFrameDim - 1);
    const float z = gDepth[p].x;
    bg = z >= 1.f;
    n = normalize(gNormW[p].xyz);
    d = bg ? 0.f : viewDepth(worldPos((float2(p) + 0.5f) / float2(gFrameDim), z));
}

[numthreads(8, 8, 1)]
void denoise(uint3 tid: SV_DispatchThreadID)
{
    const uint2 q = tid.xy;
    if (any(q >= gTraceDim)) return;
    const float4 c = gDenoiseIn[q];
    float d0; float3 n0; bool bg;
    guide(q, d0, n0, bg);
    if (bg)
    {
        gDenoiseOut[q] = c;
        return;
    }
    const float h[3] = { 0.375f, 0.25f, 0.0625f };
    float4 sum = float4(0.f);
    float wsum = 0.f;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            const int2 s = int2(q) + int2(dx, dy) * int(gAtrousStep);
            if (any(s < 0) || any(s >= int2(gTraceDim))) continue;
            float dq; float3 nq; bool bq;
            guide(uint2(s), dq, nq, bq);
            if (bq) continue;
            const float wt = h[abs(dx)] * h[abs(dy)] * pow(max(dot(n0, nq), 0.f), gSigmaN) * exp(-abs(dq - d0) / max(gSigmaZ * d0, 1e-6f));
            sum += wt * gDenoiseIn[uint2(s)];
            wsum += wt;
        }
    }
    gDenoiseOut[q] = wsum > 0.f ? sum / wsum : c;
}

[numthreads(16, 16, 1)]
void accumulate(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float z = gDepth[p].x;
    if (z >= 1.f)
    {
        gAccumOut[p] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const uint2 qc = min(p / gScale, gTraceDim - 1);
    float4 c = gDenoised[qc];
    if (gScale > 1)
    {
        const float d0 = viewDepth(worldPos((float2(p) + 0.5f) / float2(gFrameDim), z));
        const float3 n0 = normalize(gNormW[p].xyz);
        const float2 fq = (float2(p) + 0.5f) / float(gScale) - 0.5f; // position in trace texels
        float4 sum = float4(0.f);
        float wsum = 0.f;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                const int2 s = int2(qc) + int2(dx, dy);
                if (any(s < 0) || any(s >= int2(gTraceDim))) continue;
                float dq; float3 nq; bool bq;
                guide(uint2(s), dq, nq, bq);
                if (bq) continue;
                const float2 o = abs(float2(s) - fq);
                const float tent = max(1.5f - o.x, 0.f) * max(1.5f - o.y, 0.f);
                const float wt = tent * pow(max(dot(n0, nq), 0.f), 8.f) * exp(-abs(dq - d0) / max(0.05f * d0, 1e-6f));
                sum += wt * gDenoised[uint2(s)];
                wsum += wt;
            }
        }
        if (wsum > 0.f) c = sum / wsum;
    }
    float4 o = c;
    if (gHistoryValid != 0 && gTemporalAlpha > 0.f)
    {
        const float2 uv = (float2(p) + 0.5f) / float2(gFrameDim) + gMvec[p].xy;
        if (all(uv >= 0.f) && all(uv < 1.f))
        {
            const uint2 hp = min(uint2(uv * float2(gFrameDim)), gFrameDim - 1);
            float4 lo = c, hi = c;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    const int2 s = clamp(int2(qc) + int2(dx, dy), int2(0), int2(gTraceDim) - 1);
                    const float4 v = gDenoised[uint2(s)];
                    lo = min(lo, v);
                    hi = max(hi, v);
                }
            }
            o = lerp(c, clamp(gHistory[hp], lo, hi), gTemporalAlpha);
        }
    }
    gAccumOut[p] = o;
}

/// Painted bounce: hue rotation + chroma scale in YIQ, tint. Continuous in every parameter.
float3 paint(float3 b)
{
    const float Y = dot(b, float3(0.299f, 0.587f, 0.114f));
    const float I = dot(b, float3(0.595716f, -0.274453f, -0.321263f));
    const float Q = dot(b, float3(0.211456f, -0.522591f, 0.311135f));
    const float cs = cos(gHueShift), sn = sin(gHueShift);
    const float I2 = gSaturation * (cs * I - sn * Q);
    const float Q2 = gSaturation * (sn * I + cs * Q);
    const float3 rgb = float3(Y + 0.9563f * I2 + 0.6210f * Q2, Y - 0.2721f * I2 - 0.6474f * Q2, Y - 1.1070f * I2 + 1.7046f * Q2);
    return max(rgb, float3(0.f)) * gTint;
}

[numthreads(16, 16, 1)]
void composite(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 c = gColor[p];
    const float4 g = gAccum[p];
    float3 add = float3(0.f);
    if (gDepth[p].x < 1.f)
    {
        const float3 alb = gAlbedo[p].rgb;
        if (gPainted != 0)
        {
            const float3 N = normalize(gNormW[p].xyz);
            const float k = 0.5f + 0.5f * dot(N, gUp);
            const float3 hemi = lerp(gGroundColor, gSkyColor, k) * gHemiIntensity * lerp(1.f, g.a, gHemiOcclusion);
            add = alb * (paint(g.rgb) + hemi);
        }
        else
        {
            add = alb * g.rgb;
        }
    }
    gDst[p] = gIntensity == 0.f ? c : float4(c.rgb + gIntensity * add, c.a);
    gIndirectOut[p] = float4(add, g.a);
    gGiOut[p] = g;
}
