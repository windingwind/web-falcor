/**
 * SSR: screen-space reflections on a Hi-Z min-depth pyramid (see SSR.ts). Four entry points, one dispatch each:
 *  trace     : classify (background, roughness cutoff) + one reflection ray per pixel (mirror, or a GGX visible-normal
 *              sample for rough pixels) marched through the Hi-Z pyramid; writes (hit radiance, confidence) and the
 *              debug hit record (hit uv, hit depth, confidence).
 *  resolve   : neighbour reuse for rough pixels (radius grows with roughness; weights = confidence x normal/depth
 *              similarity); radius 0 copies.
 *  temporal  : exponential blend with the reprojected history (motion vectors), history clamped to the current 3x3
 *              neighbourhood; skipped (output = input) on a frame without valid history.
 *  composite : dst = color + intensity * confidence * radiance * DFG(F0, roughness, N.V) (split-sum, Karis' analytic
 *              fit); intensity 0 copies `color` bit for bit.
 *
 * Credits (ideas only, no code copied):
 *  - Hi-Z traversal (min-depth cell stepping, mip ascend/descend) after AMD FidelityFX SSSR, MIT licence,
 *    https://github.com/GPUOpen-Effects/FidelityFX-SSSR, Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *  - GGX visible-normal sampling: E. Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT 7(4), 2018.
 *  - Analytic environment BRDF: B. Karis, "Physically Based Shading on Mobile", 2014 (same fit as DeferredLighting).
 *
 * Conventions: depth is standard NDC depth (0 = near plane), the Hi-Z r channel is the minimum (nearest) depth;
 * screen space for the march is (texel x, texel y, NDC depth), in which a world-space ray is a straight line.
 */
#include "Utils/Math/MathConstants.slangh"

cbuffer CB
{
    uint2 gFrameDim;
    uint gMipCount;
    uint gMaxSteps;
    float4x4 gViewProj;      ///< World -> clip (no jitter).
    float4x4 gInvViewProj;   ///< Clip -> world.
    float3 gCamPos;
    float gThickness;        ///< Relative thickness: a hit is accepted if the ray is at most thickness x view depth behind the surface.
    float3 gCamFwd;
    float gNear;
    float gMaxRoughness;     ///< Pixels rougher than this get no reflection.
    float gMirrorRoughness;  ///< Pixels at or below this roughness trace the mirror direction.
    uint gStochastic;        ///< 1 = GGX VNDF rays for rough pixels; 0 = mirror rays + resolve blur.
    uint gFrameSeed;         ///< Random sequence index (frames since the last history reset, plus the seed).
    int gResolveRadius;      ///< Neighbour-reuse radius at roughness = maxRoughness.
    float gTemporalAlpha;
    uint gHistoryValid;
    float gIntensity;
    float gBorderFade;       ///< Confidence fade width at the screen border (uv units).
};

Texture2D<float4> gColor;
Texture2D<float4> gDepth;
Texture2D<float4> gNormW;
Texture2D<float4> gSpecRough;
Texture2D<float4> gMvec;
Texture2D<float2> gHiZ;
Texture2D<float4> gTraced;
Texture2D<float4> gResolved;
Texture2D<float4> gHistory;
Texture2D<float4> gRefl;

RWTexture2D<float4> gTraceOut;
RWTexture2D<float4> gHitOut;
RWTexture2D<float4> gResolveOut;
RWTexture2D<float4> gTemporalOut;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gReflectionOut;

// ---------------------------------------------------------------------------
// Helpers

float3 worldPos(float2 uv, float z)
{
    const float4 w = mul(gInvViewProj, float4(uv.x * 2.f - 1.f, 1.f - uv.y * 2.f, z, 1.f));
    return w.xyz / w.w;
}

/// World -> (texel x, texel y, NDC depth).
float3 toScreen(float3 p)
{
    const float4 c = mul(gViewProj, float4(p, 1.f));
    const float3 n = c.xyz / c.w;
    return float3((n.x * 0.5f + 0.5f) * float(gFrameDim.x), (0.5f - n.y * 0.5f) * float(gFrameDim.y), n.z);
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

/// GGX visible-normal sample in tangent space (z = normal); Ve = view direction in tangent space.
float3 sampleGGXVNDF(float3 Ve, float alpha, float2 u)
{
    const float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const float3 T1 = lensq > 0.f ? float3(-Vh.y, Vh.x, 0.f) / sqrt(lensq) : float3(1.f, 0.f, 0.f);
    const float3 T2 = cross(Vh, T1);
    const float r = sqrt(u.x);
    const float phi = 2.f * M_PI * u.y;
    const float t1 = r * cos(phi);
    const float s = 0.5f * (1.f + Vh.z);
    const float t2 = (1.f - s) * sqrt(max(0.f, 1.f - t1 * t1)) + s * r * sin(phi);
    const float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.f, Nh.z)));
}

float3 envBRDFApprox(float3 F0, float roughness, float NoV)
{
    const float4 c0 = float4(-1.f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = float4(1.f, 0.0425f, 1.04f, -0.04f);
    const float4 r = roughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28f * NoV)) * r.x + r.y;
    const float2 AB = float2(-1.04f, 1.04f) * a004 + r.zw;
    return F0 * AB.x + AB.y;
}

// ---------------------------------------------------------------------------
// Hi-Z march

/// Distance along d (texel units) from pos to the boundary of its cell at `mip`.
float cellExit(float3 pos, float3 d, uint mip)
{
    const float size = float(1u << mip);
    const float2 cell = floor(pos.xy / size);
    const float bx = d.x > 0.f ? (cell.x + 1.f) * size : cell.x * size;
    const float by = d.y > 0.f ? (cell.y + 1.f) * size : cell.y * size;
    const float tx = d.x != 0.f ? (bx - pos.x) / d.x : 1e30f;
    const float ty = d.y != 0.f ? (by - pos.y) / d.y : 1e30f;
    return min(tx, ty);
}

/// Marches o + t d (screen space, d scaled to one texel per unit in its major axis). Returns true with the hit position.
bool hizTrace(float3 o, float3 d, out float3 hit)
{
    hit = o;
    const float2 dim = float2(gFrameDim);
    const uint maxMip = gMipCount - 1;
    const float kEps = 1e-3f;
    float3 pos = o + d * (cellExit(o, d, 0) + kEps); // leave the origin texel
    uint mip = 0;
    for (uint i = 0; i < gMaxSteps; i++)
    {
        if (any(pos.xy < 0.f) || any(pos.xy >= dim) || pos.z < 0.f || pos.z >= 1.f) return false; // left the screen or reached the far plane
        const float size = float(1u << mip);
        const int2 cell = int2(floor(pos.xy / size));
        const float zmin = gHiZ.Load(int3(cell, int(mip))).x;
        const float tCell = cellExit(pos, d, mip);
        if (pos.z < zmin)
        {
            // In front of every surface of the cell: either reach its min-depth plane inside the cell or leave the cell.
            const float tz = d.z > 0.f ? (zmin - pos.z) / d.z : 1e30f;
            if (tz < tCell)
            {
                pos += d * tz;
                pos.z = max(pos.z, zmin);
                if (mip > 0) mip--;
            }
            else
            {
                pos += d * (tCell + kEps);
                mip = min(mip + 1, maxMip);
            }
        }
        else if (mip > 0)
        {
            mip--;
        }
        else
        {
            // Candidate at mip 0: accept if the ray is not too far behind the surface (else it passes behind it).
            const float2 uv = (float2(cell) + 0.5f) / dim;
            const float zs = gDepth[uint2(cell)].x;
            const float dRay = viewDepth(worldPos(uv, pos.z));
            const float dSurf = viewDepth(worldPos(uv, zs));
            if (zs < 1.f && dRay - dSurf <= gThickness * dSurf) // background (depth 1) never blocks a ray
            {
                hit = float3(float2(cell) + 0.5f, zs);
                return true;
            }
            pos += d * (tCell + kEps);
        }
    }
    return false;
}

// ---------------------------------------------------------------------------

[numthreads(16, 16, 1)]
void trace(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    gTraceOut[p] = float4(0.f);
    gHitOut[p] = float4(0.f);
    const float z = gDepth[p].x;
    const float4 sr = gSpecRough[p];
    const float rough = sr.a;
    if (z >= 1.f || rough > gMaxRoughness) return;

    const float2 dim = float2(gFrameDim);
    const float2 uv = (float2(p) + 0.5f) / dim;
    const float3 P = worldPos(uv, z);
    const float3 N = normalize(gNormW[p].xyz);
    const float3 V = normalize(gCamPos - P);
    if (dot(N, V) <= 0.f) return;

    float3 R = reflect(-V, N);
    if (gStochastic != 0 && rough > gMirrorRoughness)
    {
        // Tangent frame around N.
        const float3 up = abs(N.y) < 0.999f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        const float3 T = normalize(cross(up, N));
        const float3 B = cross(N, T);
        const float3 Ve = float3(dot(V, T), dot(V, B), dot(V, N));
        const float a = max(rough * rough, 1e-4f);
        const float3 He = sampleGGXVNDF(Ve, a, rand2(p, gFrameSeed));
        const float3 H = He.x * T + He.y * B + He.z * N;
        const float3 Rs = reflect(-V, H);
        if (dot(Rs, N) > 0.f) R = Rs;
    }

    // Ray end point in front of the near plane (only the screen-space direction is used).
    const float vz = viewDepth(P);
    float len = max(vz, 1e-3f);
    const float rf = dot(R, gCamFwd);
    if (rf < 0.f) len = min(len, 0.9f * (vz - gNear) / -rf);
    if (len <= 0.f) return;
    const float3 s0 = float3(uv * dim, z);
    const float3 s1 = toScreen(P + R * len);
    float3 d = s1 - s0;
    const float m = max(abs(d.x), abs(d.y));
    if (m < 1e-4f) return; // ray along the view axis: no screen-space extent
    d /= m;

    float3 hit;
    if (!hizTrace(s0, d, hit)) return;
    const uint2 hp = uint2(hit.xy);
    // Reject back-face hits (the ray reached a surface from behind: thickness artefact).
    if (dot(normalize(gNormW[hp].xyz), R) >= 0.f) return;
    const float2 huv = hit.xy / dim;
    const float2 border = min(huv, 1.f - huv);
    float conf = gBorderFade > 0.f ? saturate(min(border.x, border.y) / gBorderFade) : 1.f;
    conf *= saturate((gMaxRoughness - rough) / max(0.2f * gMaxRoughness, 1e-4f));
    gTraceOut[p] = float4(gColor[hp].rgb, conf);
    gHitOut[p] = float4(huv, hit.z, conf);
}

[numthreads(16, 16, 1)]
void resolve(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 c = gTraced[p];
    const float z = gDepth[p].x;
    const float rough = gSpecRough[p].a;
    const int r = gMaxRoughness > 0.f ? int(float(gResolveRadius) * saturate(rough / gMaxRoughness) + 0.5f) : 0;
    if (r == 0 || z >= 1.f)
    {
        gResolveOut[p] = c;
        return;
    }
    const float2 dim = float2(gFrameDim);
    const float3 n = normalize(gNormW[p].xyz);
    const float d0 = viewDepth(worldPos((float2(p) + 0.5f) / dim, z));
    float3 sumL = float3(0.f);
    float sumW = 0.f, sumG = 0.f;
    for (int dy = -r; dy <= r; dy++)
    {
        for (int dx = -r; dx <= r; dx++)
        {
            const int2 q = int2(p) + int2(dx, dy);
            if (any(q < 0) || any(q >= int2(gFrameDim))) continue;
            const float zq = gDepth[q].x;
            if (zq >= 1.f) continue;
            const float dq = viewDepth(worldPos((float2(q) + 0.5f) / dim, zq));
            const float nq = max(dot(n, normalize(gNormW[q].xyz)), 0.f);
            const float g = pow(nq, 16.f) * exp(-abs(dq - d0) / max(0.02f * d0, 1e-4f));
            const float4 t = gTraced[q];
            sumG += g;
            sumW += g * t.a;
            sumL += g * t.a * t.rgb;
        }
    }
    gResolveOut[p] = sumW > 0.f ? float4(sumL / sumW, sumW / sumG) : float4(0.f);
}

[numthreads(16, 16, 1)]
void temporal(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 c = gResolved[p];
    float4 o = c;
    if (gHistoryValid != 0 && gTemporalAlpha > 0.f)
    {
        const float2 uv = (float2(p) + 0.5f) / float2(gFrameDim) + gMvec[p].xy;
        if (all(uv >= 0.f) && all(uv < 1.f))
        {
            const uint2 q = min(uint2(uv * float2(gFrameDim)), gFrameDim - 1);
            float4 lo = c, hi = c;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    const int2 s = clamp(int2(p) + int2(dx, dy), int2(0), int2(gFrameDim) - 1);
                    const float4 v = gResolved[s];
                    lo = min(lo, v);
                    hi = max(hi, v);
                }
            }
            o = lerp(c, clamp(gHistory[q], lo, hi), gTemporalAlpha);
        }
    }
    gTemporalOut[p] = o;
}

[numthreads(16, 16, 1)]
void composite(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 c = gColor[p];
    const float4 r = gRefl[p];
    float3 refl = float3(0.f);
    const float z = gDepth[p].x;
    if (z < 1.f && r.a > 0.f)
    {
        const float2 uv = (float2(p) + 0.5f) / float2(gFrameDim);
        const float3 P = worldPos(uv, z);
        const float3 N = normalize(gNormW[p].xyz);
        const float NoV = saturate(dot(N, normalize(gCamPos - P)));
        const float4 sr = gSpecRough[p];
        refl = r.a * r.rgb * envBRDFApprox(sr.rgb, sr.a, NoV);
    }
    gDst[p] = gIntensity == 0.f ? c : float4(c.rgb + gIntensity * refl, c.a);
    gReflectionOut[p] = float4(refl, r.a);
}
