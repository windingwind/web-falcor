/**
 * ProceduralNightSky kernel (see ProceduralNightSky.ts; its evalNightSky() is the CPU reference of this shader).
 * Stars on jittered 3D grids, domain-warped fBm nebula with dust lanes, layered aurora curtains. Up axis +y.
 */

cbuffer CB
{
    uint2 gFrameDim;
    uint gHasColor;
    uint gHasPos;
    uint gMask;             ///< bit 0 stars, 1 nebula, 2 aurora
    uint gReplace;          ///< 1 = background + layers, 0 = color + layers
    uint gSalt;
    uint gCurtains;
    uint gAuroraLayers;
    uint gHasLayer;
    uint gHasLayerDepth;
    uint _pad0;
    float4 gCamRight;       ///< xyz = right * tan(fovY/2) * aspect, w = pixel angle (2 tan(fovY/2) / H)
    float4 gCamUp;          ///< xyz = up * tan(fovY/2), w = skyScale
    float4 gCamFwd;         ///< xyz = forward, w = horizonFade
    float4 gZenith;         ///< rgb, w = time
    float4 gHorizon;        ///< rgb, w = cos(rotation)
    float4 gGround;         ///< rgb, w = sin(rotation)
    float4 gStar;           ///< intensity, density, grid, size (px)
    float4 gStar2;          ///< twinkle, colourVar
    float4 gNeb;            ///< intensity, scale, threshold, warp
    float4 gNebA;           ///< rgb, w = dust
    float4 gNebB;
    float4 gAur;            ///< intensity, height, thickness, distance
    float4 gAur2;           ///< spacing, wave, freq, rayFreq
    float4 gAur3;           ///< width, extent, time * speed
    float4 gAurAz;          ///< cos(azimuth), sin(azimuth)
    float4 gAurLow;
    float4 gAurHigh;
};

Texture2D<float4> gColor;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gLayer;
RWTexture2D<float4> gLayerDepth;

static const float kSkyDepth = 3.0e38f;
static const float kStarGridScale[3] = { 1.f, 2.03f, 4.07f };
static const float kStarLayerDensity[3] = { 1.f, 0.3f, 0.08f };

uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

uint3 hashCell(int3 c, uint salt)
{
    return pcg3d(uint3(uint(c.x), uint(c.y) ^ salt, uint(c.z) + salt * 0x9e37u));
}

float u01(uint u) { return float(u >> 8u) * (1.f / 16777216.f); }

float vnoise3(float3 p, uint salt)
{
    const float3 fi = floor(p);
    const int3 i = int3(fi);
    const float3 f = p - fi;
    const float3 w = f * f * (3.f - 2.f * f);
    const float v000 = u01(hashCell(i + int3(0, 0, 0), salt).x), v100 = u01(hashCell(i + int3(1, 0, 0), salt).x);
    const float v010 = u01(hashCell(i + int3(0, 1, 0), salt).x), v110 = u01(hashCell(i + int3(1, 1, 0), salt).x);
    const float v001 = u01(hashCell(i + int3(0, 0, 1), salt).x), v101 = u01(hashCell(i + int3(1, 0, 1), salt).x);
    const float v011 = u01(hashCell(i + int3(0, 1, 1), salt).x), v111 = u01(hashCell(i + int3(1, 1, 1), salt).x);
    const float x00 = lerp(v000, v100, w.x), x10 = lerp(v010, v110, w.x);
    const float x01 = lerp(v001, v101, w.x), x11 = lerp(v011, v111, w.x);
    return lerp(lerp(x00, x10, w.y), lerp(x01, x11, w.y), w.z);
}

float fbm3(float3 p, uint oct, uint salt)
{
    float s = 0.f, a = 0.5f, n = 0.f;
    for (uint o = 0; o < oct; o++)
    {
        s += a * vnoise3(p, salt + o);
        n += a;
        p *= 2.03f;
        a *= 0.5f;
    }
    return s / n;
}

float3 layersOf(float3 d)
{
    float3 L = float3(0.f);
    const float time = gZenith.w;
    const float3 r = float3(gHorizon.w * d.x + gGround.w * d.z, d.y, -gGround.w * d.x + gHorizon.w * d.z);
    float dust = 1.f;
    if ((gMask & 2u) != 0)
    {
        const float3 q = r * gNeb.y;
        const float wx = fbm3(q + float3(1.7f, 9.2f, 3.1f), 4, gSalt + 40u);
        const float wy = fbm3(q + float3(8.3f, 2.8f, 5.4f), 4, gSalt + 50u);
        const float wz = fbm3(q + float3(4.6f, 7.7f, 0.9f), 4, gSalt + 60u);
        const float n = fbm3(q + gNeb.w * (float3(wx, wy, wz) - 0.5f), 5, gSalt + 70u);
        const float dens = pow(saturate((n - gNeb.z) / max(1.f - gNeb.z, 1e-3f)), 1.5f);
        const float hue = fbm3(2.f * q + float3(5.3f, 1.1f, 7.9f), 3, gSalt + 80u);
        const float lanes = saturate(2.5f * fbm3(3.f * q + float3(11.f, 4.f, 2.f), 4, gSalt + 90u) - 1.1f);
        dust = 1.f - gNebA.w * lanes;
        L += gNeb.x * dens * dust * lerp(gNebA.rgb, gNebB.rgb, smoothstep(0.3f, 0.7f, hue));
    }
    if ((gMask & 1u) != 0)
    {
        const float rad = gStar.w * gCamRight.w;
        for (uint layer = 0; layer < 3; layer++)
        {
            const float N = gStar.z * kStarGridScale[layer];
            const float3 P = r * N;
            const int3 c0 = int3(floor(P));
            const uint salt = gSalt + 10u + layer;
            for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                const int3 c = c0 + int3(dx, dy, dz);
                const uint3 h = hashCell(c, salt);
                const uint3 h2 = pcg3d(uint3(h.x, h.y ^ 0x5bd1e995u, h.z));
                if (u01(h2.x) >= gStar.y * kStarLayerDensity[layer]) continue;
                const float3 q = float3(c) + 0.1f + 0.8f * float3(u01(h.x), u01(h.y), u01(h.z));
                const float ql = length(q);
                if (abs(ql - N) >= 0.5f) continue;
                const float3 e = r - q / ql;
                const float a2 = dot(e, e) / (rad * rad);
                if (a2 > 9.f) continue;
                const float ub = u01(h2.y);
                const float bright = 0.04f + 3.f * pow(ub, 8.f);
                const float tw = 1.f + gStar2.x * sin(time * (2.f + 4.f * u01(h2.z)) + 6.2831853f * frac(ub * 7.13f));
                const float temp = frac(ub * 13.7f + u01(h2.z) * 3.1f);
                const float3 col = lerp(float3(1.f), lerp(float3(0.72f, 0.82f, 1.f), float3(1.f, 0.78f, 0.55f), temp), gStar2.y);
                L += gStar.x * bright * tw * exp(-2.f * a2) * dust * col;
            }
        }
    }
    if ((gMask & 4u) != 0)
    {
        const uint K = gAuroraLayers;
        float3 acc = float3(0.f);
        for (uint cu = 0; cu < gCurtains; cu++)
        {
            for (uint i = 0; i < K; i++)
            {
                const float t = float(i) / float(K - 1);
                const float h = gAur.y * (1.f + gAur.z * t);
                const float px = d.x * h / d.y, pz = d.z * h / d.y;
                const float a = gAurAz.x * px + gAurAz.y * pz, b = gAurAz.y * px - gAurAz.x * pz;
                const float ts = gAur3.z;
                const float line = gAur.w + float(cu) * gAur2.x + gAur2.y * (2.f * vnoise3(float3(a * gAur2.z, ts, float(cu) * 7.3f + 0.5f), gSalt + 20u) - 1.f);
                const float db = (b - line) / gAur3.x;
                const float band = exp(-db * db);
                const float rays = 0.35f + 0.65f * vnoise3(float3(a * gAur2.w, 2.f * ts, float(cu) * 3.1f + 11.5f), gSalt + 30u);
                const float w = band * rays * exp(-3.5f * t) * exp(-length(float2(px, pz)) / gAur3.y);
                acc += w * lerp(gAurLow.rgb, gAurHigh.rgb, smoothstep(0.1f, 0.9f, t));
            }
        }
        L += acc * gAur.x * 8.f / float(K);
    }
    return L * smoothstep(0.f, gCamFwd.w, d.y) * gCamUp.w;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    const float4 src = gHasColor != 0 ? gColor[pix] : float4(0.f, 0.f, 0.f, 1.f);
    if (gHasPos != 0 && gPosW[pix].w != 0.f)
    {
        gDst[pix] = src;
        if (gHasLayer != 0) gLayer[pix] = float4(0.f);
        if (gHasLayerDepth != 0) gLayerDepth[pix] = float4(kSkyDepth, 0.f, 0.f, 1.f);
        return;
    }
    const float2 ndc = (float2(pix) + 0.5f) / float2(gFrameDim) * 2.f - 1.f;
    const float3 d = normalize(gCamFwd.xyz + ndc.x * gCamRight.xyz - ndc.y * gCamUp.xyz);
    float3 L = float3(0.f);
    if (d.y > 0.f && gMask != 0) L = layersOf(d);
    float4 o = src;
    float4 lay = float4(L, 0.f);
    if (gReplace != 0)
    {
        const float3 bg = (d.y < 0.f ? gGround.rgb : lerp(gHorizon.rgb, gZenith.rgb, sqrt(d.y))) * gCamUp.w;
        o = float4(bg + L, 1.f);
        lay = o;
    }
    else if (any(L != 0.f))
    {
        o.rgb = src.rgb + L;
    }
    gDst[pix] = o;
    if (gHasLayer != 0) gLayer[pix] = lay;
    if (gHasLayerDepth != 0) gLayerDepth[pix] = float4(kSkyDepth, 0.f, 0.f, 1.f);
}
