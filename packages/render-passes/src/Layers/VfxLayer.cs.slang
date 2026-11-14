/**
 * VfxLayer: procedural fire / electric arc / plasma orb emitters anchored in world space, evaluated per pixel into a
 * colour + depth layer for DepthComposite (see VfxLayer.ts). Written for this pass; no third-party code.
 *
 * Emitter record (4 x float4 in gEmitters):
 *   e0 = (anchor.xyz, type)            type 0 fire, 1 electric arc, 2 plasma orb
 *   e1 = (axis.xyz, size)              fire: up axis; arc: end point; orb: unused. size: flame height / arc glow width /
 *                                      orb radius (world units)
 *   e2 = (colour.rgb, intensity)
 *   e3 = (seed, speed, shape, detail)  fire: shape = half width / height, detail = turbulence [0, 1];
 *                                      arc: shape = amplitude (x arc length), detail = strand count;
 *                                      orb: shape = corona extent (x radius), detail = filament sharpness
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gCount;
    float gTime;
    float gIntensity;
    float gSoftness;
    uint gHasPosW;
    float gMaxRadiusPx;
    float4 gViewProj0;
    float4 gViewProj1;
    float4 gViewProj2;
    float4 gViewProj3;
    float3 gCamPos;
    float gPixelsPerUnitAtUnitDepth;  ///< 0.5 * frame height / tan(0.5 * fovY)
    float3 gCamFwd;
};

static const float kFar = 3.402823466e+38f;
static const float kPi = 3.14159265f;

StructuredBuffer<float4> gEmitters;
Texture2D<float4> gPosW;
RWTexture2D<float4> gColor;
RWTexture2D<float4> gDepth;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/** Lattice value in [0, 1) at integer point (x, y) for a seed. */
float lattice(int x, int y, uint seed)
{
    return float(pcg(uint(x) + pcg(uint(y) + pcg(seed))) >> 8) * (1.f / 16777216.f);
}

/** Smooth value noise in [0, 1). */
float vnoise(float2 p, uint seed)
{
    const float2 i = floor(p);
    const float2 f = p - i;
    const float2 s = f * f * (3.f - 2.f * f);
    const int x = int(i.x), y = int(i.y);
    const float a = lattice(x, y, seed), b = lattice(x + 1, y, seed), c = lattice(x, y + 1, seed), d = lattice(x + 1, y + 1, seed);
    return lerp(lerp(a, b, s.x), lerp(c, d, s.x), s.y);
}

/** 4-octave fractal value noise in [0, 1). */
float fbm(float2 p, uint seed)
{
    float sum = 0.f, amp = 0.5f;
    for (uint k = 0; k < 4; k++)
    {
        sum += amp * vnoise(p, seed + k * 101u);
        p = p * 2.f;
        amp *= 0.5f;
    }
    return sum / 0.9375f;
}

float viewDepth(float3 x)
{
    return dot(x - gCamPos, gCamFwd);
}

/** Pixel coordinates (continuous, origin top-left) of a world position; false when behind the camera. */
bool project(float3 x, out float2 px)
{
    const float4 h = float4(x, 1.f);
    const float4 c = float4(dot(gViewProj0, h), dot(gViewProj1, h), dot(gViewProj2, h), dot(gViewProj3, h));
    px = float2(0.f);
    if (c.w <= 0.f) return false;
    const float2 ndc = c.xy / c.w;
    px = (ndc * float2(0.5f, -0.5f) + 0.5f) * float2(gFrameDim);
    return true;
}

struct Sample
{
    float3 e;     ///< emission
    float a;      ///< coverage
    float depth;  ///< view depth of the sample
};

Sample emptySample()
{
    Sample s;
    s.e = float3(0.f);
    s.a = 0.f;
    s.depth = kFar;
    return s;
}

/** Fire: a flame billboard standing on the anchor along the projected up axis, shaped by domain-warped scrolling noise. */
Sample fire(float2 p, float4 e0, float4 e1, float4 e2, float4 e3)
{
    Sample s = emptySample();
    const float d = viewDepth(e0.xyz);
    float2 c0, c1;
    if (d <= 0.f || !project(e0.xyz, c0) || !project(e0.xyz + e1.xyz * e1.w, c1)) return s;
    const float S = e1.w * gPixelsPerUnitAtUnitDepth / d; // flame height in pixels when seen side-on
    const float2 ax = c1 - c0;
    const float al = length(ax);
    const float2 dir = al > 1e-4f ? ax / al : float2(0.f, -1.f);
    const float L = max(al, 0.3f * S);
    const float halfW = max(S * e3.z, 0.5f);
    const float2 q = p - c0;
    const float v = dot(q, dir) / L;
    const float u = dot(q, float2(-dir.y, dir.x)) / halfW;
    if (v < -0.2f || v > 1.3f || abs(u) > 1.8f) return s;

    const uint seed = uint(e3.x);
    const float t = gTime * e3.y;
    const float w = fbm(float2(u * 1.5f, v * 2.f - 1.3f * t), seed);
    const float n = fbm(float2(u * 2.5f + 0.8f * w, v * 3.f - 2.f * t), seed + 7u);
    const float rw = sqrt(saturate(v * 3.f + 0.35f)) * pow(saturate(1.f - v), 0.7f);
    const float hw = rw * lerp(1.f, 0.35f + 1.3f * n, saturate(e3.w));
    float dens = saturate((hw - abs(u)) / max(0.3f * hw, 0.03f)) * saturate(1.f + v * 6.f);
    dens = dens * dens * (3.f - 2.f * dens);
    const float heat = dens * (1.f - 0.65f * saturate(v)) * (0.55f + 0.7f * n);
    s.e = e2.w * (e2.rgb * saturate(1.8f * heat) + float3(1.f, 0.95f, 0.8f) * saturate(2.5f * heat - 1.4f));
    s.a = dens;
    s.depth = d;
    return s;
}

/** Electric arc between two anchors: N noisy strands pinned at both ends, re-drawn `speed` times per second. */
Sample arc(float2 p, float4 e0, float4 e1, float4 e2, float4 e3)
{
    Sample s = emptySample();
    const float d0 = viewDepth(e0.xyz), d1 = viewDepth(e1.xyz);
    float2 c0, c1;
    if (d0 <= 0.f || d1 <= 0.f || !project(e0.xyz, c0) || !project(e1.xyz, c1)) return s;
    const float2 seg = c1 - c0;
    const float L = length(seg);
    if (L < 1.f) return s;
    const float2 tg = seg / L, nrm = float2(-tg.y, tg.x);
    const float tr = dot(p - c0, tg) / L;
    const float tc = saturate(tr);
    const float dz = 1.f / lerp(1.f / d0, 1.f / d1, tc);
    const float w = clamp(e1.w * gPixelsPerUnitAtUnitDepth / dz, 0.75f, gMaxRadiusPx);
    const float amp = e3.z * L;
    if (abs(dot(p - c0, nrm)) > amp + 4.f * w || tr < -4.f * w / L || tr > 1.f + 4.f * w / L) return s;

    const uint seed = uint(e3.x);
    const uint strands = clamp(uint(e3.w), 1u, 4u);
    const float jump = floor(gTime * e3.y);
    float glow = 0.f, core = 0.f;
    for (uint k = 0; k < strands; k++)
    {
        const float n = fbm(float2(tc * 5.f + 13.7f * float(k), jump * 3.17f + 0.37f * float(k)), seed + 31u * k);
        const float disp = amp * sin(kPi * tc) * (2.f * n - 1.f) * (k == 0 ? 1.f : 1.4f);
        const float dist = length(p - (c0 + seg * tc + nrm * disp));
        const float bk = k == 0 ? 1.f : 0.5f;
        glow += bk * pow(0.25f * w / max(dist, 0.25f * w), 1.5f) * saturate(1.f - dist / (4.f * w));
        const float cr = dist / (0.15f * w + 0.5f);
        if (dist < 4.f * w) core += bk * exp(-cr * cr); // same finite support as the glow
    }
    s.e = e2.w * (e2.rgb * glow + float3(0.8f) * core);
    s.a = saturate(glow);
    s.depth = dz;
    return s;
}

/** Plasma orb: a sphere sprite with animated filament ridges, a fresnel rim and a noisy corona. */
Sample orb(float2 p, float4 e0, float4 e1, float4 e2, float4 e3)
{
    Sample s = emptySample();
    const float d = viewDepth(e0.xyz);
    float2 c;
    if (d <= 0.f || !project(e0.xyz, c)) return s;
    const float R = max(e1.w * gPixelsPerUnitAtUnitDepth / d, 0.75f);
    const float2 o = (p - c) / R;
    const float r = length(o);
    const float ext = max(e3.z, 0.f);
    if (r >= 1.f + ext) return s;
    const uint seed = uint(e3.x);
    const float t = gTime * e3.y;
    if (r < 1.f)
    {
        const float z = sqrt(1.f - r * r);
        const float n = fbm(float2(o.x * 3.f + 0.3f * t, o.y * 3.f + 2.f * z - 0.2f * t), seed);
        const float f = pow(1.f - abs(2.f * n - 1.f), max(e3.w, 1.f));
        const float rim = (1.f - z) * (1.f - z);
        s.e = e2.w * (e2.rgb * (1.5f * f + 1.2f * rim + 0.6f * z * z * z * z) + float3(0.5f) * f * f * f);
        s.a = saturate(0.35f + rim);
        s.depth = d - e1.w * z;
    }
    else
    {
        const float2 dirv = o / r;
        const float g = 0.6f * exp(-4.f * (r - 1.f) / max(ext, 1e-3f)) * (0.7f + 0.6f * fbm(dirv * 2.f + float2(0.f, 0.25f * t), seed + 3u));
        s.e = e2.w * e2.rgb * g;
        s.a = saturate(g);
        s.depth = d;
    }
    return s;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 q = tid.xy;
    if (any(q >= gFrameDim)) return;
    float3 rgb = float3(0.f);
    float cov = 0.f, depth = kFar;
    if (gIntensity != 0.f)
    {
        const float2 p = float2(q) + 0.5f;
        float sceneDepth = kFar;
        if (gHasPosW != 0)
        {
            const float4 x = gPosW[q];
            if (x.w != 0.f) sceneDepth = viewDepth(x.xyz);
        }
        for (uint i = 0; i < gCount; i++)
        {
            const float4 e0 = gEmitters[4 * i], e1 = gEmitters[4 * i + 1], e2 = gEmitters[4 * i + 2], e3 = gEmitters[4 * i + 3];
            const uint type = uint(e0.w);
            Sample s;
            if (type == 0u) s = fire(p, e0, e1, e2, e3);
            else if (type == 1u) s = arc(p, e0, e1, e2, e3);
            else s = orb(p, e0, e1, e2, e3);
            if (!(s.a > 0.f)) continue; // every emitter has coverage > 0 wherever it emits
            float vis = s.depth < sceneDepth ? 1.f : 0.f;
            if (gSoftness > 0.f && sceneDepth < kFar) vis = saturate((sceneDepth - s.depth) / gSoftness);
            if (vis <= 0.f) continue;
            rgb += s.e * vis;
            cov += s.a * vis;
            depth = min(depth, s.depth);
        }
        rgb *= gIntensity;
    }
    gColor[q] = float4(rgb, min(cov, 1.f));
    gDepth[q] = float4(depth, 0.f, 0.f, 1.f);
}
