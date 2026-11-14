/**
 * RainLayer: falling rain streaks as a colour + depth layer for DepthComposite (see RainLayer.ts).
 * Entry points, in frame order:
 *  clearAccum (per pixel) : zero the accumulators
 *  splat      (per drop)  : drop position at `time`, streak head/tail, projected and drawn as an anti-aliased line
 *                           (one pixel column/row per step along the major axis), occluded per pixel by posW;
 *                           integer atomics (fixed point, 1/4096) keep the result independent of thread order
 *  resolve    (per pixel) : accumulators -> layer colour (premultiplied rgba) and nearest streak view depth
 * Written for this pass (splat/resolve layout as in ParticleLayer). Idea of a camera-following rain volume after the
 * three.js webgpu_compute_particles_rain example (MIT, https://github.com/mrdoob/three.js); no code copied.
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gCount;
    uint gSeed;
    float3 gUp;
    float gBox;           ///< edge of the camera-centred drop volume (world units)
    float3 gWind;
    float gSpeed;         ///< fall speed (world units / s)
    float3 gColour;
    float gIntensity;
    float3 gCamPos;
    float gTime;
    float3 gCamFwd;
    float gExposure;      ///< streak length = |velocity| x exposure
    float gHalfWidth;     ///< half line width in pixels
    float gOpacity;
    float gMaxLenPx;
    float gNear;
    uint gHasPosW;
    float gSpeedJitter;
    float4 gViewProj0;
    float4 gViewProj1;
    float4 gViewProj2;
    float4 gViewProj3;
};

static const float kFar = 3.402823466e+38f;
static const float kFixed = 4096.f;

RWStructuredBuffer<Atomic<uint>> gAccum;     ///< 4 per pixel: r, g, b, coverage (fixed point)
RWStructuredBuffer<Atomic<uint>> gDepthBits; ///< 1 per pixel: asuint(min view depth)
Texture2D<float4> gPosW;
RWTexture2D<float4> gColor;
RWTexture2D<float4> gDepth;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(uint i, uint salt)
{
    return float(pcg(i ^ pcg(salt * 7919u + pcg(gSeed))) >> 8) * (1.f / 16777216.f);
}

float viewDepth(float3 x)
{
    return dot(x - gCamPos, gCamFwd);
}

float4 clip(float3 x)
{
    const float4 h = float4(x, 1.f);
    return float4(dot(gViewProj0, h), dot(gViewProj1, h), dot(gViewProj2, h), dot(gViewProj3, h));
}

float2 toPixel(float4 c)
{
    return (c.xy / c.w * float2(0.5f, -0.5f) + 0.5f) * float2(gFrameDim);
}

float sceneDepthAt(uint2 q)
{
    if (gHasPosW == 0) return kFar;
    const float4 x = gPosW[q];
    return x.w != 0.f ? viewDepth(x.xyz) : kFar;
}

/** World position and velocity of drop i at time gTime (wrapped into the box around the camera). */
void dropState(uint i, out float3 pos, out float3 vel)
{
    const float3 base = float3(rnd(i, 0u), rnd(i, 1u), rnd(i, 2u)) * gBox;
    const float s = gSpeed * (1.f + gSpeedJitter * (rnd(i, 3u) - 0.5f));
    vel = -gUp * s + gWind;
    const float3 o = gCamPos - 0.5f * gBox;
    float3 q = base + vel * gTime - o;
    q -= gBox * floor(q / gBox);
    pos = o + q;
}

[numthreads(16, 16, 1)]
void clearAccum(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const uint k = p.y * gFrameDim.x + p.x;
    for (uint c = 0; c < 4; c++) gAccum[4 * k + c].store(0u);
    gDepthBits[k].store(0xffffffffu);
}

void plot(int2 q, float w, float d)
{
    if (any(q < 0) || any(q >= int2(gFrameDim)) || w <= 0.f) return;
    if (d >= sceneDepthAt(uint2(q))) return;
    const uint k = uint(q.y) * gFrameDim.x + uint(q.x);
    const float3 e = gColour * (w * gIntensity);
    gAccum[4 * k + 0].add(uint(e.r * kFixed + 0.5f));
    gAccum[4 * k + 1].add(uint(e.g * kFixed + 0.5f));
    gAccum[4 * k + 2].add(uint(e.b * kFixed + 0.5f));
    gAccum[4 * k + 3].add(uint(w * kFixed + 0.5f));
    gDepthBits[k].min(asuint(d));
}

[numthreads(64, 1, 1)]
void splat(uint3 tid: SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gCount) return;
    float3 head, vel;
    dropState(i, head, vel);
    const float3 tail = head - vel * gExposure;
    const float dh = viewDepth(head), dt = viewDepth(tail);
    if (dh < gNear || dt < gNear) return;
    // Fade towards the box faces, so drops wrapping around the box never pop.
    const float3 rel = abs(head - gCamPos) / (0.5f * gBox);
    const float fade = saturate((1.f - max(rel.x, max(rel.y, rel.z))) * 4.f);
    if (fade <= 0.f) return;
    const float2 a = toPixel(clip(head));
    float2 b = toPixel(clip(tail));
    float2 ab = b - a;
    float len = length(ab);
    float dtail = dt;
    if (len > gMaxLenPx)
    {
        const float s = gMaxLenPx / len;
        b = a + ab * s;
        dtail = lerp(dh, dt, s);
        ab = b - a;
        len = gMaxLenPx;
    }
    if (all(max(a, b) < -4.f) || any(min(a, b) > float2(gFrameDim) + 4.f)) return;
    const float w0 = gOpacity * fade;
    // Walk the major axis one pixel at a time; cover +-(halfWidth + 1) pixels on the minor axis.
    const bool xMajor = abs(ab.x) >= abs(ab.y);
    const float2 dir = len > 1e-6f ? ab / len : float2(0.f, 1.f);
    const float2 nrm = float2(-dir.y, dir.x);
    const float a0 = xMajor ? a.x : a.y, a1 = xMajor ? b.x : b.y;
    const int m0 = int(floor(min(a0, a1))), m1 = int(floor(max(a0, a1)));
    const int r = int(ceil(gHalfWidth)) + 1;
    for (int m = m0; m <= m1; m++)
    {
        const float mc = float(m) + 0.5f;
        const float majorSpan = xMajor ? ab.x : ab.y;
        const float t = abs(majorSpan) > 1e-6f ? saturate((mc - a0) / majorSpan) : 0.f;
        const float2 c = a + ab * t;
        const float minorC = xMajor ? c.y : c.x;
        const int n0 = int(floor(minorC)) - r, n1 = int(floor(minorC)) + r;
        for (int n = n0; n <= n1; n++)
        {
            const int2 q = xMajor ? int2(m, n) : int2(n, m);
            const float2 pc = float2(q) + 0.5f;
            // distance to the segment
            const float tt = len > 1e-6f ? saturate(dot(pc - a, dir) / len) : 0.f;
            const float dist = abs(dot(pc - a, nrm));
            const float along = dot(pc - a, dir);
            const float endCov = saturate(0.5f + min(along, len - along) + 0.5f);
            const float cov = saturate(gHalfWidth + 0.5f - dist) * endCov;
            const float taper = lerp(0.35f, 1.f, 1.f - tt);   // brighter at the head
            plot(q, w0 * cov * taper, lerp(dh, dtail, tt));
        }
    }
}

[numthreads(16, 16, 1)]
void resolve(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const uint k = p.y * gFrameDim.x + p.x;
    const float4 acc = float4(float(gAccum[4 * k].load()), float(gAccum[4 * k + 1].load()), float(gAccum[4 * k + 2].load()), float(gAccum[4 * k + 3].load())) / kFixed;
    gColor[p] = float4(acc.rgb, min(acc.a, 1.f));
    const uint db = gDepthBits[k].load();
    gDepth[p] = float4(db == 0xffffffffu ? kFar : asfloat(db), 0.f, 0.f, 1.f);
}
