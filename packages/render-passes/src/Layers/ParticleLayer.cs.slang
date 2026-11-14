/**
 * ParticleLayer: GPU particle simulation + splatting into a colour/depth layer (see ParticleLayer.ts).
 * Written for this pass (no third-party code). Entry points, in frame order:
 *  init      (per particle) : spawn all particles from the seed (on a history reset)
 *  simulate  (per particle) : one integration step with screen-space depth-buffer collision and respawn
 *  clearAccum(per pixel)    : zero the splat accumulators
 *  splat     (per particle) : soft disc splat with integer atomics (order-independent, deterministic)
 *  resolve   (per pixel)    : accumulators -> layer colour (premultiplied rgba) and nearest particle depth
 * Particle state: gState[2i] = (position.xyz, age), gState[2i+1] = (velocity.xyz, generation).
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gCount;
    uint gSeed;
    float3 gBoundsMin;
    float gDt;
    float3 gBoundsMax;
    float gTime;
    float3 gGravity;
    float gLifetime;
    float3 gWind;
    float gDrag;
    float3 gColour;
    float gTurbulence;
    float gTurbulenceScale;
    float gColourJitter;
    float gIntensity;
    float gTwinkle;
    float gTwinkleRate;
    float gSize;
    float gMaxRadiusPx;
    float gRestitution;
    float gThickness;
    uint gCollide;
    uint gHasPosW;
    uint gHasNormW;
    float4 gViewProj0;
    float4 gViewProj1;
    float4 gViewProj2;
    float4 gViewProj3;
    float3 gCamPos;
    float gPixelsPerUnitAtUnitDepth;  ///< 0.5 * frame height / tan(0.5 * fovY)
    float3 gCamFwd;
};

static const float kFar = 3.402823466e+38f;
static const float kFixed = 4096.f;   ///< fixed-point scale of the splat accumulators

RWStructuredBuffer<float4> gState;
RWStructuredBuffer<Atomic<uint>> gAccum;    ///< 4 per pixel: r, g, b, coverage (fixed point)
RWStructuredBuffer<Atomic<uint>> gDepthBits; ///< 1 per pixel: asuint(min view depth)
Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
RWTexture2D<float4> gColor;
RWTexture2D<float4> gDepth;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/** Uniform [0, 1) from (particle, generation, salt). */
float rnd(uint i, uint gen, uint salt)
{
    return float(pcg(i ^ pcg(gen * 1013u + salt * 7919u + pcg(gSeed))) >> 8) * (1.f / 16777216.f);
}

float3 rnd3(uint i, uint gen, uint salt)
{
    return float3(rnd(i, gen, salt), rnd(i, gen, salt + 1u), rnd(i, gen, salt + 2u));
}

void spawn(uint i, uint gen, float age)
{
    const float3 pos = lerp(gBoundsMin, gBoundsMax, rnd3(i, gen, 0u));
    const float3 vel = gWind + (rnd3(i, gen, 3u) - 0.5f) * 0.1f * gTurbulence;
    gState[2 * i] = float4(pos, age);
    gState[2 * i + 1] = float4(vel, float(gen));
}

float particleLifetime(uint i, uint gen)
{
    return gLifetime * (0.5f + rnd(i, gen, 6u));
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

/** Scene view depth at a pixel (+inf for background or without posW). */
float sceneDepthAt(uint2 q)
{
    if (gHasPosW == 0) return kFar;
    const float4 x = gPosW[q];
    return x.w != 0.f ? viewDepth(x.xyz) : kFar;
}

[numthreads(64, 1, 1)]
void init(uint3 tid: SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gCount) return;
    spawn(i, 0u, rnd(i, 0u, 7u) * particleLifetime(i, 0u)); // staggered ages: no synchronous respawn wave
}

[numthreads(64, 1, 1)]
void simulate(uint3 tid: SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gCount) return;
    const float4 s0 = gState[2 * i], s1 = gState[2 * i + 1];
    float3 p = s0.xyz, v = s1.xyz;
    float age = s0.w + gDt;
    const uint gen = uint(s1.w);

    const float3 ph = rnd3(i, gen, 9u) * 6.2831853f;
    const float f = gTurbulenceScale;
    const float3 noise = float3(sin(f * p.y + 1.7f * gTime + ph.x), sin(f * p.z + 1.3f * gTime + ph.y), sin(f * p.x + 1.1f * gTime + ph.z));
    v += (gGravity + gTurbulence * noise) * gDt;
    v = lerp(v, gWind, saturate(gDrag * gDt));
    float3 pn = p + v * gDt;

    if (gCollide != 0 && gHasPosW != 0)
    {
        float2 px;
        if (project(pn, px) && all(px >= 0.f) && all(px < float2(gFrameDim)))
        {
            const uint2 q = uint2(px);
            const float ds = sceneDepthAt(q), dp = viewDepth(pn);
            if (ds < kFar && dp > ds && dp < ds * (1.f + gThickness))
            {
                const float3 n = gHasNormW != 0 ? normalize(gNormW[q].xyz) : -gCamFwd;
                const float vn = dot(v, n);
                if (vn < 0.f) v -= (1.f + gRestitution) * vn * n;
                pn = p; // stay in front of the surface this step
            }
        }
    }

    const float3 ext = (gBoundsMax - gBoundsMin) * 0.05f + 1e-4f;
    const bool outside = any(pn < gBoundsMin - ext) || any(pn > gBoundsMax + ext);
    if (age > particleLifetime(i, gen) || outside)
    {
        spawn(i, gen + 1u, 0.f);
        return;
    }
    gState[2 * i] = float4(pn, age);
    gState[2 * i + 1] = float4(v, float(gen));
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

[numthreads(64, 1, 1)]
void splat(uint3 tid: SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gCount) return;
    const float4 s0 = gState[2 * i], s1 = gState[2 * i + 1];
    const uint gen = uint(s1.w);
    const float3 x = s0.xyz;
    const float d = viewDepth(x);
    float2 c;
    if (d <= 0.f || !project(x, c)) return;
    const float R = clamp(gSize * gPixelsPerUnitAtUnitDepth / d, 0.75f, gMaxRadiusPx);

    const float3 jit = (rnd3(i, gen, 12u) - 0.5f) * 2.f * gColourJitter;
    const float3 col = max(gColour * (1.f + jit), 0.f);
    const float tw = 1.f - gTwinkle * 0.5f * (1.f + sin(6.2831853f * (gTime * gTwinkleRate + rnd(i, gen, 15u))));
    const float bright = gIntensity * tw;

    const int2 lo = max(int2(floor(c - R)), int2(0));
    const int2 hi = min(int2(ceil(c + R)), int2(gFrameDim) - 1);
    const uint dbits = asuint(d);
    for (int py = lo.y; py <= hi.y; py++)
    {
        for (int pxl = lo.x; pxl <= hi.x; pxl++)
        {
            const float2 o = float2(pxl, py) + 0.5f - c;
            const float r2 = dot(o, o) / (R * R);
            if (r2 >= 1.f) continue;
            const uint2 q = uint2(pxl, py);
            if (d >= sceneDepthAt(q)) continue;
            const float w = (1.f - r2) * (1.f - r2);
            const uint k = q.y * gFrameDim.x + q.x;
            const float3 e = col * (w * bright);
            gAccum[4 * k + 0].add(uint(e.r * kFixed + 0.5f));
            gAccum[4 * k + 1].add(uint(e.g * kFixed + 0.5f));
            gAccum[4 * k + 2].add(uint(e.b * kFixed + 0.5f));
            gAccum[4 * k + 3].add(uint(w * kFixed + 0.5f));
            gDepthBits[k].min(dbits);
        }
    }
}

[numthreads(16, 16, 1)]
void resolve(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const uint k = p.y * gFrameDim.x + p.x;
    const float4 a = float4(float(gAccum[4 * k].load()), float(gAccum[4 * k + 1].load()), float(gAccum[4 * k + 2].load()), float(gAccum[4 * k + 3].load())) / kFixed;
    gColor[p] = float4(a.rgb, min(a.a, 1.f));
    const uint db = gDepthBits[k].load();
    gDepth[p] = float4(db == 0xffffffffu ? kFar : asfloat(db), 0.f, 0.f, 1.f);
}
