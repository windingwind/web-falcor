/**
 * WorldRevealMask (see WorldRevealMask.ts). dst = lerp(a, b, mask) with mask from a signed-distance region test of the
 * pixel's world- or object-space position: sphere / plane / box, a time sweep, value-noise displacement of the distance,
 * a soft edge, an optional id filter and an edge glow. Stateless.
 * Idea after three.js webgpu_postprocessing_transition (MIT, Copyright (c) 2010-2026 three.js authors); code written for
 * this pass. Hash: PCG (Jarzynski & Olano 2020).
 */

cbuffer CB
{
    uint2 gFrameDim;
    uint gShape;
    uint gSpace;
    float3 gCenter;
    float gRadius;
    float3 gNormal;
    float gOffset;
    float3 gHalfExtents;
    float gSweep;
    float gEdgeWidth;
    float gNoiseScale;
    float gNoiseAmp;
    uint gSeed;
    uint gInvert;
    float gBackground;
    float gStrength;
    float gEdgeIntensity;
    float3 gEdgeColour;
    uint gIdMode;
    uint gIdCount;
    uint gIdChannel;
    uint gHasMask;
    uint4 gIdList0;
    uint4 gIdList1;
};

Texture2D<float4> gA;
Texture2D<float4> gB;
Texture2D<float4> gPosW;
Texture2D<float4> gPosObj;
Texture2D<float4> gIds;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gMask;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/** Hash of an integer lattice point to [0, 1). */
float latticeHash(int3 c)
{
    return float(pcg(uint(c.x) + pcg(uint(c.y) + pcg(uint(c.z) + pcg(gSeed)))) >> 8) * (1.f / 16777216.f);
}

/** 3D value noise in [0, 1): trilinear interpolation of lattice hashes with a smoothstep fade. */
float valueNoise(float3 x)
{
    const float3 fl = floor(x);
    const int3 c = int3(fl);
    const float3 f = x - fl;
    const float3 u = f * f * (3.f - 2.f * f);
    const float n000 = latticeHash(c), n100 = latticeHash(c + int3(1, 0, 0));
    const float n010 = latticeHash(c + int3(0, 1, 0)), n110 = latticeHash(c + int3(1, 1, 0));
    const float n001 = latticeHash(c + int3(0, 0, 1)), n101 = latticeHash(c + int3(1, 0, 1));
    const float n011 = latticeHash(c + int3(0, 1, 1)), n111 = latticeHash(c + int3(1, 1, 1));
    const float x00 = lerp(n000, n100, u.x), x10 = lerp(n010, n110, u.x);
    const float x01 = lerp(n001, n101, u.x), x11 = lerp(n011, n111, u.x);
    return lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z);
}

float signedDistance(float3 p)
{
    if (gShape == 0) return length(p - gCenter) - (gRadius + gSweep);
    if (gShape == 1) return dot(p, gNormal) - (gOffset + gSweep);
    const float3 q = abs(p - gCenter) - (gHalfExtents + gSweep);
    return length(max(q, float3(0.f))) + min(max(q.x, max(q.y, q.z)), 0.f);
}

bool idListed(uint id)
{
    for (uint i = 0; i < gIdCount; i++)
    {
        const uint v = i < 4 ? gIdList0[i] : gIdList1[i - 4];
        if (v == id) return true;
    }
    return false;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const int2 px = int2(tid.xy);
    if (any(px >= int2(gFrameDim))) return;
    const float4 a = gA[px];
    const float4 pos = gPosW[px];
    float m, d = 0.f;
    if (pos.w == 0.f)
        m = gBackground;
    else
    {
        const float3 p = gSpace == 0 ? pos.xyz : gPosObj[px].xyz;
        d = signedDistance(p);
        if (gNoiseAmp != 0.f) d += gNoiseAmp * (2.f * valueNoise(p * gNoiseScale) - 1.f);
        m = gEdgeWidth > 0.f ? 1.f - smoothstep(-0.5f * gEdgeWidth, 0.5f * gEdgeWidth, d) : (d < 0.f ? 1.f : 0.f);
        if (gInvert != 0) m = 1.f - m;
        if (gIdMode != 0)
        {
            const uint id = uint(gIds[px][gIdChannel]);
            const bool listed = idListed(id);
            if ((gIdMode == 1 && !listed) || (gIdMode == 2 && listed)) m = 0.f;
        }
    }
    const float edge = (pos.w != 0.f && gEdgeWidth > 0.f) ? 4.f * m * (1.f - m) : 0.f;
    const float mask = m * gStrength;
    float4 c;
    if (mask <= 0.f) c = a;
    else if (mask >= 1.f) c = gB[px];
    else c = lerp(a, gB[px], mask);
    if (gEdgeIntensity != 0.f && edge > 0.f) c.rgb += gEdgeIntensity * gStrength * edge * gEdgeColour;
    gDst[px] = c;
    if (gHasMask != 0) gMask[px] = float4(mask, d, edge, 1.f);
}
