/**
 * StopMotion (see StopMotion.ts): per-object integer screen jitter per key and a per-key exposure gain. Stateless.
 * Gather: output pixel p takes the source pixel q within the jitter radius whose object offset maps q onto p; the nearest
 * view depth wins. Pixels that do not jitter (background, unlisted objects) are candidates for themselves; a jittering
 * pixel that no source maps onto keeps its own source colour. Hash: PCG (Jarzynski & Olano 2020).
 */

cbuffer CB
{
    uint2 gFrameDim;
    int gJitterPx;
    uint gJitterKey;
    uint gSeed;
    float gGain;
    uint gIdChannel;
    uint gIdCount;
    uint4 gIdList0;
    uint4 gIdList1;
    float3 gCamPos;
    uint gHasPosW;
    float3 gCamFwd;
};

Texture2D<float4> gSrc;
Texture2D<float4> gId;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;

static const float kFar = 3.4e38f;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

bool listed(uint id)
{
    if (gIdCount == 0) return true;
    for (uint i = 0; i < gIdCount; i++)
    {
        const uint v = i < 4 ? gIdList0[i] : gIdList1[i - 4];
        if (v == id) return true;
    }
    return false;
}

/** Pixel q: does it jitter, its id and its view depth (kFar for background). */
bool jitters(int2 q, out uint id, out float depth)
{
    const float4 idv = gId[q];
    id = uint(idv[gIdChannel]);
    bool bg = idv.a == 0.f;
    depth = 0.f;
    if (gHasPosW != 0)
    {
        const float4 pw = gPosW[q];
        bg = bg || pw.w == 0.f;
        depth = dot(pw.xyz - gCamPos, gCamFwd);
    }
    if (bg) depth = kFar;
    return !bg && listed(id);
}

int2 offsetOf(uint id)
{
    const uint n = uint(2 * gJitterPx + 1);
    const uint h = pcg(id + pcg(gJitterKey + pcg(gSeed)));
    const uint h2 = pcg(h);
    return int2(int(h % n), int(h2 % n)) - gJitterPx;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    float4 c = gSrc[p];
    if (gJitterPx > 0)
    {
        uint idP;
        float dP;
        int2 best = p;
        float bestDepth = jitters(p, idP, dP) ? kFar : dP;
        for (int dy = -gJitterPx; dy <= gJitterPx; dy++)
        {
            for (int dx = -gJitterPx; dx <= gJitterPx; dx++)
            {
                const int2 q = p - int2(dx, dy);
                if (any(q < 0) || any(q >= int2(gFrameDim))) continue;
                uint idQ;
                float dQ;
                if (!jitters(q, idQ, dQ)) continue;
                if (any(q + offsetOf(idQ) != p)) continue;
                if (dQ < bestDepth)
                {
                    bestDepth = dQ;
                    best = q;
                }
            }
        }
        c = gSrc[best];
    }
    if (gGain != 1.f) c.rgb *= gGain;
    gDst[p] = c;
}
