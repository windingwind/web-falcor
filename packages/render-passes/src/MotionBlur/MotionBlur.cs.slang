/** MotionBlur kernels (see MotionBlur.ts): McGuire et al. 2012 reconstruction filter.
 *  Velocities are half-extents in pixels: V = gHalfExtentScale * mvec * frameDim, |V| <= gTileSize. */

cbuffer CB
{
    uint2 gFrameDim;
    uint2 gTileDim;
    uint gTileSize;
    float gHalfExtentScale;  // 0.5 * shutter
    uint gSampleCount;
    float gSoftZ;            // soft depth-compare extent, relative to the centre depth
    uint gNoise;
    uint gFrame;
    float2 _pad0;
    float3 gCamPos;
    float _pad1;
    float3 gCamFwd;
    float _pad2;
}

Texture2D<float4> gColor;
Texture2D<float4> gMvec;
Texture2D<float4> gPosW;
Texture2D<float4> gTileIn;
RWTexture2D<float4> gOut;

float2 velocity(uint2 p)
{
    float2 v = gMvec[p].xy * float2(gFrameDim) * gHalfExtentScale;
    const float l = length(v);
    if (l > float(gTileSize)) v *= float(gTileSize) / l;
    return v;
}

[numthreads(8, 8, 1)]
void tileMaxMain(uint3 tid: SV_DispatchThreadID)
{
    const uint2 t = tid.xy;
    if (any(t >= gTileDim)) return;
    float2 best = float2(0.f);
    float bestL = 0.f;
    for (uint y = 0; y < gTileSize; y++)
    {
        for (uint x = 0; x < gTileSize; x++)
        {
            const uint2 p = t * gTileSize + uint2(x, y);
            if (any(p >= gFrameDim)) continue;
            const float2 v = velocity(p);
            const float l = dot(v, v);
            if (l > bestL) { bestL = l; best = v; }
        }
    }
    gOut[t] = float4(best, 0.f, 1.f);
}

[numthreads(8, 8, 1)]
void neighborMaxMain(uint3 tid: SV_DispatchThreadID)
{
    const int2 t = int2(tid.xy);
    if (any(t >= int2(gTileDim))) return;
    float2 best = float2(0.f);
    float bestL = 0.f;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            const int2 q = t + int2(dx, dy);
            if (any(q < 0) || any(q >= int2(gTileDim))) continue;
            const float2 v = gTileIn[uint2(q)].xy;
            const float l = dot(v, v);
            if (l > bestL) { bestL = l; best = v; }
        }
    }
    gOut[uint2(t)] = float4(best, 0.f, 1.f);
}

float depthAt(uint2 p)
{
    const float4 pw = gPosW[p];
    return pw.w != 0.f ? dot(pw.xyz - gCamPos, gCamFwd) : 1e30f;
}

float softDepthCompare(float za, float zb, float extent) { return saturate(1.f - (za - zb) / extent); }
float cone(float dist, float vLen) { return saturate(1.f - dist / vLen); }
float cylinder(float dist, float vLen) { return 1.f - smoothstep(0.95f * vLen, 1.05f * vLen, dist); }
float ign(float2 p) { return frac(52.9829189f * frac(dot(p, float2(0.06711056f, 0.00583715f)))); }

[numthreads(16, 16, 1)]
void gatherMain(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    const float4 cX = gColor[pix];
    const float2 vN = gTileIn[min(pix / gTileSize, gTileDim - 1)].xy;
    if (length(vN) <= 0.5f) { gOut[pix] = cX; return; }

    const float2 vX = velocity(pix);
    const float vXLen = max(length(vX), 0.5f);
    const float zX = depthAt(pix);
    const float extent = max(gSoftZ * min(zX, 1e6f), 1e-4f);
    float weight = 1.f / vXLen;
    float3 sum = cX.rgb * weight;
    const float j = gNoise != 0 ? ign(float2(pix) + float(gFrame % 64u) * 5.588238f) - 0.5f : 0.f;
    const uint S = gSampleCount;
    const float2 X = float2(pix) + 0.5f;
    for (uint i = 0; i < S; i++)
    {
        if (i == (S - 1) / 2) continue;
        const float t = lerp(-1.f, 1.f, (float(i) + j + 1.f) / float(S + 1));
        const float2 Yf = X + vN * t;
        const int2 Yi = int2(floor(Yf));
        if (any(Yi < 0) || any(Yi >= int2(gFrameDim))) continue;
        const uint2 Y = uint2(Yi);
        const float zY = depthAt(Y);
        const float dist = length(vN * t);
        // softDepthCompare(za, zb) = 1 when za is closer (positive view depth): f = Y in front of X, b = Y behind X
        const float f = softDepthCompare(zY, zX, extent);
        const float b = softDepthCompare(zX, zY, extent);
        const float vYLen = max(length(velocity(Y)), 0.5f);
        const float a = f * cone(dist, vYLen) + b * cone(dist, vXLen) + cylinder(dist, vYLen) * cylinder(dist, vXLen) * 2.f;
        weight += a;
        sum += a * gColor[Y].rgb;
    }
    gOut[pix] = float4(sum / weight, cX.a);
}
