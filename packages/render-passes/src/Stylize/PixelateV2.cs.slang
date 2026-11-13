/**
 * PixelateV2 (see PixelateV2.ts). Three entry points over a grid of block x block cells (shifted by gOffset):
 *  lowMain  : one thread per cell. avg = mean of the cell's pixels; nearest = the pixel at the cell centre;
 *             majority = mean of the pixels in the most frequent colour bin (4 bits per sRGB-encoded channel).
 *             gLow = colour, gLowGeo = (view depth, n.x, n.y, sign of n.z as 1|2) of the centre pixel, 0 = background.
 *  edgeMain : one thread per cell. 1-cell outline on the low-res grid: the nearer cell of a depth jump (> threshold,
 *             relative) or a normal crease, or a geometry cell next to background. gLowOut = colour * (1 - outline).
 *  upMain   : nearest upsampling of gLowOut to full resolution.
 */

cbuffer CB
{
    uint2 gFrameDim;
    uint2 gLowDim;
    int2 gOffset;
    uint gBlock;
    uint gMode;          ///< 0 avg, 1 nearest, 2 majority
    float gOutline;
    float gOutlineThreshold;
    float gNormalThreshold;
    float3 gCamPos;
    float3 gCamFwd;
    uint gHasPosW;
    uint gHasNormW;
};

Texture2D<float4> gSrc;
Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gLowIn;
Texture2D<float4> gLowGeoIn;
RWTexture2D<float4> gLow;
RWTexture2D<float4> gLowGeo;
RWTexture2D<float4> gDst;

float encodeSrgb(float x) { x = saturate(x); return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.f / 2.4f) - 0.055f; }
uint binOf(float3 c) { return (uint(encodeSrgb(c.r) * 15.f + 0.5f) << 8) | (uint(encodeSrgb(c.g) * 15.f + 0.5f) << 4) | uint(encodeSrgb(c.b) * 15.f + 0.5f); }

int2 cellOrigin(int2 cell) { return cell * int(gBlock) - gOffset; }

[numthreads(8, 8, 1)]
void lowMain(uint3 tid: SV_DispatchThreadID)
{
    const int2 cell = int2(tid.xy);
    if (any(cell >= int2(gLowDim))) return;
    const int2 o = cellOrigin(cell);
    const int2 lo = max(o, 0), hi = min(o + int(gBlock), int2(gFrameDim));   // [lo, hi)
    const int2 centre = clamp(o + int(gBlock) / 2, lo, hi - 1);
    float4 colour = float4(0.f);
    if (gMode == 1)
    {
        colour = gSrc[centre];
    }
    else if (gMode == 0)
    {
        for (int y = lo.y; y < hi.y; y++) for (int x = lo.x; x < hi.x; x++) colour += gSrc[int2(x, y)];
        colour /= float((hi.x - lo.x) * (hi.y - lo.y));
    }
    else
    {
        uint bestCount = 0, bestBin = 0;
        for (int y = lo.y; y < hi.y; y++) for (int x = lo.x; x < hi.x; x++)
        {
            const uint b = binOf(gSrc[int2(x, y)].rgb);
            uint count = 0;
            for (int y2 = lo.y; y2 < hi.y; y2++) for (int x2 = lo.x; x2 < hi.x; x2++) count += binOf(gSrc[int2(x2, y2)].rgb) == b ? 1u : 0u;
            if (count > bestCount) { bestCount = count; bestBin = b; }
        }
        for (int y = lo.y; y < hi.y; y++) for (int x = lo.x; x < hi.x; x++)
        {
            const float4 c = gSrc[int2(x, y)];
            if (binOf(c.rgb) == bestBin) colour += c;
        }
        colour /= float(bestCount);
    }
    gLow[cell] = colour;
    float4 geo = float4(0.f, 0.f, 0.f, 0.f);   // w = 1: geometry
    if (gHasPosW != 0)
    {
        const float4 pos = gPosW[centre];
        if (pos.w != 0.f)
        {
            const float3 n = gHasNormW != 0 ? normalize(gNormW[centre].xyz) : float3(0.f);
            // (view depth, n.x, n.y, 1 if n.z >= 0 else 2); w = 0 marks background
            geo = float4(dot(pos.xyz - gCamPos, gCamFwd), n.x, n.y, n.z >= 0.f ? 1.f : 2.f);
        }
    }
    gLowGeo[cell] = geo;
}

float3 unpackN(float4 g) { const float z = sqrt(saturate(1.f - g.y * g.y - g.z * g.z)); return float3(g.y, g.z, g.w > 1.5f ? -z : z); }

[numthreads(8, 8, 1)]
void edgeMain(uint3 tid: SV_DispatchThreadID)
{
    const int2 cell = int2(tid.xy);
    if (any(cell >= int2(gLowDim))) return;
    const float4 colour = gLowIn[cell];
    const float4 g = gLowGeoIn[cell];
    float edge = 0.f;
    if (gOutline > 0.f && g.w != 0.f)
    {
        const int2 offs[4] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1) };
        for (uint i = 0; i < 4; i++)
        {
            const int2 q = cell + offs[i];
            if (any(q < 0) || any(q >= int2(gLowDim))) continue;
            const float4 gq = gLowGeoIn[q];
            if (gq.w == 0.f) { edge = 1.f; continue; }
            if (gq.x < g.x) continue;
            if (gq.x - g.x > gOutlineThreshold * g.x) edge = 1.f;
            else if (gHasNormW != 0 && 1.f - dot(unpackN(g), unpackN(gq)) > gNormalThreshold) edge = 1.f;
        }
    }
    gLow[cell] = float4(colour.rgb * (1.f - gOutline * edge), colour.a);
}

[numthreads(16, 16, 1)]
void upMain(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    const int2 cell = (p + gOffset) / int(gBlock);
    gDst[p] = gLowIn[cell];
}
