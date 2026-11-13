/** FXAA 3.11 quality (Lottes 2011) in pixel coordinates; see FXAA.ts. */

cbuffer CB
{
    uint2 gFrameDim;
    float gEdgeThresholdMin;
    float gEdgeThresholdMax;
    float gSubpixelQuality;
    uint gIterations;
    uint gEnabled;
    float _pad0;
}

Texture2D<float4> gSrc;
RWTexture2D<float4> gDst;

float4 fetch(int2 p) { return gSrc[uint2(clamp(p, int2(0), int2(gFrameDim) - 1))]; }

/** Bilinear filter at pixel-space position p (pixel centres at integer + 0.5), clamp-to-edge. */
float4 bilerp(float2 p)
{
    const float2 q = p - 0.5f;
    const float2 f = q - floor(q);
    const int2 i = int2(floor(q));
    return lerp(lerp(fetch(i), fetch(i + int2(1, 0)), f.x), lerp(fetch(i + int2(0, 1)), fetch(i + int2(1, 1)), f.x), f.y);
}

float luma(float4 c) { return dot(saturate(c.rgb), float3(0.299f, 0.587f, 0.114f)); }
float lumaAt(float2 p) { return luma(bilerp(p)); }

float stepQuality(uint i)
{
    // FXAA 3.11 quality quality-preset schedule for the end-of-edge search
    // 1 x5, 1.5, 2 x4, 4, 8
    if (i < 5) return 1.f;
    if (i == 5) return 1.5f;
    if (i < 10) return 2.f;
    return i == 10 ? 4.f : 8.f;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    const int2 ip = int2(pix);
    const float4 cC = fetch(ip);
    if (gEnabled == 0) { gDst[pix] = cC; return; }
    // y grows downwards: "up" is -y
    const float lC = luma(cC);
    const float lD = luma(fetch(ip + int2(0, 1))), lU = luma(fetch(ip + int2(0, -1)));
    const float lL = luma(fetch(ip + int2(-1, 0))), lR = luma(fetch(ip + int2(1, 0)));
    const float lMin = min(lC, min(min(lD, lU), min(lL, lR)));
    const float lMax = max(lC, max(max(lD, lU), max(lL, lR)));
    const float range = lMax - lMin;
    if (range < max(gEdgeThresholdMin, lMax * gEdgeThresholdMax)) { gDst[pix] = cC; return; }

    const float lDL = luma(fetch(ip + int2(-1, 1))), lUR = luma(fetch(ip + int2(1, -1)));
    const float lUL = luma(fetch(ip + int2(-1, -1))), lDR = luma(fetch(ip + int2(1, 1)));
    const float lDU = lD + lU, lLR = lL + lR;
    const float lLeftC = lDL + lUL, lDownC = lDL + lDR, lRightC = lDR + lUR, lUpC = lUR + lUL;
    const float edgeH = abs(-2.f * lL + lLeftC) + abs(-2.f * lC + lDU) * 2.f + abs(-2.f * lR + lRightC);
    const float edgeV = abs(-2.f * lU + lUpC) + abs(-2.f * lC + lLR) * 2.f + abs(-2.f * lD + lDownC);
    const bool isH = edgeH >= edgeV;

    const float l1 = isH ? lU : lL; // towards -y / -x
    const float l2 = isH ? lD : lR; // towards +y / +x
    const float g1 = l1 - lC, g2 = l2 - lC;
    const bool is1Steepest = abs(g1) >= abs(g2);
    const float gradScaled = 0.25f * max(abs(g1), abs(g2));
    float stepLen = 1.f; // pixels, perpendicular to the edge
    float lLocalAvg;
    if (is1Steepest) { stepLen = -stepLen; lLocalAvg = 0.5f * (l1 + lC); }
    else lLocalAvg = 0.5f * (l2 + lC);

    const float2 uv = float2(pix) + 0.5f;
    float2 cur = uv;
    if (isH) cur.y += stepLen * 0.5f; else cur.x += stepLen * 0.5f;
    const float2 off = isH ? float2(1.f, 0.f) : float2(0.f, 1.f);
    float2 p1 = cur - off, p2 = cur + off;
    float e1 = lumaAt(p1) - lLocalAvg, e2 = lumaAt(p2) - lLocalAvg;
    bool r1 = abs(e1) >= gradScaled, r2 = abs(e2) >= gradScaled;
    if (!r1) p1 -= off;
    if (!r2) p2 += off;
    for (uint i = 2; i < gIterations && !(r1 && r2); i++)
    {
        if (!r1) { e1 = lumaAt(p1) - lLocalAvg; r1 = abs(e1) >= gradScaled; }
        if (!r2) { e2 = lumaAt(p2) - lLocalAvg; r2 = abs(e2) >= gradScaled; }
        if (!r1) p1 -= off * stepQuality(i);
        if (!r2) p2 += off * stepQuality(i);
    }
    const float d1 = isH ? (uv.x - p1.x) : (uv.y - p1.y);
    const float d2 = isH ? (p2.x - uv.x) : (p2.y - uv.y);
    const bool dir1 = d1 < d2;
    const float dFinal = min(d1, d2);
    const float thickness = d1 + d2;
    const float pixelOffset = -dFinal / thickness + 0.5f;
    const bool centerSmaller = lC < lLocalAvg;
    const bool correct = ((dir1 ? e1 : e2) < 0.f) != centerSmaller;
    float finalOffset = correct ? pixelOffset : 0.f;

    const float lAvg = (1.f / 12.f) * (2.f * (lDU + lLR) + lLeftC + lRightC);
    const float s1 = saturate(abs(lAvg - lC) / range);
    const float s2 = (-2.f * s1 + 3.f) * s1 * s1;
    finalOffset = max(finalOffset, s2 * s2 * gSubpixelQuality);

    float2 fuv = uv;
    if (isH) fuv.y += finalOffset * stepLen; else fuv.x += finalOffset * stepLen;
    const float4 c = bilerp(fuv);
    gDst[pix] = float4(c.rgb, cC.a);
}
