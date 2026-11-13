/** DepthOfField kernel (see DepthOfField.ts). Single-pass, area-normalised scatter-as-gather over a golden-angle spiral. */

static const float kGoldenAngle = 2.39996323f;

cbuffer CB
{
    uint2 gFrameDim;
    float gCocScale;       // A * H / (2 tan(fovY/2) * zf)
    float gFocalDistance;
    float gMaxCoC;
    float gRadiusScale;
    float gFarDepth;
    float _pad0;
    float3 gCamPos;
    float _pad1;
    float3 gCamFwd;
    float _pad2;
}

Texture2D<float4> gColor;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;

float viewDepth(int2 p)
{
    const float4 pw = gPosW[uint2(clamp(p, int2(0), int2(gFrameDim) - 1))];
    return pw.w != 0.f ? max(dot(pw.xyz - gCamPos, gCamFwd), 1e-6f) : gFarDepth;
}

float coc(float z) { return min(gCocScale * abs(z - gFocalDistance) / max(z, 1e-6f), gMaxCoC); }

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    const float4 c0 = gColor[pix];
    if (gCocScale <= 0.f) { gDst[pix] = c0; return; }
    const float zC = viewDepth(int2(pix));
    const float sizeC = coc(zC);
    // Scatter-as-gather: every source pixel spreads its colour uniformly over its CoC disk (weight 1 / disk area);
    // a tap at distance r receives coverage saturate(coc - r + 0.5). Spiral taps each stand for an area of
    // 2 pi s (s = gRadiusScale), the centre tap for pi s^2. Taps behind the centre have their CoC clamped to twice
    // the centre's, which keeps blurred background from bleeding over a sharp foreground.
    const float s = gRadiusScale;
    float wC = s * s / max(sizeC * sizeC, 0.25f);
    float3 acc = c0.rgb * wC;
    float tot = wC;
    float r = s;
    float ang = 0.f;
    const float2 center = float2(pix) + 0.5f;
    for (uint i = 0; i < 16384 && r < gMaxCoC + 0.5f; i++)
    {
        const int2 q = int2(floor(center + float2(cos(ang), sin(ang)) * r));
        if (all(q >= 0) && all(q < int2(gFrameDim)))
        {
            const float zS = viewDepth(q);
            float sizeS = coc(zS);
            if (zS > zC) sizeS = min(sizeS, sizeC * 2.f);
            const float rr = length(float2(q) + 0.5f - center); // the snapped tap's true distance
            const float w = 2.f * s * saturate(sizeS - rr + 0.5f) / max(sizeS * sizeS, 0.25f);
            acc += w * gColor[uint2(q)].rgb;
            tot += w;
        }
        ang += kGoldenAngle;
        r += s / r;
    }
    gDst[pix] = float4(acc / tot, c0.a);
}
