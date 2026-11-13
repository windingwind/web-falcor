/** DepthOfField kernel (see DepthOfField.ts). Single-pass scatter-as-gather over a golden-angle spiral. */

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
}

Texture2D<float4> gColor;
Texture2D<float4> gDepth;
RWTexture2D<float4> gDst;

float viewDepth(int2 p)
{
    const float z = gDepth[uint2(clamp(p, int2(0), int2(gFrameDim) - 1))].x;
    return z > 0.f ? z : gFarDepth;
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
    float3 acc = c0.rgb;
    float tot = 1.f;
    float r = gRadiusScale;
    float ang = 0.f;
    const float2 center = float2(pix) + 0.5f;
    for (uint i = 0; i < 4096 && r < gMaxCoC; i++)
    {
        const int2 q = int2(floor(center + float2(cos(ang), sin(ang)) * r));
        if (all(q >= 0) && all(q < int2(gFrameDim)))
        {
            const float3 cs = gColor[uint2(q)].rgb;
            const float zS = viewDepth(q);
            float sizeS = coc(zS);
            if (zS > zC) sizeS = clamp(sizeS, 0.f, sizeC * 2.f);
            const float m = smoothstep(r - 0.5f, r + 0.5f, sizeS);
            acc += lerp(acc / tot, cs, m);
            tot += 1.f;
        }
        ang += kGoldenAngle;
        r += gRadiusScale / r;
    }
    gDst[pix] = float4(acc / tot, c0.a);
}
