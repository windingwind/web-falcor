/**
 * TiltShift (see TiltShift.ts). Blur radius per pixel from its signed distance to a focus band (a line through
 * (0.5, bandCenter) at `angle`, in units of the image height): sigma = maxBlur * smoothstep(0, feather, |dist| - bandWidth / 2).
 * Separable Gaussian: hMain (src -> tmp), vMain (tmp -> dst), both with the sigma of the output pixel.
 */
cbuffer CB
{
    uint2 gFrameDim;
    float gBandCenter;
    float gBandWidth;
    float gFeather;
    float gMaxBlur;     ///< sigma in pixels at full blur
    float2 gNormal;     ///< unit normal of the band line in (x / H, y / H) units
};
Texture2D<float4> gIn;
RWTexture2D<float4> gOut;

float sigmaAt(int2 p)
{
    const float2 q = (float2(p) + 0.5f - float2(0.5f * float(gFrameDim.x), gBandCenter * float(gFrameDim.y))) / float(gFrameDim.y);
    const float d = abs(dot(q, gNormal)) - 0.5f * gBandWidth;
    return gMaxBlur * smoothstep(0.f, max(gFeather, 1e-4f), d);
}

float4 blur(int2 p, int2 dir)
{
    const float s = sigmaAt(p);
    if (s < 0.05f) return gIn[p];
    const int R = int(ceil(3.f * s));
    float4 acc = 0.f;
    float w = 0.f;
    for (int t = -R; t <= R; t++)
    {
        const float k = exp(-0.5f * float(t * t) / (s * s));
        acc += k * gIn[clamp(p + dir * t, int2(0), int2(gFrameDim) - 1)];
        w += k;
    }
    return acc / w;
}

[numthreads(16, 16, 1)]
void hMain(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    gOut[p] = blur(p, int2(1, 0));
}

[numthreads(16, 16, 1)]
void vMain(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    gOut[p] = blur(p, int2(0, 1));
}
