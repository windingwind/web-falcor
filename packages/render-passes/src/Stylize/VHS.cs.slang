/**
 * VHS (see VHS.ts). Luma/chroma split in YIQ: luma sampled with a per-line horizontal jitter (and an extra shift inside a
 * rolling tracking band), chroma low-passed horizontally (box of chromaBlur px) and offset by chromaOffset px; tape noise
 * on luma; time from gFrame. out = lerp(src, vhs, strength).
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gFrame;
    uint gSeed;
    float gChromaBlur;
    float gChromaOffset;
    float gLineJitter;
    float gTracking;
    float gTrackingSpeed;
    float gNoise;
    float gStrength;
};
Texture2D<float4> gSrc;
RWTexture2D<float4> gDst;

uint pcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rnd(uint a, uint b) { return float(pcgHash(a ^ pcgHash(b ^ pcgHash(gSeed))) >> 8) * (1.f / 16777216.f); }

float3 rgbToYiq(float3 c) { return float3(dot(c, float3(0.299f, 0.587f, 0.114f)), dot(c, float3(0.596f, -0.274f, -0.322f)), dot(c, float3(0.211f, -0.523f, 0.312f))); }
float3 yiqToRgb(float3 y) { return float3(y.x + 0.956f * y.y + 0.621f * y.z, y.x - 0.272f * y.y - 0.647f * y.z, y.x - 1.106f * y.y + 1.703f * y.z); }
float3 at(int x, int y) { return gSrc[int2(clamp(x, 0, int(gFrameDim.x) - 1), y)].rgb; }

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    const float4 src = gSrc[p];
    if (gStrength == 0.f) { gDst[p] = src; return; }
    const float H = float(gFrameDim.y);
    // tracking band: rolls down the frame at trackingSpeed frame heights per frame
    const float bandY = frac(float(gFrame) * gTrackingSpeed) * H * 1.2f - 0.1f * H;
    const float inBand = exp(-0.5f * pow((float(p.y) - bandY) / (0.03f * H), 2.f));
    const float jitter = (rnd(uint(p.y), gFrame) - 0.5f) * 2.f * gLineJitter + inBand * gTracking * 20.f * (rnd(uint(p.y) + 7919u, gFrame) - 0.2f);
    const int x = p.x + int(round(jitter));
    float Y = rgbToYiq(at(x, p.y)).x;
    float2 IQ = 0.f;
    const int R = max(int(gChromaBlur), 0);
    for (int t = -R; t <= R; t++) IQ += rgbToYiq(at(x + int(round(gChromaOffset)) + t, p.y)).yz;
    IQ /= float(2 * R + 1);
    Y += (rnd(uint(p.x) + uint(p.y) * 65537u, gFrame + 1u) - 0.5f) * gNoise * (1.f + 4.f * inBand * gTracking);
    gDst[p] = float4(lerp(src.rgb, max(yiqToRgb(float3(Y, IQ)), 0.f), gStrength), src.a);
}
