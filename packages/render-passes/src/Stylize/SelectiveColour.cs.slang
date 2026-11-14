/**
 * SelectiveColour (see SelectiveColour.ts). Hue mask in OKLCh of the mask source (src or albedo):
 * m = (1 - smoothstep(hueWidth, hueWidth + softness, |dh|)) * smoothstep(0.5 minChroma, minChroma, C);
 * s = lerp(satRest, satKeep, m); out = luma + (src - luma) * s (Rec. 709 luma of linear src).
 * OKLab: B. Ottosson, "A perceptual color space for image processing" (2020); matrices from the article
 * (reference code published under the MIT licence / public domain), re-typed here.
 */

cbuffer CB
{
    uint2 gFrameDim;
    float gKeepHue;      ///< degrees
    float gHueWidth;     ///< degrees (half width of the fully kept band)
    float gSoftness;     ///< degrees
    float gMinChroma;
    float gSatKeep;
    float gSatRest;
    uint gSource;        ///< 0 = src, 1 = albedo
    uint gHasAlbedo;
};

Texture2D<float4> gSrc;
Texture2D<float4> gAlbedo;
RWTexture2D<float4> gDst;

float3 linearSrgbToOklab(float3 c)
{
    const float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
    const float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
    const float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;
    const float l_ = sign(l) * pow(abs(l), 1.f / 3.f), m_ = sign(m) * pow(abs(m), 1.f / 3.f), s_ = sign(s) * pow(abs(s), 1.f / 3.f);
    return float3(0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
                  1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
                  0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_);
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    const float3 key = (gSource == 1 && gHasAlbedo != 0) ? gAlbedo[p].rgb : src.rgb;
    const float3 lab = linearSrgbToOklab(max(key, 0.f));
    const float C = length(lab.yz);
    const float h = degrees(atan2(lab.z, lab.y));
    float dh = abs(h - gKeepHue);
    dh = fmod(dh, 360.f);
    dh = min(dh, 360.f - dh);
    const float m = (1.f - smoothstep(gHueWidth, gHueWidth + max(gSoftness, 1e-3f), dh)) * smoothstep(0.5f * gMinChroma, max(gMinChroma, 1e-6f), C);
    const float s = lerp(gSatRest, gSatKeep, m);
    const float luma = dot(src.rgb, float3(0.2126f, 0.7152f, 0.0722f));
    gDst[p] = float4(s == 1.f ? src.rgb : luma + (src.rgb - luma) * s, src.a);
}
