/**
 * PaletteQuant (see PaletteQuant.ts). Nearest palette colour in OKLab (or linear RGB) after an optional ordered-dither
 * offset; or per-channel quantisation of the sRGB-encoded value to 2^bits levels (bits > 0, e.g. RGB555).
 * out = lerp(src, quantised, mix).
 */

static const uint kMaxColours = 16;

cbuffer CB
{
    uint2 gFrameDim;
    uint gCount;
    uint gSpace;            ///< 0 = oklab, 1 = linear rgb
    float4 gPalette[kMaxColours];     ///< linear rgb
    float4 gPaletteLab[kMaxColours];  ///< the same colours in the comparison space
    float gDither;          ///< ordered-dither amplitude in units of the palette spacing
    float gSpacing;         ///< mean nearest-neighbour distance of the palette (comparison space) or 1 / (levels - 1)
    float gMix;
    uint gBits;             ///< > 0: per-channel quantisation of the sRGB-encoded value
};

Texture2D<float4> gSrc;
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

float encodeSrgb(float x) { x = saturate(x); return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.f / 2.4f) - 0.055f; }
float decodeSrgb(float x) { return x <= 0.04045f ? x / 12.92f : pow((x + 0.055f) / 1.055f, 2.4f); }

/** 4x4 Bayer threshold in (-0.5, 0.5). */
float bayer4(uint2 p)
{
    const uint m[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    return (float(m[(p.y & 3u) * 4u + (p.x & 3u)]) + 0.5f) / 16.f - 0.5f;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    if (gMix == 0.f) { gDst[p] = src; return; }
    const float t = gDither * bayer4(p) * gSpacing;
    float3 q;
    if (gBits > 0)
    {
        const float L = float((1u << gBits) - 1u);
        float3 e = float3(encodeSrgb(src.r), encodeSrgb(src.g), encodeSrgb(src.b)) + t;
        e = floor(saturate(e) * L + 0.5f) / L;
        q = float3(decodeSrgb(e.r), decodeSrgb(e.g), decodeSrgb(e.b));
    }
    else
    {
        float3 x = gSpace == 0 ? linearSrgbToOklab(max(src.rgb, 0.f)) : src.rgb;
        if (gSpace == 0) x.x += t;   // OKLab: dither along lightness
        else x += t;                 // rgb: along the grey axis
        float best = 1e30f;
        uint bi = 0;
        for (uint i = 0; i < gCount; i++)
        {
            const float3 d = x - gPaletteLab[i].xyz;
            const float dd = dot(d, d);
            if (dd < best) { best = dd; bi = i; }
        }
        q = gPalette[bi].rgb;
    }
    gDst[p] = float4(lerp(src.rgb, q, gMix), src.a);
}
