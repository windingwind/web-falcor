/**
 * WebFalcor override of RenderPasses/SimplePostFX/SimplePostFX.cs.slang
 * (docs §4.3). Diffs vs upstream:
 *  - gDst is write-only (WTexture2D + Store; WGSL restricts read_write storage
 *    textures to r32*). The in-place read `gDst[pixel]` in upsample reads a new
 *    `gDstPrev` texture instead; the web host ping-pongs a second pyramid
 *    (WebGPU also forbids sampling and storage-writing one texture in a pass).
 *  - `bool gInPlace` is `uint` (bool is non-host-shareable in WGSL uniforms).
 *  - Border-addressed sampling (WebGPU has no border mode) is emulated with a
 *    manual bilinear (`sampleBorder`, zero outside), using the new `gSrcRes`
 *    uniform = resolution of the sampled texture. gLinearSampler is removed.
 *    Weights use exact float math instead of the HW's fixed-point fractions.
 */

#include "Utils/Color/ColorHelpers.slang"

cbuffer PerFrameCB
{
    uint2 gResolution;
    float2 gInvRes;
    float gBloomAmount;
    float gVignetteAmount;
    float gChromaticAberrationAmount;
    float2 gBarrelDistort;
    float3 gSaturationCurve;
    float3 gColorOffset;
    float3 gColorScale;
    float3 gColorPower;
    float gStar;
    float2 gStarDir1;
    float2 gStarDir2;
    float2 gStarDir3;
    float gWipe;
    uint gInPlace;
    float2 gSrcRes; ///< web-only: resolution of the texture sampled via sampleBorder.
}

Texture2D<float4> gSrc;
Texture2D<float4> gBloomed;
Texture2D<float4> gDstPrev; ///< web-only: previous gDst content for the in-place paths.
WTexture2D<float4> gDst; ///< runPostFX destination (rgba32float graph output).
[format("rgba16f")]
WTexture2D<float4> gDstMip; ///< web-only: pyramid destination (WGSL needs one static format per storage binding).

/**
 * Bilinear tap with zero border color (emulates the native border sampler).
 * Fractions are quantized to 8 bits, matching the HW's subtexel fixed point
 * (subTexelPrecisionBits = 8 on the oracle GPU).
 */
float4 sampleBorder(Texture2D<float4> t, float2 uv)
{
    float2 st = uv * gSrcRes - 0.5f;
    float2 fb = floor(st);
    float2 f = floor((st - fb) * 256.f + 0.5f) * (1.f / 256.f);
    int2 i0 = int2(fb);
    int2 res = int2(gSrcRes);
    float4 acc = float4(0.f);
    [ForceUnroll]
    for (int dy = 0; dy <= 1; dy++)
    {
        [ForceUnroll]
        for (int dx = 0; dx <= 1; dx++)
        {
            int2 c = i0 + int2(dx, dy);
            float w = (dx != 0 ? f.x : 1.f - f.x) * (dy != 0 ? f.y : 1.f - f.y);
            if (all(c >= int2(0, 0)) && all(c < res))
                acc += w * t[c];
        }
    }
    return acc;
}

float4 blurFilter2x2(Texture2D<float4> src, const float2 uv, float uscale, float vscale)
{
    uscale *= 0.75f;
    vscale *= 0.75f;
    return 0.25f * (
        sampleBorder(src, uv + float2(-uscale, -vscale)) +
        sampleBorder(src, uv + float2(uscale, -vscale)) +
        sampleBorder(src, uv + float2(-uscale, vscale)) +
        sampleBorder(src, uv + float2(uscale, vscale)));
}

float4 blurFilter3x3(Texture2D<float4> src, const float2 uv, const float uscale, const float vscale)
{
    const float w0 = (10.f + 10.f) / 32.f;
    const float x1 = 1.5f + 1.f / (1.f + 5.f), w1 = (1.f + 5.f) / 32.f;
    const float x1u = x1 * uscale;
    const float x1v = x1 * vscale;
    return
        (sampleBorder(src, uv + float2(-x1u, -x1v)) +
         sampleBorder(src, uv + float2(x1u, -x1v)) +
         sampleBorder(src, uv + float2(-x1u, x1v)) +
         sampleBorder(src, uv + float2(x1u, x1v))) * (w1 * w1) +
        (sampleBorder(src, uv + float2(0.f, -x1v)) +
         sampleBorder(src, uv + float2(0.f, x1v)) +
         sampleBorder(src, uv + float2(-x1u, 0.f)) +
         sampleBorder(src, uv + float2(x1u, 0.f))) * (w1 * w0) +
        (sampleBorder(src, uv + float2(0.f, 0.f))) * (w0 * w0);
}

float4 blurFilter5x5(Texture2D<float4> src, const float2 uv, const float uscale, const float vscale)
{
    const float w0 = (126.f + 126.f) / 512.f;
    const float x1 = 1.5f + 36.f / (36.f + 84.f), w1 = (36.f + 84.f) / 512.f;
    const float x2 = 3.5f + 1.f / (1.f + 9.f), w2 = (1.f + 10.f) / 512.f;
    const float x1u = x1 * uscale, x2u = x2 * uscale;
    const float x1v = x1 * vscale, x2v = x2 * vscale;
    float4 s0 =
        (sampleBorder(src, uv + float2(-x2u, -x2v)) +
         sampleBorder(src, uv + float2(x2u, -x2v)) +
         sampleBorder(src, uv + float2(-x2u, x2v)) +
         sampleBorder(src, uv + float2(x2u, x2v))) * (w2 * w2) +
        (sampleBorder(src, uv + float2(-x1u, -x2v)) +
         sampleBorder(src, uv + float2(x1u, -x2v)) +
         sampleBorder(src, uv + float2(-x1u, x2v)) +
         sampleBorder(src, uv + float2(x1u, x2v))) * (w1 * w2) +
        (sampleBorder(src, uv + float2(0.f, -x2v)) +
         sampleBorder(src, uv + float2(0.f, x2v))) * (w0 * w2);
    float4 s1 =
        (sampleBorder(src, uv + float2(-x2u, -x1v)) +
         sampleBorder(src, uv + float2(x2u, -x1v)) +
         sampleBorder(src, uv + float2(-x2u, x1v)) +
         sampleBorder(src, uv + float2(x2u, x1v))) * (w2 * w1) +
        (sampleBorder(src, uv + float2(-x1u, -x1v)) +
         sampleBorder(src, uv + float2(x1u, -x1v)) +
         sampleBorder(src, uv + float2(-x1u, x1v)) +
         sampleBorder(src, uv + float2(x1u, x1v))) * (w1 * w1) +
        (sampleBorder(src, uv + float2(0.f, -x1v)) +
         sampleBorder(src, uv + float2(0.f, x1v))) * (w0 * w1);
    float4 s2 =
        (sampleBorder(src, uv + float2(-x2u, 0.f)) +
         sampleBorder(src, uv + float2(x2u, 0.f))) * (w2 * w0) +
        (sampleBorder(src, uv + float2(-x1u, 0.f)) +
         sampleBorder(src, uv + float2(x1u, 0.f))) * (w1 * w0) +
        (sampleBorder(src, uv + float2(0.f, 0.f))) * (w0 * w0);
    return s0 + s1 + s2;
}

[numthreads(16, 16, 1)]
void downsample(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    int2 pixelPos = dispatchThreadId.xy;
    float2 uv = (pixelPos + 0.5f) * gInvRes;
    gDstMip.Store(pixelPos, blurFilter5x5(gSrc, uv, gInvRes.x * 0.5f, gInvRes.y * 0.5f));
}

[numthreads(16, 16, 1)]
void upsample(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    int2 pixelPos = dispatchThreadId.xy;
    float2 uv = (pixelPos + 0.5f) * gInvRes;
    float4 centertap = blurFilter3x3(gBloomed, uv, gInvRes.x, gInvRes.y);
    if (gStar > 0.f)
    {
        float4 star = 0.f;
        for (int i = -20; i <= 20; ++i)
        {
            float bri = 1.f / (10 + i * i);
            float4 c = sampleBorder(gBloomed, uv + gStarDir1 * i * 2) +
                       sampleBorder(gBloomed, uv + gStarDir2 * i * 2) +
                       sampleBorder(gBloomed, uv + gStarDir3 * i * 2);
            star += c * bri;
        }
        star *= 1.f / 1.793358534742959f; // normalize by the total of the weights of star taps
        gDstMip.Store(pixelPos, lerp(lerp(gDstPrev[pixelPos], centertap, gBloomAmount), star, gStar));
    }
    else
    {
        if (gInPlace != 0)
        {
            gDstMip.Store(pixelPos, lerp(gDstPrev[pixelPos], centertap, gBloomAmount));
        }
        else
        {
            gDstMip.Store(pixelPos, lerp(gSrc[pixelPos], centertap, gBloomAmount));
        }
    }
}

[numthreads(16, 16, 1)]
void runPostFX(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;

    float2 uv_circular = ((pixelPos + 0.5f) * 2.f - gResolution.xy) * gInvRes.y; // -1 to 1 vertically; more than -1 to 1 horizontally.
    float r_squared = dot(uv_circular, uv_circular);

    float barrel = (1.f + r_squared * gBarrelDistort.y) * gBarrelDistort.x;
    uv_circular *= barrel;

    float3 col;
    float2 ndc = float2(gResolution.y * gInvRes.x * 0.5f, 0.5f);

    float ca = 1.f + gChromaticAberrationAmount;
    if (ca != 1.f)
    {
        col.x = sampleBorder(gBloomed, uv_circular * ndc / ca + 0.5f).x;
        col.y = sampleBorder(gBloomed, uv_circular * ndc + 0.5f).y;
        col.z = sampleBorder(gBloomed, uv_circular * ndc * ca + 0.5f).z;
    }
    else
    {
        col = sampleBorder(gBloomed, uv_circular * ndc + 0.5f).xyz;
    }

    float vignette = max(0.f, 1.f - r_squared * gVignetteAmount);
    col *= vignette;

    float luma = luminance(col);
    float squashed_luma = saturate(luma / (luma + 0.5f));
    float saturation =
        max(0.f, gSaturationCurve.z + squashed_luma * gSaturationCurve.y + squashed_luma * squashed_luma * gSaturationCurve.x);
    col = lerp(luma, col, saturation);

    col = pow(max(0.f, col * gColorScale + gColorOffset), gColorPower);

    float4 output = float4(col, 1.);
    if (pixelPos.x < gWipe)
    {
        output = gSrc[pixelPos];
    }
    gDst.Store(pixelPos, output);
}
