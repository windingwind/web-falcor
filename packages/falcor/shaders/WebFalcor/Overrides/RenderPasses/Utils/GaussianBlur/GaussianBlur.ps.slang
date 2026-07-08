/**
 * WebFalcor override of RenderPasses/Utils/GaussianBlur/GaussianBlur.ps.slang
 * (docs §4.3). Diffs vs upstream:
 *  - the typed weights buffer (Buffer<float>) is a StructuredBuffer<float>
 *    (WGSL has no texel buffers);
 *  - SampleLevel's integer offset argument is folded into the sample
 *    coordinate (texC + offset/dim). WGSL requires const-expression offsets,
 *    but Slang lowers the unrolled offsets to runtime vars. Adding the offset
 *    in texel space before addressing is arithmetically identical.
 */
#ifdef _USE_TEX2D_ARRAY
Texture2DArray gSrcTex;
#else
texture2D gSrcTex;
#endif

SamplerState gSampler;

struct BlurPSIn
{
    float2 texC : TEXCOORD;
    float4 pos : SV_POSITION;
#ifdef _USE_TEX2D_ARRAY
    uint arrayIndex : SV_RenderTargetArrayIndex;
#endif
};

StructuredBuffer<float> weights;

#ifdef _USE_TEX2D_ARRAY
float4 blur(float2 texC, uint arrayIndex)
#else
float4 blur(float2 texC)
#endif
{
#ifdef _HORIZONTAL_BLUR
    const float2 dir = float2(1, 0);
#elif defined _VERTICAL_BLUR
    const float2 dir = float2(0, 1);
#else
#error Please define either _HORIZONTAL_BLUR or _VERTICAL_BLUR
#endif

    const int2 offset = -(_KERNEL_WIDTH / 2) * dir;

    float2 dim;
#ifdef _USE_TEX2D_ARRAY
    float elements;
    gSrcTex.GetDimensions(dim.x, dim.y, elements);
#else
    gSrcTex.GetDimensions(dim.x, dim.y);
#endif
    const float2 invDim = 1.f / dim;

    float4 c = float4(0, 0, 0, 0);

    [ForceUnroll]
    for (int i = 0; i < _KERNEL_WIDTH; i++)
    {
        const float2 sampleC = texC + float2(offset + i * dir) * invDim;
#ifdef _USE_TEX2D_ARRAY
        c += gSrcTex.SampleLevel(gSampler, float3(sampleC, arrayIndex), 0) * weights[i];
#else
        c += gSrcTex.SampleLevel(gSampler, sampleC, 0) * weights[i];
#endif
    }
    return c;
}

float4 main(BlurPSIn pIn) : SV_TARGET0
{
    float4 fragColor = float4(1.f, 1.f, 1.f, 1.f);

#ifdef _USE_TEX2D_ARRAY
    fragColor = blur(pIn.texC, pIn.arrayIndex);
#else
    fragColor = blur(pIn.texC);
#endif
    return fragColor;
}
