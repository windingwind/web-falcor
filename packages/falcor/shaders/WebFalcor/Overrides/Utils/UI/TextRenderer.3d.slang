/**
 * WebFalcor override of Utils/UI/TextRenderer.3d.slang (docs §4.3): the upstream
 * vertex entry uses `inout`/`out` parameters, which the WGSL backend emits as
 * invalid code; the varyings ride a struct instead. Shading is unchanged.
 */
cbuffer PerFrameCB
{
    float4x4 gvpTransform;
    float3 gFontColor;
};

Texture2D gFontTex;

struct TextVSOut
{
    float2 texC : TEXCOORD;
    float4 posSV : SV_POSITION;
};

float4 transform(float2 posS)
{
    return mul(gvpTransform, float4(posS, 0.5f, 1));
}

TextVSOut vsMain(float2 posS: POSITION, float2 texC: TEXCOORD)
{
    TextVSOut vsOut;
    vsOut.texC = texC;
    vsOut.posSV = transform(posS);
    return vsOut;
}

float4 calcColor(float2 texC)
{
    float4 color = gFontTex.Load(int3(texC, 0));
    color.rgb = gFontColor;
    return color;
}

float4 psMain(TextVSOut vsOut) : SV_TARGET0
{
    return calcColor(vsOut.texC);
}
