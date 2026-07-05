/** FullScreenPass test pixel shader: tint * texture sample (see raster-pass.gpu.test.ts). */

cbuffer TintCB
{
    float4 gTint;
}

Texture2D gSrcTex;
SamplerState gSampler;

float4 main(float2 texC: TEXCOORD) : SV_Target
{
    return gSrcTex.SampleLevel(gSampler, texC, 0.0) * gTint;
}
