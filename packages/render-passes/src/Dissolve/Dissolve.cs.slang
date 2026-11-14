/**
 * Dissolve (see Dissolve.ts): burning edge along the GBufferRaster per-instance clip.
 * e = clipEdge (1 at the cut, 0 beyond the edge width); dst = colour * (1 - charAmount * e)
 *     + edgeIntensity * lerp(emberColour, edgeColour, e^2) * e^edgeSharpness; e = 0 copies the colour. Stateless.
 * Idea after AdultLink/SphereDissolve (MIT, Copyright (c) 2018 Guillermo Angel); code written for this pass.
 */

cbuffer CB
{
    uint2 gFrameDim;
    float gEdgeIntensity;
    float gEdgeSharpness;
    float3 gEdgeColour;
    float gCharAmount;
    float3 gEmberColour;
};

Texture2D<float4> gColor;
Texture2D<float> gClipEdge;
RWTexture2D<float4> gDst;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 px = tid.xy;
    if (any(px >= gFrameDim)) return;
    const float4 c = gColor[px];
    const float e = gClipEdge[px];
    if (!(e > 0.f) || (gEdgeIntensity == 0.f && gCharAmount == 0.f))
    {
        gDst[px] = c;
        return;
    }
    const float3 glow = lerp(gEmberColour, gEdgeColour, e * e) * pow(e, gEdgeSharpness);
    gDst[px] = float4(c.rgb * (1.f - gCharAmount * e) + gEdgeIntensity * glow, c.a);
}
