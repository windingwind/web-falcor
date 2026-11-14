/**
 * DepthComposite: merges a colour+depth layer into the scene by a per-pixel depth test (see DepthComposite.ts).
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gAlphaMode;        ///< 0 = premultiplied, 1 = straight, 2 = additive
    float gOpacity;
    float gSoftness;        ///< 0 = hard depth test
    uint gSceneDepthFromPosW;
    uint gHasSceneDepth;
    uint gHasLayerDepth;
    uint gHasTransmittance;
    float3 gCamPos;
    float3 gCamFwd;
};

Texture2D<float4> gSrc;
Texture2D<float4> gSceneDepth;
Texture2D<float4> gScenePosW;
Texture2D<float4> gLayer;
Texture2D<float4> gLayerDepth;
Texture2D<float4> gTransmittance;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gDstDepth;

static const float kFar = 3.402823466e+38f;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];

    float sceneDepth = kFar;
    if (gSceneDepthFromPosW != 0)
    {
        const float4 x = gScenePosW[p];
        if (x.w != 0.f) sceneDepth = dot(x.xyz - gCamPos, gCamFwd);
    }
    else if (gHasSceneDepth != 0)
    {
        sceneDepth = gSceneDepth[p].r;
    }
    const float layerDepth = gHasLayerDepth != 0 ? gLayerDepth[p].r : -kFar;

    // Visibility of the layer sample: hard test, or a linear fade over `softness` depth units in front of the scene.
    float vis = layerDepth < sceneDepth ? 1.f : 0.f;
    if (gSoftness > 0.f && gHasLayerDepth != 0 && sceneDepth < kFar) vis = saturate((sceneDepth - layerDepth) / gSoftness);
    const float k = vis * gOpacity;

    float4 o = src;
    float outDepth = sceneDepth;
    const float4 l = gLayer[p];
    // An empty layer sample (rgba = 0, no transmittance) leaves src untouched bit for bit (also -0 and NaN payloads).
    if (k > 0.f && (any(l != 0.f) || gHasTransmittance != 0))
    {
        float3 add;
        float3 keep;
        if (gAlphaMode == 2) { add = l.rgb; keep = float3(1.f); }
        else if (gAlphaMode == 1) { add = l.rgb * l.a; keep = float3(1.f - l.a); }
        else { add = l.rgb; keep = float3(1.f - l.a); }
        if (gHasTransmittance != 0) keep = gTransmittance[p].rgb;
        // k scales the layer's contribution: keep -> lerp(1, keep, k), add -> k * add.
        o.rgb = src.rgb * lerp(float3(1.f), keep, k) + add * k;
        if (gHasLayerDepth != 0 && l.a > 0.f) outDepth = min(sceneDepth, layerDepth);
    }
    gDst[p] = o;
    gDstDepth[p] = float4(outDepth, 0.f, 0.f, 1.f);
}
