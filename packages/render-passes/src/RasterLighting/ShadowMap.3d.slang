/**
 * WebFalcor raster shadow map (directional light, cascaded).
 * Draws the scene once per cascade into one R32Float atlas (cascades side by side,
 * selected by the viewport). Stores the orthographic light-space depth in [0,1];
 * the lighting pass does the depth comparison and PCF filtering (no comparison
 * sampler / depth-texture binding needed, which keeps the WGSL bindings simple).
 */
#include "Scene/VertexAttrib.slangh"

import Scene.Raster;

cbuffer PerCascadeCB
{
    float4x4 gLightViewProj; ///< World -> light clip space of the current cascade (row-major host matrix).
};

struct ShadowVSOut
{
    float2 texC : TEXCRD;
    nointerpolation uint materialID : MATERIAL_ID;
    float4 posH : SV_POSITION;
};

ShadowVSOut vsMain(VSIn vIn)
{
    ShadowVSOut vOut;
    GeometryInstanceID instanceID;
    instanceID.index = vIn.instanceID;
    const float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    const float3 posW = mul(worldMat, float4(vIn.pos, 1.f)).xyz;
    vOut.posH = mul(gLightViewProj, float4(posW, 1.f));
    vOut.texC = vIn.texC;
    vOut.materialID = gScene.getMaterialID(instanceID);
    return vOut;
}

float psMain(ShadowVSOut vsOut) : SV_Target0
{
#if USE_ALPHA_TEST
    VertexData v = {};
    v.texC = vsOut.texC;
    let lod = ExplicitLodTextureSampler(0.f);
    if (gScene.materials.alphaTest(v, vsOut.materialID, lod))
        discard;
#endif
    return vsOut.posH.z; // orthographic: w == 1
}
