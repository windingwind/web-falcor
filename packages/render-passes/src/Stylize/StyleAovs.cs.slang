/**
 * StyleAovs: per-pixel IDs and stylization helper buffers from the V-buffer (see StyleAovs.ts).
 * All outputs are RGBA32Float; background pixels (no hit) are 0 with alpha 0.
 *  gId         : r = hash01(id) of the selected id source, g = raw id (float), b = 0, a = 1
 *  gIdColor    : rgb = three independent hashes of the selected id (a random flat colour per id), a = 1
 *  gMaterialId : r = hash01(material id), g = raw material id, a = 1
 *  gFlatAlbedo : material base colour sampled at a coarse texture LOD (the texture's mean colour when mips exist)
 *  gNdotL      : r = unclamped dot(N, L) in [-1, 1] for the key light, g = 0.5 * dot + 0.5 (half-Lambert), a = 1
 *  gIds        : r = raw geometry instance id, g = raw material id, b = raw mesh (geometry) id, a = 1 (exact below 2^24)
 *  gPosObj     : rgb = object-space position (barycentric interpolation of the triangle's object-space vertices), a = 1
 *  gCurvature  : r = signed raw curvature in 1/world units (> 0 convex, < 0 concave), a = 1; needs posW and normW.
 *                Mean over the valid 4-neighbours j of dot(n_j - n, x_j - x) / |x_j - x|^2 (the normal curvature along
 *                each screen axis); a neighbour is valid when it has geometry and its view depth differs by at most
 *                gCurvDepthThreshold x the centre's view depth. No neighbour valid -> 0.
 */
#include "Scene/SceneDefines.slangh"

import Scene.Scene;
import Scene.HitInfo;
import Scene.Material.TextureSampler;

cbuffer CB
{
    uint2 gFrameDim;
    uint gIdSource;     ///< 0 = geometry instance, 1 = material, 2 = mesh (geometry)
    uint gSeed;
    float3 gLightDir;   ///< Unit vector towards the key light.
    float gFlatLod;
    uint gHasTexC;
    uint gHasNormW;
    uint gHasPosW;
    float gCurvDepthThreshold;
    float3 gCamPos;
    float3 gCamFwd;
};

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gTexC;
Texture2D<float4> gNormW;
Texture2D<float4> gPosW;

RWTexture2D<float4> gId;
RWTexture2D<float4> gIdColor;
RWTexture2D<float4> gMaterialId;
RWTexture2D<float4> gFlatAlbedo;
RWTexture2D<float4> gNdotL;
RWTexture2D<float4> gIds;
RWTexture2D<float4> gPosObj;
RWTexture2D<float4> gCurvature;

/** PCG-style integer hash (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering"). */
uint pcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float hash01(uint v, uint salt)
{
    return float(pcgHash(v ^ pcgHash(gSeed + salt)) >> 8) * (1.f / 16777216.f);
}

/** Signed raw curvature at p from the posW/normW neighbours (see the header). */
float rawCurvature(uint2 p)
{
    const float3 x = gPosW[p].xyz;
    const float3 n = normalize(gNormW[p].xyz);
    const float d = dot(x - gCamPos, gCamFwd);
    const int2 offs[4] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1) };
    float sum = 0.f;
    uint cnt = 0;
    for (uint k = 0; k < 4; k++)
    {
        const int2 q = int2(p) + offs[k];
        if (any(q < 0) || any(q >= int2(gFrameDim))) continue;
        const float4 xq = gPosW[q];
        if (xq.w == 0.f) continue;
        if (abs(dot(xq.xyz - gCamPos, gCamFwd) - d) > gCurvDepthThreshold * abs(d)) continue;
        const float3 dx = xq.xyz - x;
        const float l2 = dot(dx, dx);
        if (l2 <= 0.f) continue;
        sum += dot(normalize(gNormW[q].xyz) - n, dx) / l2;
        cnt++;
    }
    return cnt > 0 ? sum / float(cnt) : 0.f;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;

    const HitInfo hit = HitInfo(gVBuffer[p]);
    if (!hit.isValid() || hit.getType() != HitType::Triangle)
    {
        gId[p] = float4(0.f);
        gIdColor[p] = float4(0.f);
        gMaterialId[p] = float4(0.f);
        gFlatAlbedo[p] = float4(0.f);
        gNdotL[p] = float4(0.f);
        gIds[p] = float4(0.f);
        gPosObj[p] = float4(0.f);
        gCurvature[p] = float4(0.f);
        return;
    }

    const TriangleHit th = hit.getTriangleHit();
    const GeometryInstanceData inst = gScene.getGeometryInstance(th.instanceID);
    const uint instanceId = th.instanceID.index;
    const uint materialId = inst.materialID;
    const uint meshId = inst.geometryID;
    const uint id = gIdSource == 1 ? materialId : (gIdSource == 2 ? meshId : instanceId);

    gId[p] = float4(hash01(id, 0u), float(id), 0.f, 1.f);
    gIdColor[p] = float4(hash01(id, 1u), hash01(id, 2u), hash01(id, 3u), 1.f);
    gMaterialId[p] = float4(hash01(materialId, 4u), float(materialId), 0.f, 1.f);

    float4 albedo = float4(0.f, 0.f, 0.f, 1.f);
    if (gScene.materials.isBasicMaterial(materialId))
    {
        const BasicMaterialData md = gScene.materials.getBasicMaterialData(materialId);
        const SamplerState s = gScene.materials.getDefaultTextureSampler(materialId);
        const float2 uv = gHasTexC != 0 ? gTexC[p].xy : float2(0.5f);
        ExplicitLodTextureSampler lod = { gFlatLod };
        albedo = gScene.materials.sampleTexture(md.texBaseColor, s, uv, float4(md.baseColor), lod);
        albedo.a = 1.f;
    }
    gFlatAlbedo[p] = albedo;

    if (gHasNormW != 0)
    {
        const float3 n = normalize(gNormW[p].xyz);
        const float d = dot(n, gLightDir);
        gNdotL[p] = float4(d, 0.5f * d + 0.5f, 0.f, 1.f);
    }
    else
    {
        gNdotL[p] = float4(0.f, 0.f, 0.f, 1.f);
    }

    gIds[p] = float4(float(instanceId), float(materialId), float(meshId), 1.f);

    const uint3 vi = gScene.getIndices(th.instanceID, th.primitiveIndex);
    const float3 b = th.getBarycentricWeights();
    // Accumulated in a loop on purpose: a single expression gets folded into the textureStore below, and the WGSL binding
    // reflection then sees the scene's integer textures (passed to Scene_getVertex) inside a texture call and marks the
    // scene samplers non-filtering.
    float3 posObj = float3(0.f);
    for (uint k = 0; k < 3; k++) posObj += gScene.getVertex(vi[k]).position * b[k];
    gPosObj[p] = float4(posObj, 1.f);

    gCurvature[p] = float4(gHasPosW != 0 && gHasNormW != 0 ? rawCurvature(p) : 0.f, 0.f, 0.f, 1.f);
}
