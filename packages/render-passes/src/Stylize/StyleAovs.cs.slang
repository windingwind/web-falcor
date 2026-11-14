/**
 * StyleAovs: per-pixel IDs and stylization helper buffers from the V-buffer (see StyleAovs.ts).
 * All outputs are RGBA32Float; background pixels (no hit) are 0 with alpha 0.
 *  gId         : r = hash01(id) of the selected id source, g = raw id (float), b = 0, a = 1
 *  gIdColor    : rgb = three independent hashes of the selected id (a random flat colour per id), a = 1
 *  gMaterialId : r = hash01(material id), g = raw material id, a = 1
 *  gFlatAlbedo : material base colour sampled at a coarse texture LOD (the texture's mean colour when mips exist)
 *  gNdotL      : r = unclamped dot(N, L) in [-1, 1] for the key light, g = 0.5 * dot + 0.5 (half-Lambert), a = 1
 *  gCustom     : (only with USE_CUSTOM_AOV) r = gInstanceScalars[instance id] (0 past the end), a = 1
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
#ifdef USE_CUSTOM_AOV
    uint gInstanceScalarCount;
#endif
};

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gTexC;
Texture2D<float4> gNormW;

RWTexture2D<float4> gId;
RWTexture2D<float4> gIdColor;
RWTexture2D<float4> gMaterialId;
RWTexture2D<float4> gFlatAlbedo;
RWTexture2D<float4> gNdotL;
#ifdef USE_CUSTOM_AOV
RWTexture2D<float4> gCustom;
StructuredBuffer<float> gInstanceScalars;
#endif

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
#ifdef USE_CUSTOM_AOV
        gCustom[p] = float4(0.f);
#endif
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
#ifdef USE_CUSTOM_AOV
    gCustom[p] = float4(instanceId < gInstanceScalarCount ? gInstanceScalars[instanceId] : 0.f, 0.f, 0.f, 1.f);
#endif

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
}
