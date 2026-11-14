/**
 * VisibilityQuery: per-pixel light visibility of the visible surface through the visibility service
 * (Visibility.slangh), as an AOV and as a usage example of the service.
 * Output (RGBA32Float): x = visibility (1 lit, 0 occluded), y = dot(oriented face normal, L), z = distance to the light
 * (FLT_MAX for directional lights), w = 1 on geometry; background and pixels without a valid light = (1, 0, 0, 0).
 */
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Rendering.Lights.LightHelpers;
#include "RenderPasses/RasterLighting/Visibility.slangh"

cbuffer CB
{
    uint2 gFrameDim;
    int gLightIndex;
};

Texture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> gOutput;

ShadingData loadShadingData(const HitInfo hit, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
    return gScene.materials.prepareShadingData(v, materialID, -rayDir);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim)) return;
    float4 result = float4(1.f, 0.f, 0.f, 0.f);
    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid() && gLightIndex >= 0 && uint(gLightIndex) < gScene.getLightCount())
    {
        const float3 rayDir = gScene.camera.computeRayPinhole(pixel, gFrameDim).dir;
        const ShadingData sd = loadShadingData(hit, rayDir);
        AnalyticLightSample ls;
        if (evalLightApproximate(sd.posW, gScene.getLight(uint(gLightIndex)), ls))
        {
            const float vis = visibility(sd.posW, sd.faceN, ls.dir, ls.distance);
            result = float4(vis, dot(sd.getOrientedFaceNormal(), ls.dir), ls.distance, 1.f);
        }
    }
    gOutput[pixel] = result;
}
