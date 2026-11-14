/**
 * FroxelFog dispatch 1 of 3 (scatter): per froxel, media (height fog + local volumes) and in-scattered light of one
 * analytic light with visibility from the shared visibility service (shadow map or BVH rays), optionally the scene's
 * other analytic lights (shadowed only with BVH rays), HG phase and an isotropic ambient term; blended with the
 * reprojected previous froxel volume when the history is valid (temporal mode).
 * Writes gFroxelDst = (sigma_s * L_in, sigma_t). Reads gFroxelSrc = the previous frame's volume (temporal mode).
 * Adapted in part from Godot Engine volumetric_fog_process.glsl (MIT, see FroxelFogCommon.slangh for the notice).
 */
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Rendering.Lights.LightHelpers;
#include "RenderPasses/RasterLighting/Visibility.slangh"
#include "RenderPasses/FroxelFog/FroxelFogCommon.slangh"

[numthreads(4, 4, 4)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id >= gGrid.xyz)) return;
    const float3 grid = float3(gGrid.xyz);
    float3 p = froxelWorld((float3(id) + 0.5f) / grid);

    // Temporal: reproject the froxel centre into the previous grid; only froxels that reproject are jittered.
    float4 hist = float4(0.f);
    float blend = 0.f;
    if ((gGrid.w & 1u) != 0u)
    {
        float3 q;
        if (prevFroxel(p, q))
        {
            hist = sampleFroxels(q);
            blend = gSlices.w;
            p = froxelWorld((float3(id) + gJitter.xyz) / grid);
        }
    }

    float3 sigmaS;
    float sigmaT;
    evalMedia(p, sigmaS, sigmaT);
    float3 L = float3(0.f);
    if (sigmaT > 0.f)
    {
        float3 Lin = gAmbient.rgb;
        const float3 v = normalize(p - gCamPos.xyz);
        const uint primary = gMisc.w - 1u; // 0xffffffff = none
        if (primary < gScene.getLightCount())
        {
            AnalyticLightSample ls;
            if (evalLightApproximate(p, gScene.getLight(primary), ls))
            {
                const float vis = visibility(p, float3(0.f), ls.dir, ls.distance);
                Lin += vis * ls.Li * gLightScale.rgb * hgPhase(dot(v, ls.dir), gGlobal.w);
            }
        }
        if (gWind.w > 0.f)
        {
            // The scene's other analytic lights (point/spot attenuation from evalLightApproximate); shadowed only with BVH
            // rays (the shadow map belongs to the primary light).
            for (uint j = 0; j < gScene.getLightCount(); j++)
            {
                if (j == primary) continue;
                AnalyticLightSample ls;
                if (evalLightApproximate(p, gScene.getLight(j), ls))
                {
#if VIS_MODE == 2
                    const float vis = visibility(p, float3(0.f), ls.dir, ls.distance);
#else
                    const float vis = 1.f;
#endif
                    Lin += vis * ls.Li * gWind.w * hgPhase(dot(v, ls.dir), gGlobal.w);
                }
            }
        }
        L = sigmaS * Lin;
    }
    float4 cur = float4(L, sigmaT);
    if (blend > 0.f) cur = cur + (hist - cur) * blend;
    gFroxelDst[froxelIndex(id)] = packFroxel(cur);
}
