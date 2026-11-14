/**
 * SubsetRaster mask (see SubsetRaster.ts): classifies each pixel of the subset V-buffer against the full scene's.
 *  0 = no subset hit, 1 = the scene shows the same triangle (subset visible), 2 = the scene shows something else (hidden).
 */
#include "Scene/SceneDefines.slangh"

import Scene.HitInfo;

cbuffer CB
{
    uint2 gFrameDim;
};

Texture2D<PackedHitInfo> gSubsetVBuffer;
Texture2D<PackedHitInfo> gSceneVBuffer;
RWTexture2D<float> gSubsetMask;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const HitInfo sub = HitInfo(gSubsetVBuffer[p]);
    float m = 0.f;
    if (sub.isValid())
    {
        const HitInfo full = HitInfo(gSceneVBuffer[p]);
        bool same = false;
        if (full.isValid() && sub.getType() == HitType::Triangle && full.getType() == HitType::Triangle)
        {
            const TriangleHit a = sub.getTriangleHit();
            const TriangleHit b = full.getTriangleHit();
            same = a.instanceID.index == b.instanceID.index && a.primitiveIndex == b.primitiveIndex;
        }
        m = same ? 1.f : 2.f;
    }
    gSubsetMask[p] = m;
}
