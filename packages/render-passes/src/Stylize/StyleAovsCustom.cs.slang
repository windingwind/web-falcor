/**
 * StyleAovs `custom` output (see StyleAovs.ts): a per-instance scalar AOV from the V-buffer, in its own small kernel so
 * the main StyleAovs kernel keeps its storage-texture count (WebGPU allows 8 storage textures per shader stage).
 *  gCustom : r = gInstanceScalars[geometry instance id] (0 past the end of the list), a = 1; background pixels are 0.
 */
#include "Scene/SceneDefines.slangh"

import Scene.HitInfo;

cbuffer CB
{
    uint2 gFrameDim;
    uint gInstanceScalarCount;
};

Texture2D<PackedHitInfo> gVBuffer;
StructuredBuffer<float> gInstanceScalars;
RWTexture2D<float4> gCustom;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;

    const HitInfo hit = HitInfo(gVBuffer[p]);
    if (!hit.isValid() || hit.getType() != HitType::Triangle)
    {
        gCustom[p] = float4(0.f);
        return;
    }
    const uint instanceId = hit.getTriangleHit().instanceID.index;
    gCustom[p] = float4(instanceId < gInstanceScalarCount ? gInstanceScalars[instanceId] : 0.f, 0.f, 0.f, 1.f);
}
