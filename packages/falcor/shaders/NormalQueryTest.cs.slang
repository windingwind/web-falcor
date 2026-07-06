/** WebFalcor GPU test kernel: shading normal (VertexData.normalW) per pixel via
 *  SceneRayQuery — compared against native GBufferRT guideNormalW. */

import Scene.RaytracingInline;
import Scene.Intersection;

cbuffer CB
{
    uint2 gFrameDim;
}

RWStructuredBuffer<float4> gNormalW;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gFrameDim)) return;
    const uint idx = tid.y * gFrameDim.x + tid.x;

    const float2 pixelCenter = (float2(tid.xy) + 0.5f) / float2(gFrameDim);
    const float2 ndc = float2(2.f, -2.f) * pixelCenter + float2(-1.f, 1.f);
    const float3 dir = normalize(ndc.x * gScene.camera.data.cameraU + ndc.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW);

    Ray ray = Ray(gScene.camera.data.posW, dir, 0.f, 1e30f);

    SceneRayQuery<0> query;
    float hitT;
    const HitInfo hit = query.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff);

    float4 result = float4(0.f);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        VertexData v = gScene.getVertexData(triangleHit);
        result = float4(v.normalW, 1.f);
    }
    gNormalW[idx] = result;
}
