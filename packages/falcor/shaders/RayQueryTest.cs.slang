/**
 * SoftwareRT test kernel: primary rays through SceneRayQuery (BVH traversal
 * via the RaytracingInline override), writing posW like GBufferRT.
 */
import Scene.RaytracingInline;

cbuffer CB
{
    uint2 gFrameDim;
}

RWStructuredBuffer<float4> gPosW;
RWStructuredBuffer<float4> gVisibility;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gFrameDim)) return;
    const uint idx = tid.y * gFrameDim.x + tid.x;

    // Primary ray through the pixel center (mirrors GBufferRT ray generation).
    const float2 pixelCenter = (float2(tid.xy) + 0.5f) / float2(gFrameDim);
    const float2 ndc = float2(2.f, -2.f) * pixelCenter + float2(-1.f, 1.f);
    const float3 dir = normalize(ndc.x * gScene.camera.data.cameraU + ndc.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW);

    Ray ray = Ray(gScene.camera.data.posW, dir, 0.f, 1e30f);

    SceneRayQuery<0> query;
    float hitT;
    const HitInfo hit = query.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff);

    float4 posW = float4(0.f);
    if (hit.isValid())
    {
        posW = float4(ray.origin + ray.dir * hitT, 1.f);
    }
    gPosW[idx] = posW;

    // Visibility ray from the hit point back to the camera (should be visible).
    float vis = 1.f;
    if (hit.isValid())
    {
        const float3 toCam = gScene.camera.data.posW - posW.xyz;
        const float dist = length(toCam);
        Ray visRay = Ray(posW.xyz + normalize(toCam) * 1e-4f, normalize(toCam), 0.f, dist - 1e-3f);
        vis = query.traceVisibilityRay(visRay, RAY_FLAG_NONE, 0xff) ? 1.f : 0.f;
    }
    gVisibility[idx] = float4(vis, hitT, 0.f, 0.f);
}
