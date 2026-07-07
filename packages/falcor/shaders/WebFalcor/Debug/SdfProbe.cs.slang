/**
 * Debug probe (dev-only): runs the REAL NDSDFGrid intersectSDF for supplied
 * local-space rays and dumps hit/t plus hwSample/swSample values at supplied
 * probe points. Used to adjudicate GPU-vs-CPU-port divergence (iter 78).
 */
import Scene.Scene;
import Scene.SDFs.SDFGrid;
import Scene.Intersection;

// Per ray: [0] = origin.xyz + tMin, [1] = dir.xyz + tMax, [2] = probeP.xyz + lod.
StructuredBuffer<float4> input;
// Per ray: [0] = (hit, tWorld, hitT_local, hitLod), [1] = (hwSample, swSample, lodCount, coarsestNorm).
RWStructuredBuffer<float4> result;

[numthreads(1, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint i = tid.x;
    GeometryInstanceID instanceID;
    instanceID.index = gScene.webfalcorSdfInstanceFirst;
    SDFGrid grid;
    gScene.getSDFGrid(instanceID, grid);

    const float4 o4 = input[i * 3];
    const float4 d4 = input[i * 3 + 1];
    const float4 p4 = input[i * 3 + 2];

    float t;
    SDFGridHitData hitData;
    bool hit = grid.intersectSDF(o4.xyz, d4.xyz, o4.w, d4.w, 0, t, hitData);
    result[i * 2] = float4(hit ? 1.f : 0.f, t, hitData.hitT, float(hitData.lod));

    const uint lod = uint(p4.w);
    result[i * 2 + 1] = float4(grid.hwSample(p4.xyz, lod), grid.swSample(p4.xyz, lod), float(grid.lodCount), grid.coarsestLODNormalizationFactor);

    // Full render path: SDFGridIntersector (instance matrices) + scene camera ray.
    const Ray worldRay = Ray(o4.xyz, d4.xyz, o4.w, d4.w);
    float t2;
    SDFGridHitData hd2;
    bool hit2 = SDFGridIntersector::intersect(worldRay, instanceID, 0, hd2, t2);

    const uint2 pixel = uint2(asuint(p4.x) >> 16, asuint(p4.x) & 0xffff);
    const Ray camRay = gScene.camera.computeRayPinhole(uint2(uint(p4.y), uint(p4.z)), uint2(640, 360));
    float t3;
    SDFGridHitData hd3;
    bool hit3 = SDFGridIntersector::intersect(camRay, instanceID, 0, hd3, t3);
    result[i * 2 + 1] = float4(hit2 ? t2 : -1.f, hit3 ? t3 : -1.f, dot(camRay.dir, d4.xyz), camRay.tMin);
}
