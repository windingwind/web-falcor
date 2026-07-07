/**
 * WebFalcor override of Scene/RaytracingInline.slang (DESIGN.md §5).
 *
 * WebGPU has no ray tracing API: DXR 1.1 RayQuery is replaced by a stack-based
 * traversal over the software BVH built by the host (Scene/SoftwareRT/Bvh.ts,
 * bound as gScene.webfalcorBvhNodes/webfalcorBvhTris). The public surface
 * (SceneRayQuery<UseAlphaTest> implementing ISceneRayQuery) matches upstream.
 *
 * Semantics: triangle geometry via the BVH; SDF grid instances via a linear
 * loop (SDFGridIntersector — upstream code; instance counts are tiny).
 * RAY_FLAG_ACCEPT_FIRST_HIT honored via traceVisibilityRay;
 * instanceInclusionMask ignored (all scenes use 0xff so far -- divergence
 * recorded in the parity matrix).
 */
#include "Scene/SceneDefines.slangh"

__exported import Scene.Shading;
__exported import Scene.SceneRayQueryInterface;
#if SCENE_SDF_GRID_COUNT > 0
import Scene.Intersection;
#endif

/** Moller-Trumbore against a precomputed (v0, e1, e2) triangle. */
bool intersectTriangle(const float3 origin, const float3 dir, const float tMin, const float tMax,
                       const float3 v0, const float3 e1, const float3 e2,
                       out float t, out float2 barycentrics)
{
    t = 0.f;
    barycentrics = float2(0.f);
    const float3 pvec = cross(dir, e2);
    const float det = dot(e1, pvec);
    if (abs(det) < 1e-9f) return false;
    const float invDet = 1.f / det;
    const float3 tvec = origin - v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.f || u > 1.f) return false;
    const float3 qvec = cross(tvec, e1);
    const float v = dot(dir, qvec) * invDet;
    if (v < 0.f || u + v > 1.f) return false;
    t = dot(e2, qvec) * invDet;
    if (t < tMin || t > tMax) return false;
    barycentrics = float2(u, v);
    return true;
}

/** Ray/AABB slab test. Returns true if [tMin, tMax] overlaps the box. */
bool intersectAabb(const float3 origin, const float3 invDir, const float tMin, const float tMax,
                   const float3 bbMin, const float3 bbMax)
{
    const float3 t0 = (bbMin - origin) * invDir;
    const float3 t1 = (bbMax - origin) * invDir;
    const float3 tSmall = min(t0, t1);
    const float3 tBig = max(t0, t1);
    const float tEnter = max(max(tSmall.x, tSmall.y), max(tSmall.z, tMin));
    const float tExit = min(min(tBig.x, tBig.y), min(tBig.z, tMax));
    return tEnter <= tExit;
}

/** Software BVH traversal; returns closest triangle hit (or none). */
bool traverseBvh(const Ray ray, const bool anyHit, const bool useAlphaTest,
                 out uint hitInstance, out uint hitPrimitive, out float2 hitBarycentrics, out float hitT)
{
    hitInstance = 0;
    hitPrimitive = 0;
    hitBarycentrics = float2(0.f);
    hitT = ray.tMax;
    bool found = false;

    const float3 invDir = float3(1.f) / ray.dir;

    uint stack[32];
    uint stackSize = 0;
    uint node = 0;

    // Empty scene guard: node count encoded implicitly by buffer size is not
    // available; hosts always upload at least the root.
    for (;;)
    {
        const float4 n0 = gScene.webfalcorBvhNodes[node * 2];
        const float4 n1 = gScene.webfalcorBvhNodes[node * 2 + 1];
        if (intersectAabb(ray.origin, invDir, ray.tMin, hitT, n0.xyz, n1.xyz))
        {
            const uint leftFirst = asuint(n0.w);
            const uint triCount = asuint(n1.w);
            if (triCount > 0)
            {
                // Leaf: test triangles.
                for (uint i = 0; i < triCount; i++)
                {
                    const uint tri = leftFirst + i;
                    const float4 t0 = gScene.webfalcorBvhNodes[gScene.webfalcorBvhTrisOffset + tri * 3];
                    const float4 t1 = gScene.webfalcorBvhNodes[gScene.webfalcorBvhTrisOffset + tri * 3 + 1];
                    const float4 t2 = gScene.webfalcorBvhNodes[gScene.webfalcorBvhTrisOffset + tri * 3 + 2];
                    float t;
                    float2 uv;
                    if (intersectTriangle(ray.origin, ray.dir, ray.tMin, hitT, t0.xyz, t1.xyz, t2.xyz, t, uv))
                    {
                        bool accept = true;
                        if (useAlphaTest)
                        {
                            // Alpha test through the material system (mirrors upstream candidate handling).
                            TriangleHit candidate;
                            candidate.instanceID.index = asuint(t0.w);
                            candidate.primitiveIndex = asuint(t1.w);
                            candidate.barycentrics = uv;
                            const VertexData v = gScene.getVertexData(candidate);
                            const uint materialID = gScene.getMaterialID(candidate.instanceID);
                            if (gScene.materials.alphaTest(v, materialID, 0.f)) accept = false;
                        }
                        if (accept)
                        {
                            hitT = t;
                            hitInstance = asuint(t0.w);
                            hitPrimitive = asuint(t1.w);
                            hitBarycentrics = uv;
                            found = true;
                            if (anyHit) return true;
                        }
                    }
                }
            }
            else
            {
                // Inner: descend left (node + 1), push right (leftFirst).
                if (stackSize < 32)
                {
                    stack[stackSize] = leftFirst;
                    stackSize++;
                }
                node = node + 1;
                continue;
            }
        }
        if (stackSize == 0) break;
        stackSize--;
        node = stack[stackSize];
    }
    return found;
}

#if SCENE_SDF_GRID_COUNT > 0
/** Closest SDF grid instance hit (mirrors the native sdfGridIntersection shader
    over procedural AABBs; linear loop instead — see header note). */
bool intersectSdfGrids(const Ray ray, const float tMaxIn, out uint outInstance, out SDFGridHitData outHitData, out float outT)
{
    outInstance = 0;
    outHitData = {};
    outT = tMaxIn;
    bool found = false;
    for (uint i = 0; i < gScene.webfalcorSdfInstanceCount; i++)
    {
        GeometryInstanceID instanceID;
        instanceID.index = gScene.webfalcorSdfInstanceFirst + i;
        Ray clipped = ray;
        clipped.tMax = outT;
        SDFGridHitData hitData;
        float t;
        if (SDFGridIntersector::intersect(clipped, instanceID, 0, hitData, t))
        {
            outT = t;
            outInstance = instanceID.index;
            outHitData = hitData;
            found = true;
        }
    }
    return found;
}

/** Any-hit over SDF grid instances (visibility rays). */
bool intersectSdfGridsAny(const Ray ray)
{
    for (uint i = 0; i < gScene.webfalcorSdfInstanceCount; i++)
    {
        GeometryInstanceID instanceID;
        instanceID.index = gScene.webfalcorSdfInstanceFirst + i;
        if (SDFGridIntersector::intersectAny(ray, instanceID, 0)) return true;
    }
    return false;
}
#endif

/** Scene ray query (mirrors upstream SceneRayQuery<UseAlphaTest>). */
struct SceneRayQuery<let UseAlphaTest : int> : ISceneRayQuery
{
    HitInfo traceRay(const Ray ray, out float hitT, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff)
    {
        uint instance, primitive;
        float2 barycentrics;
        float t;
        HitInfo hit = {};
        hitT = 0.f;
        if (traverseBvh(ray, false, UseAlphaTest != 0, instance, primitive, barycentrics, t))
        {
            TriangleHit triangleHit;
            triangleHit.instanceID.index = instance;
            triangleHit.primitiveIndex = primitive;
            triangleHit.barycentrics = barycentrics;
            hit = HitInfo(triangleHit);
            hitT = t;
        }
#if SCENE_SDF_GRID_COUNT > 0
        {
            // SDF grids compete with the closest triangle hit.
            const float sdfTMax = hit.isValid() ? hitT : ray.tMax;
            uint sdfInstance;
            SDFGridHitData sdfHitData;
            float sdfT;
            if (intersectSdfGrids(ray, sdfTMax, sdfInstance, sdfHitData, sdfT))
            {
                SDFGridHit sdfGridHit;
                sdfGridHit.instanceID.index = sdfInstance;
                sdfGridHit.hitData = sdfHitData;
                hit = HitInfo(sdfGridHit);
                hitT = sdfT;
            }
        }
#endif
        return hit;
    }

    bool traceVisibilityRay(const Ray ray, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff)
    {
        uint instance, primitive;
        float2 barycentrics;
        float t;
        if (traverseBvh(ray, true, UseAlphaTest != 0, instance, primitive, barycentrics, t)) return false;
#if SCENE_SDF_GRID_COUNT > 0
        if (intersectSdfGridsAny(ray)) return false;
#endif
        return true;
    }
};
