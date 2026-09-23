/**
 * WebFalcor override of RenderPasses/TestPasses/TestRtProgram.rt.slang
 * (docs §5 megakernel lowering). Diffs vs upstream:
 * - RT pipeline -> compute kernel; the shader binding table becomes explicit
 *   selection: ray type and miss index pick the colour, any-hit shaders become
 *   ITriangleHitFilter candidates in the software BVH traversal.
 * - Intersection shaders: the inscribed-sphere test runs over every custom
 *   primitive (host buffer), competing with the closest triangle hit.
 * - output is a write-only global (WGSL storage-texture rule).
 * - geometryID is the native per-mesh ID (host buffer; the web scene has one
 *   instance per mesh). createDynamicObject<IMtl> -> switch over the same types.
 */
/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Scene/SceneDefines.slangh"

import Scene.RaytracingInline; // WebFalcor: software traversal replaces TraceRay
import Utils.Geometry.IntersectionHelpers;
import Utils.Math.Ray;

#ifndef MODE
#error MODE is expected to be defined
#endif

// WebFalcor: per-instance native geometry ID, and custom primitive AABBs as
// (min.xyz, asfloat(userID)), (max.xyz, 0) pairs.
StructuredBuffer<uint> gGeometryIDs;
StructuredBuffer<float4> gCustomPrimitives;

// WebFalcor: write-only (WGSL rgba32float storage rule), so a global, not a TestProgram member.
WTexture2D<float4> gOutput;

cbuffer CB
{
    uint gCustomPrimitiveCount;
}

struct RayData
{
    float t;
    float3 color;
};

/** BLAS geometry of opaque materials is flagged RtGeometryFlags::Opaque: any-hit never runs. */
bool isOpaqueGeometry(const uint instance)
{
    GeometryInstanceID instanceID;
    instanceID.index = instance;
    return gScene.materials.getMaterialHeader(gScene.getMaterialID(instanceID)).getAlphaMode() == AlphaMode::Opaque;
}

// Miss shaders.
float3 miss0() { return float3(0.05f); }
float3 miss1() { return float3(0.1f); }

#if MODE == 0

/** Default triangle hit groups run the alpha-test any-hit; the green/red overrides have none. */
struct Mode0HitFilter : ITriangleHitFilter
{
    bool accept(const uint instance, const uint primitive, const float2 barycentrics, const float t)
    {
        const uint geometryID = gGeometryIDs[instance];
        if (geometryID == 1 || geometryID == 3 || isOpaqueGeometry(instance)) return true;
        AlphaTestHitFilter alphaTest = { true };
        return alphaTest.accept(instance, primitive, barycentrics, t);
    }
};

float3 closestHitTriangle(const uint geometryID, const uint rayType)
{
    if (geometryID == 1) return rayType == 0 ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);  // closestHitGreen / closestHitRed
    if (geometryID == 3) return rayType == 0 ? float3(1.f, 0.f, 0.f) : float3(0.f, 1.f, 0.f);  // closestHitRed / closestHitGreen
    return rayType == 0 ? float3(0.5f) : float3(0.0f, 0.0f, 1.f);                              // closestHitMtl0 / closestHitMtl1
}

float3 closestHitSphere(const uint userID, const uint rayType)
{
    // Use non-default material for custom primitives with even userID.
    if (userID % 2 == 0) return rayType == 0 ? float3(0.75f, 0.f, 1.f) : float3(1.f, 1.f, 0.f);  // Purple / Yellow
    return rayType == 0 ? float3(0.25f) : float3(0.5f, 0.f, 0.5f);                              // SphereMtl0 / SphereMtl1
}

#elif MODE == 1

interface IMtl
{
    float3 getColor();
    bool alphaTest(uint2 pixel);
};

struct Mtl0 : IMtl
{
    float3 getColor() { return float3(1, 0, 0); }

    bool alphaTest(uint2 pixel) { return pixel.x % 8 == 0; }
}

struct Mtl1 : IMtl
{
    float3 getColor() { return float3(0, 1, 0); }

    bool alphaTest(uint2 pixel) { return pixel.y % 8 == 0; }
}

struct Mtl2 : IMtl
{
    float3 getColor() { return float3(0, 0, 1); }

    bool alphaTest(uint2 pixel) { return (pixel.x + pixel.y) % 8 == 0; }
}

// WebFalcor: static stand-in for createDynamicObject<IMtl, int>(type, dummy).
bool mtlAlphaTest(const uint type, const uint2 pixel)
{
    if (type == 0) { Mtl0 m; return m.alphaTest(pixel); }
    if (type == 1) { Mtl1 m; return m.alphaTest(pixel); }
    Mtl2 m;
    return m.alphaTest(pixel);
}

float3 mtlGetColor(const uint type)
{
    if (type == 0) { Mtl0 m; return m.getColor(); }
    if (type == 1) { Mtl1 m; return m.getColor(); }
    Mtl2 m;
    return m.getColor();
}

/** anyHit: the discard pattern depends on which IMtl type the geometry selects. */
struct Mode1HitFilter : ITriangleHitFilter
{
    uint2 pixel;
    bool accept(const uint instance, const uint primitive, const float2 barycentrics, const float t)
    {
        if (isOpaqueGeometry(instance)) return true;
        const uint type = gGeometryIDs[instance] % 3;
        return !mtlAlphaTest(type, pixel);
    }
};

#endif

struct TestProgram
{
    uint2 frameDim;

    void execute(const uint2 pixel)
    {
        const Ray ray = gScene.camera.computeRayPinhole(pixel, frameDim);
        RayData rayData = {};

        // Select miss shader index in a checkerboard pattern.
        uint missIndex = (pixel.x ^ pixel.y) & 0x10 ? 1 : 0;

#if MODE == 0
        // Select ray type in a diagonal line pattern.
        uint rayType = (pixel.x + pixel.y) & 0x8 ? 1 : 0;
        Mode0HitFilter filter;
#elif MODE == 1
        uint rayType = 0;
        Mode1HitFilter filter = { pixel };
#endif
        uint instance, primitive;
        float2 barycentrics;
        float hitT;
        const bool hitTriangle = traverseBvhFiltered(ray, false, filter, instance, primitive, barycentrics, hitT);
        rayData.t = hitTriangle ? hitT : ray.tMax;
        rayData.color = missIndex == 0 ? miss0() : miss1();

#if MODE == 0
        if (hitTriangle) rayData.color = closestHitTriangle(gGeometryIDs[instance], rayType);

        // intersectSphere over each custom primitive: inscribed sphere, closest t wins.
        for (uint i = 0; i < gCustomPrimitiveCount; i++)
        {
            const float4 lo = gCustomPrimitives[2 * i];
            const float4 hi = gCustomPrimitives[2 * i + 1];
            const float3 e = hi.xyz - lo.xyz;
            const float radius = min(e.x, min(e.y, e.z)) * 0.5f;
            float t;
            bool isect = intersectRaySphere(ray.origin, ray.dir, 0.5f * (lo.xyz + hi.xyz), radius, t);
            if (isect && t >= ray.tMin && t < rayData.t)
            {
                rayData.t = t;
                rayData.color = closestHitSphere(asuint(lo.w), rayType);
            }
        }
#elif MODE == 1
        if (hitTriangle) rayData.color = mtlGetColor(gGeometryIDs[instance] % 3);
#endif

        gOutput.Store(pixel, float4(rayData.color, 1.f));
    }
};

ParameterBlock<TestProgram> gTestProgram;

// WebFalcor: compute entry point replaces rayGen.
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gTestProgram.frameDim)) return;
    gTestProgram.execute(pixel);
}
