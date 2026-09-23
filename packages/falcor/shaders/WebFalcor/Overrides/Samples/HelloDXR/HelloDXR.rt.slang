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

/**
 * WebFalcor override of Samples/HelloDXR/HelloDXR.rt.slang (docs §5): the RT pipeline
 * becomes a compute kernel over SceneRayQuery. rayGen is main(); TraceRay over hit
 * group 0 is traceRay with the primaryAnyHit alpha test; the shadow ray (hit group 1,
 * whose any-hit accepts every hit) is an untested visibility ray; primaryClosestHit and
 * primaryMiss are functions, and the depth-1 reflection is a second, non-recursive trace
 * (WGSL has no recursion). gOutput is write-only; the launch size is viewportDims (no
 * DispatchRaysDimensions in compute).
 */
import Scene.RaytracingInline;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

WTexture2D<float4> gOutput;

cbuffer PerFrameCB
{
    float4x4 invView;
    float2 viewportDims;
    float tanHalfFovY;
    uint sampleIndex;
    bool useDOF;
};

struct PrimaryRayData
{
    float4 color;
    uint depth;
    float hitT;
};

static const float3 kMissColor = float3(0.38f, 0.52f, 0.10f);

/** Was shadowMiss/shadowAnyHit: any hit occludes. */
bool traceShadowRay(Ray ray)
{
    SceneRayQuery<0> query;
    return !query.traceVisibilityRay(ray, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff);
}

/** Was primaryClosestHit without its reflection ray (the caller adds that at depth 0). */
float3 shadeHit(const TriangleHit triangleHit, const float3 rayDirW, const uint2 launchIndex, out ShadingData sd, out BSDFProperties bsdfProperties, out float3 normalW)
{
    const GeometryInstanceID instanceID = triangleHit.instanceID;
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(instanceID);
    sd = gScene.materials.prepareShadingData(v, materialID, -rayDirW);
    normalW = v.normalW;

    // Create material instance and query its properties.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    bsdfProperties = mi.getProperties(sd);

    float3 color = bsdfProperties.emission;
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex, sampleIndex);

    // Direct lighting from analytic light sources.
    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        if (evalLightApproximate(sd.posW, gScene.getLight(i), ls))
        {
            Ray shadowRay = Ray(sd.posW, ls.dir, 0.001f, ls.distance);
            if (!traceShadowRay(shadowRay))
            {
                color += mi.eval(sd, ls.dir, sg) * ls.Li;
            }
        }
    }
    return color;
}

/** Was TraceRay over hit group 0 with the primary hit/miss shaders, at hit depth 1 (no reflection). */
PrimaryRayData traceReflectionRay(const Ray ray, const uint2 launchIndex)
{
    PrimaryRayData hitData;
    hitData.depth = 1;
    SceneRayQuery<1> query;
    float hitT;
    const HitInfo hit = query.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        ShadingData sd;
        BSDFProperties bsdfProperties;
        float3 normalW;
        hitData.color = float4(shadeHit(hit.getTriangleHit(), ray.dir, launchIndex, sd, bsdfProperties, normalW), 1.f);
        hitData.hitT = hitT;
    }
    else
    {
        hitData.color = float4(kMissColor, 1.f);
        hitData.hitT = -1.f;
    }
    return hitData;
}

float3 getReflectionColor(const ShadingData sd, const BSDFProperties bsdfProperties, float3 normalW, float3 worldRayDir, uint2 launchIndex)
{
    Ray ray = Ray(sd.posW, reflect(worldRayDir, normalW), 0.001f, 100000.f);
    PrimaryRayData secondaryRay = traceReflectionRay(ray, launchIndex);

    // A very non-PBR inaccurate way to do reflections.
    float3 reflectColor = secondaryRay.hitT < 0.f ? float3(0.f) : secondaryRay.color.rgb;
    float falloff = max(1.f, (secondaryRay.hitT * secondaryRay.hitT));
    reflectColor *= 20.f / falloff;
    float ggxAlpha = min(0.5f, max(1e-8f, bsdfProperties.roughness * bsdfProperties.roughness));
    reflectColor += bsdfProperties.specularReflectance * reflectColor * (ggxAlpha * ggxAlpha);
    return reflectColor;
}

/** Was [shader("raygeneration")] rayGen. */
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 launchIndex = dispatchThreadId.xy;
    if (any(launchIndex >= uint2(viewportDims)))
        return;
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex, sampleIndex);

    Ray ray;
    if (!useDOF)
    {
        ray = gScene.camera.computeRayPinhole(launchIndex, uint2(viewportDims));
    }
    else
    {
        float2 u = sampleNext2D(sg);
        ray = gScene.camera.computeRayThinlens(launchIndex, uint2(viewportDims), u);
    }

    float4 color;
    SceneRayQuery<1> query;
    float hitT;
    const HitInfo hit = query.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        ShadingData sd;
        BSDFProperties bsdfProperties;
        float3 normalW;
        float3 c = shadeHit(hit.getTriangleHit(), ray.dir, launchIndex, sd, bsdfProperties, normalW);
        // Shoot a reflection ray.
        c += getReflectionColor(sd, bsdfProperties, normalW, ray.dir, launchIndex);
        color = float4(c, 1.f);
    }
    else
    {
        color = float4(kMissColor, 1.f);
    }
    gOutput.Store(launchIndex, color);
}
