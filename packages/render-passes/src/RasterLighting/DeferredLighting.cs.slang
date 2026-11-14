/**
 * WebFalcor raster deferred direct lighting (V-buffer -> radiance).
 *
 * Per pixel: reconstruct the scene ShadingData from the V-buffer (same material
 * stack as GBufferRaster / MinimalPathTracer), then sum the direct light of
 *  - every point / spot / directional analytic light (evalLightApproximate),
 *    shadowed by the cascaded shadow map (PCF), by ray-traced visibility, or not at all;
 *  - the env map / constant sky as ambient light: diffuse irradiance from order-2 SH
 *    (unshadowed), specular from the env map in the reflection direction, optionally
 *    occluded by one visibility ray along that direction (ENV_SPEC_OCCLUSION);
 *  - emission.
 *
 * Host defines:
 *  BRDF_MODEL          0 = scene material eval (IMaterialInstance::eval; Falcor StandardBSDF:
 *                          Frostbite diffuse + GGX, height-correlated Smith, Schlick Fresnel)
 *                      1 = classic Lambert + GGX from the material's BSDFProperties
 *  SHADOW_MODE         0 = none, 1 = shadow map (cascaded, PCF), 2 = ray traced (SceneRayQuery)
 *  USE_ANALYTIC_LIGHTS / USE_ENV_LIGHT / USE_ENV_BACKGROUND / USE_EMISSIVE  (0/1)
 *  ENV_SPEC_OCCLUSION  1 = the env specular term is dropped where the reflection ray hits scene
 *                          geometry (without it, glossy surfaces facing other geometry mirror the
 *                          env map's distant content, which reads as see-through windows/walls)
 *  is_valid_gViewW, is_valid_gVisibility, is_valid_gDiffuse, is_valid_gSpecular  (optional I/O)
 */
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
#if SHADOW_MODE == 2 || ENV_SPEC_OCCLUSION
import Scene.RaytracingInline;
#endif
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;

#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxCascades = 4;

cbuffer CB
{
    uint2 gFrameDim;
    int gShadowLightIndex;    ///< Analytic light that the shadow map belongs to (-1: none).
    uint gCascadeCount;
    float4x4 gCascadeViewProj[kMaxCascades];
    float4 gCascadeSplits;    ///< Far view depth of each cascade.
    float4 gCascadeTexelWorld;///< World-space size of one shadow texel per cascade.
    float4 gCascadeDepthScale;///< 1 / light-space depth range (world units) per cascade.
    uint gCascadeSize;        ///< Square cascade resolution (atlas = count*size x size).
    float gDepthBias;         ///< Constant depth bias in world units.
    float gNormalBias;        ///< Normal offset in shadow texels.
    int gPcfRadius;           ///< PCF kernel radius in texels (bilinear taps, (2r+1)^2 of them).
    float3 gCamForward;       ///< Camera forward (normalized) for cascade selection.
    float gEnvSpecLodScale;   ///< Env map lod = roughness * scale.
};

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;          // optional
Texture2D<float> gShadowMap;       // SHADOW_MODE == 1
StructuredBuffer<float4> gEnvSH;   // USE_ENV_LIGHT

WTexture2D<float4> gOutputColor;
WTexture2D<float> gVisibility;     // optional: visibility of the shadowed light
WTexture2D<float4> gDiffuse;       // optional: direct diffuse radiance (analytic + env)
WTexture2D<float4> gSpecular;      // optional: direct specular radiance (analytic + env)

// ---------------------------------------------------------------------------
// Shadows

float shadowCompare(uint cascade, int2 t, float z)
{
    t = clamp(t, int2(0), int2(int(gCascadeSize) - 1));
    const float d = gShadowMap.Load(int3(int(cascade * gCascadeSize) + t.x, t.y, 0));
    return z <= d ? 1.f : 0.f;
}

/// Bilinear PCF over (2r+1)^2 taps (each tap = 2x2 comparisons with bilinear weights).
float sampleShadowMap(float3 posW, float3 faceN, float3 L)
{
    const float viewZ = dot(posW - gScene.camera.getPosition(), gCamForward);
    uint c = 0;
    while (c + 1 < gCascadeCount && viewZ > gCascadeSplits[c]) c++;

    // Normal offset (in world units, scaled by the cascade's texel size), oriented to the light side.
    const float3 n = dot(faceN, L) >= 0.f ? faceN : -faceN;
    const float3 p = posW + n * (gNormalBias * gCascadeTexelWorld[c]);
    const float4 q = mul(gCascadeViewProj[c], float4(p, 1.f));
    const float2 uv = float2(q.x * 0.5f + 0.5f, 0.5f - q.y * 0.5f);
    if (any(uv < 0.f) || any(uv > 1.f) || q.z > 1.f) return 1.f;
    const float z = q.z - gDepthBias * gCascadeDepthScale[c];

    const float2 tc = uv * float(gCascadeSize) - 0.5f;
    const int2 t0 = int2(floor(tc));
    const float2 f = tc - float2(t0);
    float sum = 0.f;
    for (int dy = -gPcfRadius; dy <= gPcfRadius; dy++)
    {
        for (int dx = -gPcfRadius; dx <= gPcfRadius; dx++)
        {
            const int2 t = t0 + int2(dx, dy);
            const float s00 = shadowCompare(c, t, z);
            const float s10 = shadowCompare(c, t + int2(1, 0), z);
            const float s01 = shadowCompare(c, t + int2(0, 1), z);
            const float s11 = shadowCompare(c, t + int2(1, 1), z);
            sum += lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
        }
    }
    const float k = float(2 * gPcfRadius + 1);
    return sum / (k * k);
}

float evalVisibility(const ShadingData sd, const AnalyticLightSample ls, uint lightIndex)
{
#if SHADOW_MODE == 1
    if (int(lightIndex) == gShadowLightIndex) return sampleShadowMap(sd.posW, sd.faceN, ls.dir);
    return 1.f;
#elif SHADOW_MODE == 2
    // Same ray setup as MinimalPathTracer's shadow rays.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);
    SceneRayQuery<1> query;
    const Ray ray = Ray(origin, ls.dir, 0.f, ls.distance);
    return query.traceVisibilityRay(ray, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff) ? 1.f : 0.f;
#else
    return 1.f;
#endif
}

// ---------------------------------------------------------------------------
// BRDFs

/// Lambert + GGX (height-correlated Smith, Schlick Fresnel), times cos(theta_l). Split into diffuse / specular.
void evalLambertGGX(const BSDFProperties p, const float3 N, const float3 V, const float3 L, out float3 diff, out float3 spec)
{
    diff = float3(0.f);
    spec = float3(0.f);
    const float NoL = dot(N, L);
    const float NoV = dot(N, V);
    if (NoL <= kMinCosTheta || NoV <= kMinCosTheta) return;
    diff = p.diffuseReflectionAlbedo * (M_1_PI * NoL);
    if (p.roughness <= 0.f) return; // delta specular: no contribution from a point light (as eval())
    const float3 H = normalize(V + L);
    const float NoH = saturate(dot(N, H));
    const float VoH = saturate(dot(V, H));
    const float a = max(p.roughness * p.roughness, 0.0064f);
    const float a2 = a * a;
    const float dd = NoH * NoH * (a2 - 1.f) + 1.f;
    const float D = a2 / (M_PI * dd * dd);
    const float vis = 0.5f / (NoL * sqrt(NoV * NoV * (1.f - a2) + a2) + NoV * sqrt(NoL * NoL * (1.f - a2) + a2));
    const float3 F = p.specularReflectance + (1.f - p.specularReflectance) * pow(1.f - VoH, 5.f);
    spec = F * (D * vis * NoL);
}

/// Karis' analytic fit of the split-sum environment BRDF (mobile).
float3 envBRDFApprox(float3 F0, float roughness, float NoV)
{
    const float4 c0 = float4(-1.f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = float4(1.f, 0.0425f, 1.04f, -0.04f);
    const float4 r = roughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28f * NoV)) * r.x + r.y;
    const float2 AB = float2(-1.04f, 1.04f) * a004 + r.zw;
    return F0 * AB.x + AB.y;
}

float3 shIrradiance(float3 n)
{
    // E(n) = sum_l A_l sum_m L_lm Y_lm(n); A0 = pi, A1 = 2pi/3, A2 = pi/4 (Ramamoorthi & Hanrahan 2001).
    const float A0 = M_PI, A1 = 2.f * M_PI / 3.f, A2 = M_PI / 4.f;
    float3 E = A0 * 0.282095f * gEnvSH[0].rgb;
    E += A1 * 0.488603f * (n.y * gEnvSH[1].rgb + n.z * gEnvSH[2].rgb + n.x * gEnvSH[3].rgb);
    E += A2 * (1.092548f * (n.x * n.y * gEnvSH[4].rgb + n.y * n.z * gEnvSH[5].rgb + n.x * n.z * gEnvSH[7].rgb)
               + 0.315392f * (3.f * n.z * n.z - 1.f) * gEnvSH[6].rgb + 0.546274f * (n.x * n.x - n.y * n.y) * gEnvSH[8].rgb);
    return max(E, float3(0.f));
}

// ---------------------------------------------------------------------------

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

    const float3 rayDir = is_valid(gViewW) ? -gViewW[pixel].xyz : gScene.camera.computeRayPinhole(pixel, gFrameDim).dir;
    const HitInfo hit = HitInfo(gVBuffer[pixel]);

    float3 diffuse = float3(0.f);
    float3 specular = float3(0.f);
    float3 other = float3(0.f);
    float visOut = 1.f;

    if (hit.isValid())
    {
        const ShadingData sd = loadShadingData(hit, rayDir);
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        const BSDFProperties props = mi.getProperties(sd);
        SampleGenerator sg = SampleGenerator(pixel, 0);

#if USE_EMISSIVE
        other += props.emission;
#endif

#if USE_ANALYTIC_LIGHTS
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasReflection = (lobeTypes & uint(LobeType::Reflection)) != 0;
        const bool hasTransmission = (lobeTypes & uint(LobeType::Transmission)) != 0;
        const float3 orientedFaceN = sd.getOrientedFaceNormal();
        const uint lightCount = gScene.getLightCount();
        for (uint i = 0; i < lightCount; i++)
        {
            AnalyticLightSample ls;
            if (!evalLightApproximate(sd.posW, gScene.getLight(i), ls)) continue;
            const float NdotL = dot(orientedFaceN, ls.dir);
            if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection)) continue;
            const float vis = evalVisibility(sd, ls, i);
            if (int(i) == gShadowLightIndex) visOut = vis;
            if (vis <= 0.f) continue;
#if BRDF_MODEL == 0
            // Scene material: total BSDF (diffuse/specular split not exposed by eval(); reported as diffuse).
            diffuse += mi.eval(sd, ls.dir, sg) * ls.Li * vis;
#else
            float3 fd, fs;
            evalLambertGGX(props, sd.frame.N, sd.V, ls.dir, fd, fs);
            diffuse += fd * ls.Li * vis;
            specular += fs * ls.Li * vis;
#endif
        }
#endif

#if USE_ENV_LIGHT
        // Unshadowed sky: diffuse irradiance (SH) and specular from the reflected direction.
        const float3 N = sd.frame.N;
        const float NoV = saturate(dot(N, sd.V));
        diffuse += props.diffuseReflectionAlbedo * shIrradiance(N) * M_1_PI;
        const float3 R = reflect(-sd.V, N);
        float3 Lspec = gScene.envMap.eval(R, props.roughness * gEnvSpecLodScale);
#if ENV_SPEC_OCCLUSION
        // Mirror-direction specular occlusion: the env map only reaches the surface if the reflection ray escapes.
        {
            const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, R) >= 0.f ? sd.faceN : -sd.faceN);
            SceneRayQuery<1> query;
            const Ray ray = Ray(origin, R, 0.f, FLT_MAX);
            if (!query.traceVisibilityRay(ray, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff)) Lspec = float3(0.f);
        }
#endif
#if BRDF_MODEL == 0
        specular += Lspec * props.specularReflectionAlbedo;
#else
        specular += Lspec * envBRDFApprox(props.specularReflectance, props.roughness, NoV);
#endif
#endif
    }
    else
    {
#if USE_ENV_BACKGROUND
        other = gScene.envMap.eval(rayDir);
#endif
    }

    gOutputColor.Store(pixel, float4(diffuse + specular + other, 1.f));
    if (is_valid(gVisibility)) gVisibility.Store(pixel, visOut);
    if (is_valid(gDiffuse)) gDiffuse.Store(pixel, float4(diffuse, 1.f));
    if (is_valid(gSpecular)) gSpecular.Store(pixel, float4(specular, 1.f));
}
