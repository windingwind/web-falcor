/**
 * StyleShade: deferred re-shading with a per-material style table (see StyleShade.ts).
 *
 * Per pixel the V-buffer hit is turned into ShadingData (same material stack as DeferredLighting), then two results are
 * computed in one light loop:
 *  - ref    : DeferredLighting's LambertGGX model (Lambert diffuse + GGX specular per analytic light, SH9 sky diffuse,
 *             env-map specular, emission), with the visibility input applied to the shadowed light;
 *  - styled : the same terms with the material's style record applied (diffuse model, hemisphere ambient, flat-albedo
 *             mix, specular shape, specular scale).
 * Output = styleMix == 0 ? ref : lerp(ref, styled, styleMix). Background pixels and emission are never styled.
 *
 * Style record (kRecordSize float4 per record, record 0 = default for every material without its own record):
 *  r0 = (diffuseModel, wrap, minnaertK, specModel)    diffuseModel 0 lambert | 1 wrap | 2 halfLambert | 3 minnaert
 *                                                      specModel    0 ggx | 1 blinn | 2 hardSpec
 *  r1 = (skyColor.rgb, hemiMix)                        hemisphere ambient: lerp(ground, sky, 0.5 N.y + 0.5) * albedo
 *  r2 = (groundColor.rgb, flatMix)                     diffuse = lerp(shaded diffuse, albedo, flatMix)
 *  r3 = (shininess, hardThreshold, hardSoftness, specScale)
 *  r4 = (materialId or -1, 0, 0, 0)
 * The highlight shaping (hardSpec) acts on the specular lobe only: one smooth threshold, never a lit/shadow band.
 *
 * Formulas (textbook; no code copied): wrap lighting max((N.L + w) / (1 + w), 0) (GPU Gems 1, ch. 16); half-Lambert
 * (0.5 N.L + 0.5)^2 (Valve, Mitchell et al. 2006); Minnaert (N.L)^k (N.V)^(k-1) (Minnaert 1941); normalized
 * Blinn-Phong (n + 8) / (8 pi) (N.H)^n.
 *
 * Host defines: USE_ANALYTIC_LIGHTS / USE_ENV_LIGHT / USE_ENV_BACKGROUND / USE_EMISSIVE (0/1),
 *               is_valid_gViewW, is_valid_gVisibility, is_valid_gMaterialId, is_valid_gDiffuse, is_valid_gSpecular.
 */
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;

#define is_valid(name) (is_valid_##name != 0)
static const uint kRecordSize = 5;

cbuffer CB
{
    uint2 gFrameDim;
    int gShadowLightIndex;    ///< Analytic light the visibility input belongs to (-1: none).
    uint gRecordCount;        ///< Number of records (>= 1; record 0 = default).
    float gStyleMix;          ///< 0 = DeferredLighting LambertGGX result, 1 = fully styled.
    float gEnvSpecLodScale;   ///< Env map lod = roughness * scale.
};

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;          // optional
Texture2D<float4> gVisibility;     // optional: visibility of the shadowed light in r
Texture2D<float4> gMaterialId;     // optional: StyleAovs.materialId (raw id in g, alpha 0 = background)
StructuredBuffer<float4> gEnvSH;   // USE_ENV_LIGHT
StructuredBuffer<float4> gStyles;  // gRecordCount * kRecordSize

RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gDiffuse;      // optional: styled diffuse radiance (analytic + ambient)
RWTexture2D<float4> gSpecular;     // optional: styled specular radiance (analytic + env)

// ---------------------------------------------------------------------------
// Reference model (identical to DeferredLighting's BRDF_MODEL 1)

void evalLambertGGX(const BSDFProperties p, const float3 N, const float3 V, const float3 L, out float3 diff, out float3 spec)
{
    diff = float3(0.f);
    spec = float3(0.f);
    const float NoL = dot(N, L);
    const float NoV = dot(N, V);
    if (NoL <= kMinCosTheta || NoV <= kMinCosTheta) return;
    diff = p.diffuseReflectionAlbedo * (M_1_PI * NoL);
    if (p.roughness <= 0.f) return;
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
    const float A0 = M_PI, A1 = 2.f * M_PI / 3.f, A2 = M_PI / 4.f;
    float3 E = A0 * 0.282095f * gEnvSH[0].rgb;
    E += A1 * 0.488603f * (n.y * gEnvSH[1].rgb + n.z * gEnvSH[2].rgb + n.x * gEnvSH[3].rgb);
    E += A2 * (1.092548f * (n.x * n.y * gEnvSH[4].rgb + n.y * n.z * gEnvSH[5].rgb + n.x * n.z * gEnvSH[7].rgb)
               + 0.315392f * (3.f * n.z * n.z - 1.f) * gEnvSH[6].rgb + 0.546274f * (n.x * n.x - n.y * n.y) * gEnvSH[8].rgb);
    return max(E, float3(0.f));
}

// ---------------------------------------------------------------------------
// Styled model

struct StyleRecord
{
    uint diffuseModel;
    float wrap;
    float minnaertK;
    uint specModel;
    float3 sky;
    float hemiMix;
    float3 ground;
    float flatMix;
    float shininess;
    float hardThreshold;
    float hardSoftness;
    float specScale;
};

StyleRecord loadRecord(uint r)
{
    const float4 r0 = gStyles[r * kRecordSize + 0];
    const float4 r1 = gStyles[r * kRecordSize + 1];
    const float4 r2 = gStyles[r * kRecordSize + 2];
    const float4 r3 = gStyles[r * kRecordSize + 3];
    StyleRecord s;
    s.diffuseModel = uint(r0.x);
    s.wrap = r0.y;
    s.minnaertK = r0.z;
    s.specModel = uint(r0.w);
    s.sky = r1.rgb;
    s.hemiMix = r1.w;
    s.ground = r2.rgb;
    s.flatMix = r2.w;
    s.shininess = r3.x;
    s.hardThreshold = r3.y;
    s.hardSoftness = r3.z;
    s.specScale = r3.w;
    return s;
}

/// Record of a material: the first record whose id matches, else record 0.
uint findRecord(uint materialId)
{
    for (uint r = 1; r < gRecordCount; r++)
        if (gStyles[r * kRecordSize + 4].x == float(materialId)) return r;
    return 0;
}

/// Diffuse light factor (multiplies albedo / pi * Li). NoL is the cosine with the shading normal.
float styledDiffuseFactor(const StyleRecord s, float NoL, float NoV)
{
    if (s.diffuseModel == 1) return max((NoL + s.wrap) / (1.f + s.wrap), 0.f);
    if (s.diffuseModel == 2) { const float h = 0.5f * NoL + 0.5f; return h * h; }
    if (s.diffuseModel == 3)
    {
        if (NoL <= 0.f) return 0.f;
        return pow(NoL, s.minnaertK) * pow(max(NoV, 1e-4f), s.minnaertK - 1.f);
    }
    return max(NoL, 0.f); // lambert (only used for the CPU reference; the shader uses the reference term)
}

/// Blinn-Phong exponent: the record's shininess, or the GGX-equivalent 2 / a^2 - 2 from the roughness when it is <= 0.
float blinnExponent(const StyleRecord s, float roughness)
{
    if (s.shininess > 0.f) return s.shininess;
    const float a = max(roughness * roughness, 0.0064f);
    return max(2.f / (a * a) - 2.f, 1.f);
}

/// Specular term per unit Li (cosine included where the model has one); 0 for NoL <= 0.
float3 styledSpecular(const StyleRecord s, const BSDFProperties p, const float3 N, const float3 V, const float3 L, float3 ggxSpec)
{
    if (s.specModel == 0) return ggxSpec;
    const float NoL = dot(N, L);
    if (NoL <= kMinCosTheta || dot(N, V) <= kMinCosTheta) return float3(0.f);
    const float NoH = saturate(dot(N, normalize(V + L)));
    const float n = blinnExponent(s, p.roughness);
    const float lobe = pow(NoH, n);
    if (s.specModel == 1) return p.specularReflectance * ((n + 8.f) / (8.f * M_PI) * lobe * NoL);
    // hardSpec: one smooth threshold on the normalized lobe; highlight only.
    const float e0 = s.hardThreshold - s.hardSoftness, e1 = s.hardThreshold + s.hardSoftness;
    const float t = e1 > e0 ? smoothstep(e0, e1, lobe) : (lobe >= s.hardThreshold ? 1.f : 0.f);
    return p.specularReflectance * t;
}

// ---------------------------------------------------------------------------

ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, out uint materialID)
{
    VertexData v = {};
    materialID = 0;
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

    float3 diffuse = float3(0.f), specular = float3(0.f);   // reference
    float3 sDiffuse = float3(0.f), sSpecular = float3(0.f); // styled
    float3 other = float3(0.f);

    if (hit.isValid())
    {
        uint materialID;
        const ShadingData sd = loadShadingData(hit, rayDir, materialID);
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        const BSDFProperties props = mi.getProperties(sd);
        if (is_valid(gMaterialId))
        {
            const float4 m = gMaterialId[pixel];
            if (m.a > 0.f) materialID = uint(m.g + 0.5f);
        }
        const StyleRecord st = loadRecord(findRecord(materialID));
        const float3 N = sd.frame.N;
        const float NoVs = dot(N, sd.V);
        const float3 albedo = props.diffuseReflectionAlbedo;
        float3 sDirect = float3(0.f);

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
            float vis = 1.f;
            if (is_valid(gVisibility) && int(i) == gShadowLightIndex) vis = gVisibility[pixel].r;
            if (vis <= 0.f) continue;
            const float NdotL = dot(orientedFaceN, ls.dir);
            const bool culled = (NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection);
            float3 fd = float3(0.f), fs = float3(0.f);
            if (!culled)
            {
                evalLambertGGX(props, N, sd.V, ls.dir, fd, fs);
                diffuse += fd * ls.Li * vis;
                specular += fs * ls.Li * vis;
            }
            // Styled diffuse: wrap / half-Lambert also light the side facing away from the light (no face culling).
            // lambert = the reference term itself (so the default record reproduces the reference bit for bit).
            const float NoL = dot(N, ls.dir);
            if (st.diffuseModel == 0) sDirect += fd * ls.Li * vis;
            else if (NoVs > kMinCosTheta && hasReflection) sDirect += albedo * (M_1_PI * styledDiffuseFactor(st, NoL, NoVs)) * ls.Li * vis;
            if (!culled) sSpecular += styledSpecular(st, props, N, sd.V, ls.dir, fs) * ls.Li * vis;
        }
#endif
        float3 ambient = float3(0.f);
#if USE_ENV_LIGHT
        const float NoV = saturate(dot(N, sd.V));
        ambient = props.diffuseReflectionAlbedo * shIrradiance(N) * M_1_PI;
        diffuse += ambient;
        const float3 R = reflect(-sd.V, N);
        const float3 Lspec = gScene.envMap.eval(R, props.roughness * gEnvSpecLodScale);
        const float3 envSpec = Lspec * envBRDFApprox(props.specularReflectance, props.roughness, NoV);
        specular += envSpec;
        sSpecular += envSpec;
#endif
        const float3 hemi = albedo * lerp(st.ground, st.sky, 0.5f * N.y + 0.5f);
        sDiffuse = lerp(sDirect + lerp(ambient, hemi, st.hemiMix), albedo, st.flatMix);
        sSpecular *= st.specScale;
    }
    else
    {
#if USE_ENV_BACKGROUND
        other = gScene.envMap.eval(rayDir);
#endif
    }

    const float3 ref = diffuse + specular + other;
    float3 color = ref;
    float3 outDiff = diffuse, outSpec = specular;
    if (gStyleMix != 0.f && hit.isValid())
    {
        color = lerp(ref, sDiffuse + sSpecular + other, gStyleMix);
        outDiff = lerp(diffuse, sDiffuse, gStyleMix);
        outSpec = lerp(specular, sSpecular, gStyleMix);
    }
    gOutputColor[pixel] = float4(color, 1.f);
    if (is_valid(gDiffuse)) gDiffuse[pixel] = float4(outDiff, 1.f);
    if (is_valid(gSpecular)) gSpecular[pixel] = float4(outSpec, 1.f);
}
