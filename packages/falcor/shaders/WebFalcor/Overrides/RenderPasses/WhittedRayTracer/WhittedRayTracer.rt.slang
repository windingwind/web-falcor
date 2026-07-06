/**
 * WebFalcor override of RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang
 * (DESIGN.md §4.3). Diffs vs upstream:
 *  - RT pipeline -> compute megakernel over SceneRayQuery (DESIGN.md §5):
 *    raygen becomes main(), TraceRay becomes traceRay/traceVisibilityRay,
 *    the hit/miss shaders become functions taking the TriangleHit.
 *  - gOutputColor is write-only (WGSL r32*-only read_write rule).
 *  - The interface-typed `ITextureSampler lod` locals are restructured into
 *    generic helpers (WGSL has no dynamic dispatch for existentials).
 *  - gFrameDim added to CB (no DispatchRaysDimensions in compute).
 */
#include "Utils/Math/MathConstants.slangh"
import Scene.RaytracingInline;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Rendering.Materials.Fresnel;
import WhittedRayTracerTypes;

cbuffer CB
{
    uint gFrameCount;
    uint gPRNGDimension;
    float gScreenSpacePixelSpreadAngle;
    uint2 gFrameDim;
}

// Inputs
Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldShadingTangent;
Texture2D<float4> gWorldFaceNormal;
Texture2D<float2> gTextureCoord;
Texture2D<float4> gTextureGrads; // Optional
Texture2D<uint4> gMaterialData;
Texture2D<PackedHitInfo> gVBuffer;

// Outputs
WTexture2D<float4> gOutputColor;

#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const TexLODMode kTexLODMode = TexLODMode(TEX_LOD_MODE);
static const RayConeMode kRayConeMode = RayConeMode(RAY_CONE_MODE);
static const RayFootprintFilterMode kRayConeFilterMode = RayFootprintFilterMode(RAY_CONE_FILTER_MODE);
static const RayFootprintFilterMode kRayDiffFilterMode = RayFootprintFilterMode(RAY_DIFF_FILTER_MODE);
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0.25f, 0.25f, 0.25f);
static const float kRayTMax = FLT_MAX;
static const bool kUseRoughnessToVariance = USE_ROUGHNESS_TO_VARIANCE;
static const bool kVisualizeSurfaceSpread = VISUALIZE_SURFACE_SPREAD;
static const bool kUseFresnelAsBRDF = USE_FRESNEL_AS_BRDF;

float roughnessToSpread(float roughness)
{
    const float ggxAlpha = roughness * roughness;
    const float s = ggxAlpha * ggxAlpha;
    const float sigma2 = (s / max(0.0001f, (1.0f - s))) * 0.5f;
    return 2.0f * sqrt(sigma2);
}

float3 getFlippedShadingNormal(const ShadingData sd)
{
    return !sd.frontFacing && sd.mtl.isDoubleSided() ? -sd.frame.N : sd.frame.N;
}

struct ScatterRayData
{
    float3 radiance;
    bool terminated;
    bool hasHitRefractiveMaterial;
    float3 thp;
    uint pathLength;
    float3 origin;
    float3 direction;
    RayCone rayCone;
    RayDiff rayDiff;
    SampleGenerator sg;

    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.hasHitRefractiveMaterial = false;
        this.pathLength = 0;
        this.radiance = float3(0.0f, 0.0f, 0.0f);
        this.thp = float3(1.0f, 1.0f, 1.0f);
        this.origin = float3(0.0f, 0.0f, 0.0f);
        this.direction = float3(0.0f, 0.0f, 0.0f);
        this.sg = sg;
        this.rayCone = RayCone(0.f, 0.f);
        this.rayDiff = RayDiff(float3(0.f), float3(0.f), float3(0.f), float3(0.f));
    }
};

ShadingData loadShadingData(const uint2 pixel, const float3 posW, const float3 rayDir)
{
    float3 normal = gWorldShadingNormal[pixel].xyz;
    float4 tangent = gWorldShadingTangent[pixel];
    float3 faceNormal = gWorldFaceNormal[pixel].xyz;
    float2 texC = gTextureCoord[pixel];
    uint materialID = gMaterialData[pixel].x;

    VertexData v = {};
    v.posW = posW;
    v.normalW = normalize(normal);
    v.tangentW = float4(normalize(tangent.xyz), tangent.w);
    v.faceNormalW = normalize(faceNormal);
    v.texC = texC;
    v.coneTexLODValue = 0.f;

    return gScene.materials.prepareShadingData(v, materialID, -rayDir);
}

float computeEta(const ShadingData sd)
{
    float insideIoR = gScene.materials.evalIoR(sd.materialID);
    float eta = sd.frontFacing ? (sd.IoR / insideIoR) : (insideIoR / sd.IoR);
    return eta;
}

bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    const Ray ray = Ray(origin, dir, 0.f, distance);
    SceneRayQuery<1> query;
    return query.traceVisibilityRay(ray, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff);
}

float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, const float3 rayOrigin, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    float invPdf = 1.0f / lightCount;

    float3 contribution = float3(0.0f);
    for (uint lightIndex = 0; lightIndex < lightCount; lightIndex++)
    {
        AnalyticLightSample ls;
        bool valid = sampleLight(rayOrigin, gScene.getLight(lightIndex), sg, ls);

        if (valid && dot(ls.dir, getFlippedShadingNormal(sd)) > kMinCosTheta)
        {
            bool V = traceShadowRay(rayOrigin, ls.dir, ls.distance);
            contribution += V ? mi.eval(sd, ls.dir, sg) * ls.Li * invPdf : float3(0);
        }
    }
    return contribution;
}

void generateReflectionRay(
    const ShadingData sd,
    const BSDFProperties bsdfProperties,
    const float3 rayOrigin,
    const float3 rayDir,
    const float3 normal,
    bool tir,
    inout ScatterRayData rayData
)
{
    float3 wo = reflect(rayDir, normal);
    rayData.origin = rayOrigin;
    rayData.direction = wo;
    float3 specular = bsdfProperties.specularReflectance;

    if (kUseFresnelAsBRDF)
    {
        rayData.thp *= tir ? 1.0f : evalFresnelSchlick(specular, 1.0f, max(0.00001f, dot(normal, wo)));
    }
    else
    {
        rayData.thp *= tir ? 1.0f : (specular.x + specular.y + specular.z) / 3.0f;
    }
}

/** Scatter-miss (was [shader("miss")] scatterMiss). */
void handleScatterMiss(inout ScatterRayData rayData, const float3 rayDir)
{
    rayData.terminated = true;

    if (kUseEnvLight)
    {
        float lod = 0.0f;
        if (kTexLODMode == TexLODMode::RayCones)
        {
            lod = computeEnvironmentMapLOD(rayData.rayCone.getSpreadAngle(), gScene.envMap.envMap);
        }
        else if (kTexLODMode == TexLODMode::RayDiffs)
        {
            lod = computeEnvironmentMapLOD(rayData.rayDiff.getdDdx(), rayData.rayDiff.getdDdy(), gScene.envMap.envMap);
        }

        float3 Le = gScene.envMap.eval(rayDir, lod);
        rayData.radiance += rayData.thp * Le;
    }
}

/** Tail of the closest-hit shader after the LOD sampler is known; generic over
 *  the concrete sampler type (WGSL cannot dispatch interface-typed locals). */
void shadeScatterHit<L : ITextureSampler>(
    const L lod,
    const ShadingData sd,
    const VertexData v,
    const float eta0,
    const float3 refractedRayDir0,
    const bool tir0,
    const float surfaceSpreadAngle,
    const float3 rayOrg,
    const float3 rayDir,
    inout ScatterRayData rayData
)
{
    float eta = eta0;
    float3 refractedRayDir = refractedRayDir0;
    bool tir = tir0;

    let mi = gScene.materials.getMaterialInstance(sd, lod);
    let bsdfProperties = mi.getProperties(sd);

    if (kTexLODMode == TexLODMode::RayCones)
    {
        float spread = surfaceSpreadAngle;
#if USE_ROUGHNESS_TO_VARIANCE
        spread += roughnessToSpread(bsdfProperties.roughness);
#endif
        if (eta == 1.0f)
        {
            rayData.rayCone = rayData.rayCone.addToSpreadAngle(2.0f * spread);
        }
        else
        {
            if (!refractRayCone(
                    rayData.rayCone, rayOrg, rayDir, sd.posW, getFlippedShadingNormal(sd), spread, eta, refractedRayDir
                ))
            {
                rayData.rayCone = rayData.rayCone.addToSpreadAngle(spread);
                tir = true;
            }
        }
    }

    if (kUseEmissiveLights)
    {
        rayData.radiance += rayData.thp * bsdfProperties.emission;
    }

    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    float3 rayOriginAtHitPoint = sd.computeRayOrigin();

    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, rayOriginAtHitPoint, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
    }

    if (eta == 1.0f || tir)
    {
        generateReflectionRay(sd, bsdfProperties, rayOriginAtHitPoint, rayDir, v.normalW, tir, rayData);
    }
    else
    {
        rayData.direction = refractedRayDir;
        rayData.origin = sd.computeRayOrigin(false);

        if (kUseFresnelAsBRDF)
        {
            rayData.thp *=
                1.0f -
                evalFresnelSchlick(bsdfProperties.specularReflectance, 1.0f, max(0.00001f, dot(getFlippedShadingNormal(sd), -rayDir)));
        }
        else
        {
            float roughness = bsdfProperties.roughness;
            rayData.thp *= (1.0f - roughness);
        }
        rayData.hasHitRefractiveMaterial = true;
    }
    rayData.pathLength++;

    if (max(max(rayData.thp.x, rayData.thp.y), rayData.thp.z) < 0.01f)
    {
        rayData.terminated = true;
        return;
    }
}

/** Scatter closest hit (was [shader("closesthit")] scatterClosestHit). */
void handleScatterHit(const TriangleHit triangleHit, const float hitT, inout ScatterRayData rayData)
{
    const float3 rayDir = rayData.direction;
    const float3 rayOrg = rayData.origin;
    float3 refractedRayDir = {};
    bool tir = false;

    const GeometryInstanceID instanceID = triangleHit.instanceID;
    const uint primitiveIndex = triangleHit.primitiveIndex;
    const float3 barycentrics = triangleHit.getBarycentricWeights();
    const uint materialID = gScene.getMaterialID(instanceID);
    float surfaceSpreadAngle = 0.0f;
    float eta = 1.f;

    if (kTexLODMode == TexLODMode::RayCones)
    {
        VertexData v = gScene.getVertexDataRayCones(instanceID, primitiveIndex, barycentrics);
        float curvature = gScene.computeCurvatureIsotropic(instanceID, primitiveIndex);

        if (kRayConeMode == RayConeMode::Combo || kRayConeMode == RayConeMode::Unified)
        {
            float rayConeWidth = rayData.rayCone.getSpreadAngle() * hitT + rayData.rayCone.getWidth();
            surfaceSpreadAngle = computeSpreadAngleFromCurvatureIso(curvature, rayConeWidth, rayDir, v.normalW);
        }

        rayData.rayCone = rayData.rayCone.propagateDistance(hitT);
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
        eta = computeEta(sd);

        if (kRayConeFilterMode == RayFootprintFilterMode::Isotropic ||
            (kRayConeFilterMode == RayFootprintFilterMode::AnisotropicWhenRefraction && !rayData.hasHitRefractiveMaterial))
        {
            float lambda = rayData.rayCone.computeLOD(v.coneTexLODValue, rayDir, v.normalW);
            shadeScatterHit(ExplicitRayConesLodTextureSampler(lambda), sd, v, eta, refractedRayDir, tir, surfaceSpreadAngle, rayOrg, rayDir, rayData);
        }
        else
        {
            float3 positions[3];
            float2 txcoords[3];
            gScene.getVertexPositionsW(instanceID, primitiveIndex, positions);
            gScene.getVertexTexCoords(instanceID, primitiveIndex, txcoords);

            float coneWidthAtHitPoint = rayData.rayCone.getWidth();
            float2 ddx = float2(0), ddy = float2(0);
            computeAnisotropicEllipseAxes(v.posW, v.faceNormalW, rayDir, coneWidthAtHitPoint, positions, txcoords, v.texC, ddx, ddy);
            shadeScatterHit(ExplicitGradientTextureSampler(ddx, ddy), sd, v, eta, refractedRayDir, tir, surfaceSpreadAngle, rayOrg, rayDir, rayData);
        }
    }
    else if (kTexLODMode == TexLODMode::RayDiffs)
    {
        float2 dUVdx, dUVdy;
        float2 dBarydx, dBarydy;
        float3 geometricNormal = gScene.getFaceNormalW(instanceID, primitiveIndex);
        RayDiff newRayDiff = rayData.rayDiff.propagate(rayOrg, rayDir, hitT, geometricNormal);

        StaticVertexData triangleVertices[3];
        VertexData v = gScene.getVertexData(triangleHit, triangleVertices);

        float3 unnormalizedN;
        float3 normals[3];

        prepareRayDiffAtHitPoint(
            v,
            triangleVertices,
            barycentrics,
            rayDir,
            gScene.getWorldMatrix(instanceID),
            gScene.getInverseTransposeWorldMatrix(instanceID),
            newRayDiff,
            unnormalizedN,
            normals,
            dBarydx,
            dBarydy,
            dUVdx,
            dUVdy
        );

        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
        eta = computeEta(sd);

        if (eta == 1.0f)
        {
            reflectRayDifferential(newRayDiff, rayDir, unnormalizedN, v.normalW, dBarydx, dBarydy, normals);
        }
        else
        {
            if (!refractRayDifferential(
                    newRayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals, eta, refractedRayDir
                ))
            {
                reflectRayDifferential(newRayDiff, rayDir, unnormalizedN, v.normalW, dBarydx, dBarydy, normals);
                tir = true;
            }
        }
        rayData.rayDiff = newRayDiff;

        if (kRayDiffFilterMode == RayFootprintFilterMode::Isotropic ||
            (kRayDiffFilterMode == RayFootprintFilterMode::AnisotropicWhenRefraction && !rayData.hasHitRefractiveMaterial))
        {
            shadeScatterHit(ExplicitRayDiffsIsotropicTextureSampler(dUVdx, dUVdy), sd, v, eta, refractedRayDir, tir, surfaceSpreadAngle, rayOrg, rayDir, rayData);
        }
        else
        {
            shadeScatterHit(ExplicitGradientTextureSampler(dUVdx, dUVdy), sd, v, eta, refractedRayDir, tir, surfaceSpreadAngle, rayOrg, rayDir, rayData);
        }
    }
    else // kTexLODMode == TexLODMode::Mip0
    {
        VertexData v = gScene.getVertexData(triangleHit);
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
        eta = computeEta(sd);

        if (eta != 1.0f)
        {
            if (!refractWithTIR(rayDir, getFlippedShadingNormal(sd), eta, refractedRayDir))
            {
                tir = true;
            }
        }
        shadeScatterHit(ExplicitLodTextureSampler(0.f), sd, v, eta, refractedRayDir, tir, surfaceSpreadAngle, rayOrg, rayDir, rayData);
    }
}

/** Traces a scatter ray (was TraceRay over hit group 0). */
void traceScatterRay(inout ScatterRayData rayData)
{
    const Ray ray = Ray(rayData.origin, rayData.direction, 0.f, kRayTMax);
    SceneRayQuery<1> query;
    float hitT;
    const HitInfo hit = query.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        handleScatterHit(hit.getTriangleHit(), hitT, rayData);
    }
    else
    {
        handleScatterMiss(rayData, ray.dir);
    }
}

/** Entry point (was [shader("raygeneration")] rayGen). */
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 launchIndex = dispatchThreadId.xy;
    const uint2 launchDim = gFrameDim;
    if (any(launchIndex >= launchDim))
        return;

    float3 outColor = float3(0.0f);

    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(launchIndex, launchDim);
    const float3 rayDir = normalize(nonNormalizedRayDir);
    const float4 worldPos = gWorldPosition[launchIndex];

    if (worldPos.w != 0.f)
    {
        const ShadingData sd = loadShadingData(launchIndex, worldPos.xyz, rayDir);
        const float eta = computeEta(sd);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount);
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        const float3 rayOriginAtHitPoint = sd.computeRayOrigin();
        ScatterRayData rayData = ScatterRayData(sg);

        // Primary-hit shading with the concrete sampler per branch (see header).
        BSDFProperties bsdfProperties;
        if (is_valid(gTextureGrads))
        {
            float4 texGrads = gTextureGrads[launchIndex];
            let mi = gScene.materials.getMaterialInstance(sd, ExplicitGradientTextureSampler(texGrads.xy, texGrads.zw));
            bsdfProperties = mi.getProperties(sd);
            outColor += bsdfProperties.emission;
            outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, mi, rayOriginAtHitPoint, rayData.sg) : float3(0.f);
        }
        else
        {
            let mi = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0.f));
            bsdfProperties = mi.getProperties(sd);
            outColor += bsdfProperties.emission;
            outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, mi, rayOriginAtHitPoint, rayData.sg) : float3(0.f);
        }

        float3 refractedRayDir = float3(0.0f);
        bool tir = false;

        if (kTexLODMode == TexLODMode::RayCones)
        {
            float surfaceSpreadAngle = 0.0f;
            const float3 rayOrg = gScene.camera.getPosition();
            const float hitT = length(rayOrg - worldPos.xyz);

            const HitInfo hit = HitInfo(gVBuffer[launchIndex]);
            if (hit.isValid() && hit.getType() == HitType::Triangle)
            {
                const TriangleHit triangleHit = hit.getTriangleHit();
                const float4x4 worldMat = gScene.getWorldMatrix(triangleHit.instanceID);
                const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(triangleHit.instanceID);
                const uint3 vertexIndices = gScene.getIndices(triangleHit.instanceID, triangleHit.primitiveIndex);
                const float3 barycentrics = triangleHit.getBarycentricWeights();
                float2 txcoords[3], dBarydx, dBarydy, dUVdx, dUVdy;

                StaticVertexData vertices[3] = {
                    gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2])
                };

                float curvature = gScene.computeCurvatureIsotropicFirstHit(triangleHit.instanceID, triangleHit.primitiveIndex, rayDir);

                if (kRayConeMode == RayConeMode::Combo)
                {
                    RayDiff rayDiff;
                    float3 dDdx, dDdy, unnormalizedN, normals[3], dNdx, dNdy, edge1, edge2;
                    prepareVerticesForRayDiffs(
                        rayDir, vertices, worldMat, worldInvTransposeMat, barycentrics, edge1, edge2, normals, unnormalizedN, txcoords
                    );
                    computeRayDirectionDifferentials(
                        nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, launchDim, dDdx, dDdy
                    );
                    RayDiff rd = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy);
                    rayDiff = rd.propagate(worldPos.xyz, rayDir, hitT, getFlippedShadingNormal(sd));
                    computeBarycentricDifferentials(rayDiff, rayDir, edge1, edge2, sd.faceN, dBarydx, dBarydy);
                    computeNormalDifferentials(rayDiff, unnormalizedN, dBarydx, dBarydy, normals, dNdx, dNdy);
                    float3 right =
                        normalize(gScene.camera.computeNonNormalizedRayDirPinhole(launchIndex + uint2(1, 0), launchDim)) - rayDir;
                    float3 up = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(launchIndex + uint2(0, 1), launchDim)) - rayDir;
                    surfaceSpreadAngle = computeScreenSpaceSurfaceSpreadAngle(right, up, dNdx, dNdy);
                }
                else // kRayConeMode == RayConeMode::Unified
                {
                    surfaceSpreadAngle = computeSpreadAngleFromCurvatureIso(
                        curvature, hitT * gScreenSpacePixelSpreadAngle, rayDir, getFlippedShadingNormal(sd)
                    );
                }

#if USE_ROUGHNESS_TO_VARIANCE
                surfaceSpreadAngle += roughnessToSpread(bsdfProperties.roughness);
#endif
                if (kVisualizeSurfaceSpread)
                {
                    const float eps = 1.0e-4;
                    float r = surfaceSpreadAngle > eps ? surfaceSpreadAngle : 0.0f;
                    float g = surfaceSpreadAngle <= -eps ? -surfaceSpreadAngle : 0.0f;
                    float b = abs(surfaceSpreadAngle) < eps ? 1.0f : 0.0f;
                    gOutputColor.Store(launchIndex, float4(r, g, b, 1.0f) * 10.0f);
                    return;
                }

                RayCone rc = RayCone(0.0f, gScreenSpacePixelSpreadAngle);
                rayData.rayCone = rc.propagateDistance(hitT);
                if (eta == 1.0f)
                {
                    rayData.rayCone = rayData.rayCone.addToSpreadAngle(2.0f * surfaceSpreadAngle);
                }
                else
                {
                    float3 hitPoint = rayOrg + rayDir * hitT;
                    if (!refractRayCone(
                            rayData.rayCone, rayOrg, rayDir, hitPoint, getFlippedShadingNormal(sd), surfaceSpreadAngle, eta, refractedRayDir
                        ))
                    {
                        rayData.rayCone = rayData.rayCone.addToSpreadAngle(surfaceSpreadAngle);
                        tir = true;
                    }
                }
            }
        }
        else if (kTexLODMode == TexLODMode::RayDiffs)
        {
            float3 dDdx, dDdy;
            const float3 rayOrg = gScene.camera.getPosition();
            const float hitT = length(rayOrg - worldPos.xyz);

            computeRayDirectionDifferentials(
                nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, launchDim, dDdx, dDdy
            );

            rayData.rayDiff = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy);
            rayData.rayDiff = rayData.rayDiff.propagate(worldPos.xyz, rayDir, hitT, sd.faceN);

            const HitInfo hit = HitInfo(gVBuffer[launchIndex]);
            if (hit.isValid() && hit.getType() == HitType::Triangle)
            {
                const TriangleHit triangleHit = hit.getTriangleHit();
                const float4x4 worldMat = gScene.getWorldMatrix(triangleHit.instanceID);
                const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(triangleHit.instanceID);
                const uint3 vertexIndices = gScene.getIndices(triangleHit.instanceID, triangleHit.primitiveIndex);
                const float3 barycentrics = triangleHit.getBarycentricWeights();
                float3 unnormalizedN, normals[3], dNdx, dNdy, edge1, edge2;
                float2 txcoords[3], dBarydx, dBarydy, dUVdx, dUVdy;

                StaticVertexData vertices[3] = {
                    gScene.getVertex(vertexIndices[0]),
                    gScene.getVertex(vertexIndices[1]),
                    gScene.getVertex(vertexIndices[2]),
                };
                prepareVerticesForRayDiffs(
                    rayDir, vertices, worldMat, worldInvTransposeMat, barycentrics, edge1, edge2, normals, unnormalizedN, txcoords
                );

                computeBarycentricDifferentials(rayData.rayDiff, rayDir, edge1, edge2, sd.faceN, dBarydx, dBarydy);

                if (eta == 1.0f)
                {
                    reflectRayDifferential(rayData.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals);
                }
                else
                {
                    if (!refractRayDifferential(
                            rayData.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals, eta, refractedRayDir
                        ))
                    {
                        reflectRayDifferential(rayData.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals);
                        tir = true;
                    }
                }
            }
        }
        else // Mip0.
        {
            if (!refractWithTIR(rayDir, getFlippedShadingNormal(sd), eta, refractedRayDir))
            {
                tir = true;
            }
        }

        // Re-query BSDF properties for the scatter decision (same values as above).
        if (eta == 1.0f || tir)
        {
            generateReflectionRay(sd, bsdfProperties, rayOriginAtHitPoint, rayDir, getFlippedShadingNormal(sd), tir, rayData);
        }
        else
        {
            rayData.direction = refractedRayDir;
            rayData.origin = sd.computeRayOrigin(false);

            if (kUseFresnelAsBRDF)
            {
                rayData.thp *=
                    1.0f -
                    evalFresnelSchlick(bsdfProperties.specularReflectance, 1.0f, max(0.00001f, dot(getFlippedShadingNormal(sd), -rayDir)));
            }
            else
            {
                rayData.thp *= (1.0f - bsdfProperties.roughness);
            }
        }

        if (bsdfProperties.roughness <= 0.19f)
        {
            for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
            {
                traceScatterRay(rayData);
            }
        }
        outColor += rayData.radiance;
    }
    else
    {
        if (kUseEnvBackground)
        {
            float lod = 0.0f;
            if (kTexLODMode == TexLODMode::RayCones)
            {
                lod = computeEnvironmentMapLOD(gScreenSpacePixelSpreadAngle, gScene.envMap.envMap);
            }
            else if (kTexLODMode == TexLODMode::RayDiffs)
            {
                float3 dDdx, dDdy;
                computeRayDirectionDifferentials(
                    nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, launchDim, dDdx, dDdy
                );
                lod = computeEnvironmentMapLOD(dDdx, dDdy, gScene.envMap.envMap);
            }
            outColor = gScene.envMap.eval(rayDir, lod);
        }
        else
        {
            outColor = kDefaultBackgroundColor;
        }
    }

    gOutputColor.Store(launchIndex, float4(outColor, 1.0f));
}
