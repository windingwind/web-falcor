/**
 * WebFalcor override of RenderPasses/RTXDIPass/FinalShading.cs.slang
 * (docs §4.3). Diffs vs upstream:
 * - The six output textures move from global scope INTO the FinalShading
 *   parameter block: as globals they claim their own bind group and the
 *   pipeline needs 5 (globals, gFinalShading, gRTXDI, gScene, materials) —
 *   WebGPU allows at most 4.
 * - Write-only outputs (WTexture2D + Store; WGSL restricts read_write
 *   storage textures to r32*).
 */
import LoadShadingData;
import Scene.RaytracingInline;
import Rendering.Materials.IsotropicGGX;
import Rendering.RTXDI.RTXDI;
import Utils.Sampling.TinyUniformSampleGenerator;

#define is_valid(name) (is_valid_##name != 0)

/**
 * Perform shading with final samples from ReSTIR.
 */
struct FinalShading
{
    // Static configuration.
    static const bool kUseEnvBackground = USE_ENV_BACKGROUND;

    Texture2D<PackedHitInfo> vbuffer;

    // WebFalcor: outputs live in the block (bind-group budget), write-only.
    WTexture2D<float4> gColor;
    WTexture2D<float4> gEmission;
    WTexture2D<float4> gDiffuseIllumination;
    WTexture2D<float4> gDiffuseReflectance;
    WTexture2D<float4> gSpecularIllumination;
    WTexture2D<float4> gSpecularReflectance;

    uint2 frameDim;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim))
            return;

        float3 color = {};
        float3 emission = {};
        float3 diffuseReflectance = {};
        float3 diffuseIllumination = {};
        float3 specularReflectance = {};
        float3 specularIllumination = {};
        float hitT = 10000.f; // TODO: Remove magic number also used in SpatioTemporalResampling pass.

        ShadingData sd;
        if (loadShadingData(pixel, frameDim, gScene.camera, vbuffer, sd))
        {
            // Create material instance and query its properties.
            let lod = ExplicitLodTextureSampler(0.f); // TODO: Implement texture level-of-detail.
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            let bsdfProperties = mi.getProperties(sd);

            // Get final sample from RTXDI.
            float3 dir;
            float distance;
            float3 Li;
            bool valid = gRTXDI.getFinalSample(pixel, dir, distance, Li);

            // Check light visibility.
            if (valid)
            {
                // Create a DXR 1.1 query object to trace a ray (the <1> means use alpha testing)
                SceneRayQuery<1> rayQuery;
                const Ray ray = Ray(sd.computeRayOrigin(), dir, 0.f, distance);
                if (!rayQuery.traceVisibilityRay(ray))
                {
                    valid = false;
                }
            }

            // Evaluate diffuse and specular lobes.
            float3 diffuse = {};
            float3 specular = {};
            if (valid)
            {
                TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gRTXDI.frameIndex);

                sd.mtl.setActiveLobes((uint)LobeType::DiffuseReflection);
                diffuse = mi.eval(sd, dir, sg) * Li;

                sd.mtl.setActiveLobes((uint)LobeType::SpecularReflection);
                specular = mi.eval(sd, dir, sg) * Li;

                hitT = distance;
            }

            emission = bsdfProperties.emission;
            color = diffuse + specular + emission;

            // RTXDI uses a simple material model with only diffuse and specular reflection lobes.
            // We query the BSDF for the respective albedos and use them for demodulation below.

            // Demodulate diffuse reflectance (albedo) from diffuse color.
            diffuseReflectance = bsdfProperties.diffuseReflectionAlbedo;
            float3 diffuseFactor = select(diffuseReflectance <= 0.f, 0.f, 1.f / diffuseReflectance);
            diffuseIllumination = diffuse * diffuseFactor;

            // Demodulate preintegrated specular reflectance from specular color.
            // We use an approximate shading model with a single specular lobe.
            float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
            float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectance, ggxAlpha, NdotV);
            float3 specularFactor = select(specularReflectance <= 0.f, 0.f, 1.f / specularReflectance);
            specularIllumination = specular * specularFactor;
        }
        else
        {
            if (kUseEnvBackground)
            {
                float3 dir = gScene.camera.computeRayPinhole(pixel, frameDim).dir;
                emission = gScene.envMap.eval(dir);
                color = emission;
            }
        }

        // Write active outputs.
        if (is_valid(gColor))
            gColor.Store(pixel, float4(color, 1.f));
        if (is_valid(gEmission))
            gEmission.Store(pixel, float4(emission, 1.f));
        if (is_valid(gDiffuseReflectance))
            gDiffuseReflectance.Store(pixel, float4(diffuseReflectance, 1.f));
        if (is_valid(gDiffuseIllumination))
            gDiffuseIllumination.Store(pixel, float4(diffuseIllumination, hitT));
        if (is_valid(gSpecularReflectance))
            gSpecularReflectance.Store(pixel, float4(specularReflectance, 1.f));
        if (is_valid(gSpecularIllumination))
            gSpecularIllumination.Store(pixel, float4(specularIllumination, hitT));
    }
};

ParameterBlock<FinalShading> gFinalShading;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gFinalShading.execute(dispatchThreadId.xy);
}
