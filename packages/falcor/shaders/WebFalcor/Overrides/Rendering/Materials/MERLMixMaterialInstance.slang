/**
 * WebFalcor override of Rendering/Materials/MERLMixMaterialInstance.slang (docs §9).
 *
 * Two diffs vs upstream:
 * - `bufferID` is dropped: every measured table shares one material buffer, so
 *   the instance carries only the absolute byte offset of its BRDF's samples.
 * - `getExtraBSDFProperties` is dropped. It returns an `ExtraBSDFProperties`
 *   type that this Falcor revision declares nowhere (the upstream file does not
 *   compile as shipped), and nothing in the port's render passes asks for it.
 */
#include "Utils/Math/MathConstants.slangh"

__exported import Scene.Material.DiffuseSpecularData;
__exported import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.BSDFs.DiffuseSpecularBRDF;
import Rendering.Materials.MERLCommon;
import Utils.Math.MathHelpers;
import Scene.Scene;

// Enable to use fitted BRDF approximation for sampling.
#define SAMPLE_FITTED_BRDF 1

/** Implementation of the BSDF for the measured MERLMix material.
*/
struct MERLMixMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;    ///< Shading frame in world space.
    uint byteOffset;    ///< Byte offset of this BRDF's samples in the shared material buffer.
    float3 albedo;      ///< Approximate albedo.
    uint brdfIndex;     ///< Index of the BRDF selected at the shading point.
    DiffuseSpecularBRDF fittedBrdf;

    __init(const ShadingFrame sf, const uint byteOffset, const float3 albedo, const uint brdfIndex, const DiffuseSpecularData extraData)
    {
        this.sf = sf;
        this.byteOffset = byteOffset;
        this.albedo = albedo;
        this.brdfIndex = brdfIndex;

        // Setup BRDF approximation.
        this.fittedBrdf = DiffuseSpecularBRDF(extraData);
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        return evalLocal(wiLocal, woLocal);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};
#if SAMPLE_FITTED_BRDF
        if (!fittedBrdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg, BSDFContext())) return false;
#else
        woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf);
        result.lobeType = (uint)LobeType::DiffuseReflection;
#endif
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

        result.weight = evalLocal(wiLocal, woLocal) / result.pdf;

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

#if SAMPLE_FITTED_BRDF
        return fittedBrdf.evalPdf(wiLocal, woLocal, BSDFContext());
#else
        return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
#endif
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;

        // Return fitted roughness as approximation.
        p.roughness = fittedBrdf.roughness;

        // Return approximation of the albedos.
        // The BSDF reports only a diffuse reflection lobe, so leaving the other albedos as zero.
        p.diffuseReflectionAlbedo = albedo;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection | (uint)LobeType::SpecularReflection;
    }


    // Internal helpers

    /** Evaluate the BRDF in the local frame.
        The vectors are assumed to be in the upper hemisphere.
        \param[in] wi Incident direction in the local frame.
        \param[in] wo Outgoing direction in the local frame.
        \return f(wi, wo) * wo.z
    */
    float3 evalLocal(const float3 wi, const float3 wo)
    {
        return MERLCommon::eval(wi, wo, gScene.materials.getBuffer(0), byteOffset);
    }
};
