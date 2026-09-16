/**
 * WebFalcor override of Rendering/Materials/MERLMaterialInstance.slang (docs §9).
 *
 * Only `evalLocal` differs: all measured tables share one material buffer, so
 * the instance's `bufferID` is the byte offset of this BRDF's samples (the same
 * offset mechanism upstream uses for MERLMix) rather than a buffer index.
 */
#include "Utils/Math/MathConstants.slangh"
__exported import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.BSDFs.DiffuseSpecularBRDF;
import Rendering.Materials.MERLCommon;
import Utils.Math.MathHelpers;
import Scene.Scene;

// Enable to use fitted BRDF approximation for sampling.
#define SAMPLE_FITTED_BRDF 1

struct MERLMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;    ///< Shading frame in world space.
    uint byteOffset;    ///< Byte offset of this BRDF's samples in the shared material buffer.
    float3 albedo;      ///< Approximate albedo.
    DiffuseSpecularBRDF fittedBrdf;

    __init(const ShadingFrame sf, const uint byteOffset, const float3 albedo, const DiffuseSpecularData extraData)
    {
        this.sf = sf;
        this.byteOffset = byteOffset;
        this.albedo = albedo;
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
        if (!fittedBrdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg, BSDFContext())) return false;
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
        return fittedBrdf.evalPdf(wiLocal, woLocal, BSDFContext());
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.guideNormal = sf.N;
        // Compute approximation of the perceived roughness. For now assume a medium roughness.
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

    /** Evaluate the MERL BRDF in the local frame; returns f(wi, wo) * wo.z. */
    float3 evalLocal(const float3 wi, const float3 wo)
    {
        return MERLCommon::eval(wi, wo, gScene.materials.getBuffer(0), byteOffset);
    }
};
