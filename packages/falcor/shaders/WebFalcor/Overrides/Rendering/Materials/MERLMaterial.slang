/**
 * WebFalcor override of Rendering/Materials/MERLMaterial.slang (docs §9).
 *
 * Diffs vs upstream: the measured data and the precomputed albedo LUT both live
 * in the one shared material buffer. WGSL has no binding arrays, so `bufferID`
 * is a byte offset into that buffer instead of a buffer index, and the LUT —
 * float data the packed RGBA8 material texture array cannot hold — is read from
 * the buffer at `texAlbedoLUT` (also a byte offset) and interpolated here
 * exactly as the linear texture sampler would.
 */
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.MERLMaterialInstance;
import Scene.Material.MERLMaterialData;

struct MERLMaterial : MaterialBase, IMaterial
{
    MERLMaterialData data;

    // Specify that the associated `MaterialInstance` type is `MERLMaterialInstance`.
    typedef MERLMaterialInstance MaterialInstance;

    MERLMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        // Evaluate the albedo for the current incident (view) direction.
        float3 albedo = {};
        float cosTheta = dot(sf.N, sd.V);
        if (cosTheta >= kMinCosTheta)
        {
            // The albedo is precomputed in a lookup table sampled with piecewise
            // linear interpolation. The values are sampled at cosTheta =
            // (1,2,...N)/N, so the -0.5/N offset picks the two nearest samples.
            const uint kLUTSize = MERLMaterialData::kAlbedoLUTSize;
            float u = saturate(cosTheta - 0.5f / kLUTSize);
            // Texel centres sit at (i + 0.5) / N; clamp addressing at both ends.
            float x = u * kLUTSize - 0.5f;
            float t = saturate(x - floor(x));
            int i0 = clamp((int)floor(x), 0, (int)kLUTSize - 1);
            int i1 = clamp(i0 + 1, 0, (int)kLUTSize - 1);
            ByteAddressBuffer lut = ms.getBuffer(0);
            float3 a0 = asfloat(lut.Load3(data.texAlbedoLUT.packedData + i0 * 16));
            float3 a1 = asfloat(lut.Load3(data.texAlbedoLUT.packedData + i1 * 16));
            albedo = lerp(a0, a1, t);
        }

        return MERLMaterialInstance(sf, data.bufferID, albedo, data.extraData);
    }

    [Differentiable]
    MERLMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
