/**
 * WebFalcor override of Rendering/Materials/MERLMixMaterial.slang (docs §9).
 *
 * Diffs vs upstream, all forced by "WGSL has no binding arrays" (§6.2):
 * - `bufferID` is the byte offset of BRDF 0 in the one shared material buffer,
 *   so the instance gets an absolute offset instead of a buffer index.
 * - `texAlbedoLUT` is the byte offset of the 256 x brdfCount float4 LUT, read
 *   from that buffer — float data the packed RGBA8 texture array cannot hold.
 * - `texIndexMap` is the byte offset of a `[width, height, one byte per texel]`
 *   block in the same buffer. The packed texture array shares a single linear
 *   sampler and interpolating BRDF indices is meaningless, so the fetch below
 *   reproduces upstream's dedicated point/wrap sampler exactly.
 */
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.MERLMixMaterialInstance;
import Scene.Material.MERLMixMaterialData;

struct MERLMixMaterial : MaterialBase, IMaterial
{
    MERLMixMaterialData data;

    // Specify that the associated `MaterialInstance` type is `MERLMixMaterialInstance`.
    typedef MERLMixMaterialInstance MaterialInstance;

    MERLMixMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());
            float3 encodedNormal = ms.sampleTexture(data.texNormalMap, s, sd.uv, float4(0.f), lod).rgb;
            sf = computeShadingFrameFromNormalMap(sd, data.getNormalMapType(), encodedNormal);
        }
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        ByteAddressBuffer brdfData = ms.getBuffer(0);

        // Fetch the BRDF index to use at the shading point: nearest texel with
        // wrap addressing, one byte per texel (upstream's 8-bit unorm red
        // channel scaled by 255). The index wraps around modulo the BRDF count.
        uint brdfIndex = 0;
        {
            uint mapOffset = data.texIndexMap.packedData;
            int width = (int)brdfData.Load(mapOffset);
            int height = (int)brdfData.Load(mapOffset + 4);
            int x = (int)floor(sd.uv.x * width) % width;
            int y = (int)floor(sd.uv.y * height) % height;
            if (x < 0) x += width;
            if (y < 0) y += height;
            uint texel = (uint)(y * width + x);
            uint word = brdfData.Load(mapOffset + 8 + (texel >> 2) * 4);
            brdfIndex = ((word >> ((texel & 3) * 8)) & 0xff) % data.brdfCount;
        }
        uint byteOffset = data.bufferID + brdfIndex * data.byteStride;

        // Evaluate the albedo for the current incident (view) direction.
        float3 albedo = {};
        float cosTheta = dot(sf.N, sd.V);

        if (cosTheta >= kMinCosTheta)
        {
            // The albedo is precomputed in a lookup table, which we sample using piecewise linear interpolation.
            // The columns store precomputed albedo sampled at cosTheta = (1,2,...N)/N, where N is the LUT size.
            // We offset the lookup by -0.5/N to interpolate the two nearest samples.
            // The rows store one lookup table per BRDF index.
            const uint kLUTSize = MERLMixMaterialData::kAlbedoLUTSize;
            float u = saturate(cosTheta - 0.5f / kLUTSize);
            float x = u * kLUTSize - 0.5f;
            float t = saturate(x - floor(x));
            int i0 = clamp((int)floor(x), 0, (int)kLUTSize - 1);
            int i1 = clamp(i0 + 1, 0, (int)kLUTSize - 1);
            uint row = data.texAlbedoLUT.packedData + brdfIndex * kLUTSize * 16;
            float3 a0 = asfloat(brdfData.Load3(row + i0 * 16));
            float3 a1 = asfloat(brdfData.Load3(row + i1 * 16));
            albedo = lerp(a0, a1, t);
        }

        // Load extra data for sampling.
        uint extraDataByteOffset = data.extraDataOffset + brdfIndex * data.extraDataStride;
        DiffuseSpecularData extraData = brdfData.Load<DiffuseSpecularData>(extraDataByteOffset);

        return MERLMixMaterialInstance(sf, byteOffset, albedo, brdfIndex, extraData);
    }

    [Differentiable]
    MERLMixMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
