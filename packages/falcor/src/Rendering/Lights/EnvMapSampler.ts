/**
 * Environment map importance sampler mirroring
 * Falcor/Rendering/Lights/EnvMapSampler.{h,cpp}: builds a hierarchical
 * (full mip chain) luminance importance map with the unmodified upstream
 * EnvMapSamplerSetup.cs.slang kernel, then box-filters the chain via blits
 * (Texture::generateMips), exactly as native.
 */

import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { Texture } from "../../Core/API/Texture.js";
import { Sampler, TextureAddressingMode, TextureFilteringMode } from "../../Core/API/Sampler.js";
import { ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import type { EnvMap } from "../../Scene/Lights/EnvMap.js";

const kShaderFilenameSetup = "Rendering/Lights/EnvMapSamplerSetup.cs.slang";
const kDefaultDimension = 512;
const kDefaultSpp = 64;

export class EnvMapSampler {
    private readonly importanceMap: Texture;
    private readonly importanceSampler: Sampler;

    constructor(
        private readonly device: Device,
        ctx: RenderContext,
        envMap: EnvMap,
        dimension = kDefaultDimension,
        samples = kDefaultSpp,
    ) {
        this.importanceSampler = new Sampler(device, {
            magFilter: TextureFilteringMode.Point,
            minFilter: TextureFilteringMode.Point,
            mipFilter: TextureFilteringMode.Point,
            addressModeU: TextureAddressingMode.Clamp,
            addressModeV: TextureAddressingMode.Clamp,
            addressModeW: TextureAddressingMode.Clamp,
        });

        // log2(N)+1 mips from NxN ... 1x1.
        const mips = Math.log2(dimension) + 1;
        this.importanceMap = new Texture(device, {
            type: ResourceType.Texture2D,
            width: dimension,
            height: dimension,
            arraySize: 1,
            mipLevels: mips,
            format: ResourceFormat.R32Float,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget | ResourceBindFlags.UnorderedAccess,
            name: "EnvMapSampler::importanceMap",
        });

        const setup = ComputePass.create(device, { path: kShaderFilenameSetup });
        const root = setup.getRootVar();
        root["gEnvMap"] = envMap.texture;
        root["gEnvSampler"] = envMap.sampler;
        root["gImportanceMap"] = this.importanceMap;
        const samplesX = Math.max(1, Math.floor(Math.sqrt(samples)));
        const samplesY = Math.floor(samples / samplesX);
        root["CB"]["outputDim"] = [dimension, dimension];
        root["CB"]["outputDimInSamples"] = [dimension * samplesX, dimension * samplesY];
        root["CB"]["numSamples"] = [samplesX, samplesY];
        root["CB"]["invSamples"] = 1 / (samplesX * samplesY);
        setup.execute(ctx, dimension, dimension);

        this.importanceMap.generateMips(ctx);
    }

    bindShaderData(var_: ShaderVar): void {
        var_["importanceBaseMip"] = this.importanceMap.mipCount - 1; // 1x1 mip
        var_["importanceInvDim"] = [1 / this.importanceMap.width, 1 / this.importanceMap.height];
        var_["importanceMap"] = this.importanceMap;
        var_["importanceSampler"] = this.importanceSampler;
    }
}
