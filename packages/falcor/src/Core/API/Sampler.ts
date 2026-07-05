/**
 * Texture sampler mirroring Falcor/Core/API/Sampler.h.
 *
 * TextureReductionMode.Min/Max (D3D12 MIN/MAX filtering) has no WebGPU
 * equivalent and throws UnsupportedFeatureError (parity matrix §8.1).
 */

import { UnsupportedFeatureError } from "../Error.js";
import { ComparisonFunc, toGpuCompareFunction } from "./Types.js";
import type { Device } from "./Device.js";

export enum TextureFilteringMode {
    Point,
    Linear,
}

export enum TextureAddressingMode {
    Wrap,
    Clamp,
    Mirror,
    Border, // 🟡 WebGPU has clamp-to-edge only; Border is approximated by Clamp (no border color)
    MirrorOnce, // 🟡 approximated by Mirror
}

export enum TextureReductionMode {
    Standard,
    Comparison,
    Min, // ❌ not in WebGPU
    Max, // ❌ not in WebGPU
}

export interface SamplerDesc {
    magFilter?: TextureFilteringMode;
    minFilter?: TextureFilteringMode;
    mipFilter?: TextureFilteringMode;
    maxAnisotropy?: number;
    minLod?: number;
    maxLod?: number;
    lodBias?: number; // ❌ WebGPU has no sampler LOD bias; must be applied in-shader (documented)
    comparisonFunc?: ComparisonFunc;
    reductionMode?: TextureReductionMode;
    addressModeU?: TextureAddressingMode;
    addressModeV?: TextureAddressingMode;
    addressModeW?: TextureAddressingMode;
}

function toGpuAddressMode(mode: TextureAddressingMode): GPUAddressMode {
    switch (mode) {
        case TextureAddressingMode.Wrap: return "repeat";
        case TextureAddressingMode.Clamp: return "clamp-to-edge";
        case TextureAddressingMode.Mirror: return "mirror-repeat";
        case TextureAddressingMode.Border: return "clamp-to-edge";
        case TextureAddressingMode.MirrorOnce: return "mirror-repeat";
    }
}

function toGpuFilter(mode: TextureFilteringMode): GPUFilterMode {
    return mode === TextureFilteringMode.Linear ? "linear" : "nearest";
}

export class Sampler {
    readonly gpuSampler: GPUSampler;
    readonly desc: Required<Omit<SamplerDesc, "comparisonFunc" | "reductionMode">> &
        Pick<SamplerDesc, "comparisonFunc" | "reductionMode">;

    constructor(device: Device, desc: SamplerDesc = {}) {
        const full = {
            magFilter: desc.magFilter ?? TextureFilteringMode.Linear,
            minFilter: desc.minFilter ?? TextureFilteringMode.Linear,
            mipFilter: desc.mipFilter ?? TextureFilteringMode.Linear,
            maxAnisotropy: desc.maxAnisotropy ?? 1,
            minLod: desc.minLod ?? 0,
            maxLod: desc.maxLod ?? 1000,
            lodBias: desc.lodBias ?? 0,
            comparisonFunc: desc.comparisonFunc,
            reductionMode: desc.reductionMode,
            addressModeU: desc.addressModeU ?? TextureAddressingMode.Wrap,
            addressModeV: desc.addressModeV ?? TextureAddressingMode.Wrap,
            addressModeW: desc.addressModeW ?? TextureAddressingMode.Wrap,
        };
        this.desc = full;

        if (full.reductionMode === TextureReductionMode.Min || full.reductionMode === TextureReductionMode.Max) {
            throw new UnsupportedFeatureError("TextureReductionMode.Min/Max", "WebGPU has no MIN/MAX filtering");
        }

        this.gpuSampler = device.gpuDevice.createSampler({
            magFilter: toGpuFilter(full.magFilter),
            minFilter: toGpuFilter(full.minFilter),
            mipmapFilter: full.mipFilter === TextureFilteringMode.Linear ? "linear" : "nearest",
            addressModeU: toGpuAddressMode(full.addressModeU),
            addressModeV: toGpuAddressMode(full.addressModeV),
            addressModeW: toGpuAddressMode(full.addressModeW),
            lodMinClamp: full.minLod,
            lodMaxClamp: full.maxLod,
            // Anisotropy > 1 requires all-linear filtering in WebGPU.
            maxAnisotropy:
                full.magFilter === TextureFilteringMode.Linear &&
                full.minFilter === TextureFilteringMode.Linear &&
                full.mipFilter === TextureFilteringMode.Linear
                    ? full.maxAnisotropy
                    : 1,
            compare:
                full.reductionMode === TextureReductionMode.Comparison && full.comparisonFunc !== undefined
                    ? toGpuCompareFunction(full.comparisonFunc)
                    : undefined,
        });
    }
}
