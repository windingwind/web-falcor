/**
 * Resource format definitions mirroring Falcor/Core/API/Formats.h.
 *
 * ResourceFormat keeps Falcor's enum names; toGpuTextureFormat() maps to WebGPU
 * GPUTextureFormat strings. Formats without a WebGPU equivalent map to `undefined`
 * and are documented in the parity matrix (DESIGN.md §Formats).
 */

export enum ResourceFormat {
    Unknown,

    R8Unorm,
    R8Snorm,
    R16Unorm,
    R16Snorm,
    RG8Unorm,
    RG8Snorm,
    RG16Unorm,
    RG16Snorm,
    RGBA8Unorm,
    RGBA8Snorm,
    RGBA8UnormSrgb,
    R16Float,
    RG16Float,
    RGBA16Float,
    R32Float,
    RG32Float,
    RGB32Float,
    RGBA32Float,
    R11G11B10Float,
    RGB10A2Unorm,
    RGB10A2Uint,
    R8Int,
    R8Uint,
    R16Int,
    R16Uint,
    R32Int,
    R32Uint,
    RG8Int,
    RG8Uint,
    RG16Int,
    RG16Uint,
    RG32Int,
    RG32Uint,
    RGBA8Int,
    RGBA8Uint,
    RGBA16Int,
    RGBA16Uint,
    RGBA32Int,
    RGBA32Uint,
    BGRA8Unorm,
    BGRA8UnormSrgb,
    RGB9E5Float,

    // Depth-stencil
    D32Float,
    D32FloatS8Uint,
    D16Unorm,
    D24UnormS8Uint,

    // Compressed (BC1-7 ↔ WebGPU 'texture-compression-bc' feature)
    BC1Unorm,
    BC1UnormSrgb,
    BC2Unorm,
    BC2UnormSrgb,
    BC3Unorm,
    BC3UnormSrgb,
    BC4Unorm,
    BC4Snorm,
    BC5Unorm,
    BC5Snorm,
    BC6HS16,
    BC6HU16,
    BC7Unorm,
    BC7UnormSrgb,
}

const kGpuFormatMap: Partial<Record<ResourceFormat, GPUTextureFormat>> = {
    [ResourceFormat.R8Unorm]: "r8unorm",
    [ResourceFormat.R8Snorm]: "r8snorm",
    [ResourceFormat.R16Unorm]: "r16unorm",
    [ResourceFormat.R16Snorm]: "r16snorm",
    [ResourceFormat.RG8Unorm]: "rg8unorm",
    [ResourceFormat.RG8Snorm]: "rg8snorm",
    [ResourceFormat.RG16Unorm]: "rg16unorm",
    [ResourceFormat.RG16Snorm]: "rg16snorm",
    [ResourceFormat.RGBA8Unorm]: "rgba8unorm",
    [ResourceFormat.RGBA8Snorm]: "rgba8snorm",
    [ResourceFormat.RGBA8UnormSrgb]: "rgba8unorm-srgb",
    [ResourceFormat.R16Float]: "r16float",
    [ResourceFormat.RG16Float]: "rg16float",
    [ResourceFormat.RGBA16Float]: "rgba16float",
    [ResourceFormat.R32Float]: "r32float",
    [ResourceFormat.RG32Float]: "rg32float",
    // RGB32Float: no WebGPU texture format (buffer-only); see DESIGN.md
    [ResourceFormat.RGBA32Float]: "rgba32float",
    [ResourceFormat.R11G11B10Float]: "rg11b10ufloat",
    [ResourceFormat.RGB10A2Unorm]: "rgb10a2unorm",
    [ResourceFormat.RGB10A2Uint]: "rgb10a2uint",
    [ResourceFormat.R8Int]: "r8sint",
    [ResourceFormat.R8Uint]: "r8uint",
    [ResourceFormat.R16Int]: "r16sint",
    [ResourceFormat.R16Uint]: "r16uint",
    [ResourceFormat.R32Int]: "r32sint",
    [ResourceFormat.R32Uint]: "r32uint",
    [ResourceFormat.RG8Int]: "rg8sint",
    [ResourceFormat.RG8Uint]: "rg8uint",
    [ResourceFormat.RG16Int]: "rg16sint",
    [ResourceFormat.RG16Uint]: "rg16uint",
    [ResourceFormat.RG32Int]: "rg32sint",
    [ResourceFormat.RG32Uint]: "rg32uint",
    [ResourceFormat.RGBA8Int]: "rgba8sint",
    [ResourceFormat.RGBA8Uint]: "rgba8uint",
    [ResourceFormat.RGBA16Int]: "rgba16sint",
    [ResourceFormat.RGBA16Uint]: "rgba16uint",
    [ResourceFormat.RGBA32Int]: "rgba32sint",
    [ResourceFormat.RGBA32Uint]: "rgba32uint",
    [ResourceFormat.BGRA8Unorm]: "bgra8unorm",
    [ResourceFormat.BGRA8UnormSrgb]: "bgra8unorm-srgb",
    [ResourceFormat.RGB9E5Float]: "rgb9e5ufloat",
    [ResourceFormat.D32Float]: "depth32float",
    [ResourceFormat.D32FloatS8Uint]: "depth32float-stencil8",
    [ResourceFormat.D16Unorm]: "depth16unorm",
    [ResourceFormat.D24UnormS8Uint]: "depth24plus-stencil8",
    [ResourceFormat.BC1Unorm]: "bc1-rgba-unorm",
    [ResourceFormat.BC1UnormSrgb]: "bc1-rgba-unorm-srgb",
    [ResourceFormat.BC2Unorm]: "bc2-rgba-unorm",
    [ResourceFormat.BC2UnormSrgb]: "bc2-rgba-unorm-srgb",
    [ResourceFormat.BC3Unorm]: "bc3-rgba-unorm",
    [ResourceFormat.BC3UnormSrgb]: "bc3-rgba-unorm-srgb",
    [ResourceFormat.BC4Unorm]: "bc4-r-unorm",
    [ResourceFormat.BC4Snorm]: "bc4-r-snorm",
    [ResourceFormat.BC5Unorm]: "bc5-rg-unorm",
    [ResourceFormat.BC5Snorm]: "bc5-rg-snorm",
    [ResourceFormat.BC6HS16]: "bc6h-rgb-float",
    [ResourceFormat.BC6HU16]: "bc6h-rgb-ufloat",
    [ResourceFormat.BC7Unorm]: "bc7-rgba-unorm",
    [ResourceFormat.BC7UnormSrgb]: "bc7-rgba-unorm-srgb",
};

/** Returns the WebGPU texture format for a ResourceFormat, or undefined if not representable. */
export function toGpuTextureFormat(format: ResourceFormat): GPUTextureFormat | undefined {
    return kGpuFormatMap[format];
}

export function isDepthFormat(format: ResourceFormat): boolean {
    switch (format) {
        case ResourceFormat.D32Float:
        case ResourceFormat.D32FloatS8Uint:
        case ResourceFormat.D16Unorm:
        case ResourceFormat.D24UnormS8Uint:
            return true;
        default:
            return false;
    }
}

export function isCompressedFormat(format: ResourceFormat): boolean {
    return format >= ResourceFormat.BC1Unorm && format <= ResourceFormat.BC7UnormSrgb;
}
