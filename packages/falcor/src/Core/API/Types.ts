/**
 * Resource-related enums mirroring Falcor/Core/API/Types.h and Resource.h.
 */

/** Mirrors Falcor::ResourceBindFlags (subset representable in WebGPU). */
export enum ResourceBindFlags {
    None = 0,
    Vertex = 1 << 0,
    Index = 1 << 1,
    Constant = 1 << 2,
    StreamOutput = 1 << 3, // ❌ no WebGPU equivalent (transform feedback absent)
    ShaderResource = 1 << 4,
    UnorderedAccess = 1 << 5,
    RenderTarget = 1 << 6,
    DepthStencil = 1 << 7,
    IndirectArg = 1 << 8,
    Shared = 1 << 9, // ❌ cross-API sharing not available in browsers
    AccelerationStructure = 1 << 10, // 🟡 software BVH: lowered to storage buffer usage
}

/** Mirrors Falcor::MemoryType. */
export enum MemoryType {
    DeviceLocal,
    Upload,
    ReadBack,
}

/** Buffer usage flags for a given bind-flag combination. */
export function bindFlagsToBufferUsage(bindFlags: ResourceBindFlags, memoryType: MemoryType): GPUBufferUsageFlags {
    let usage: GPUBufferUsageFlags = 0;
    if (bindFlags & ResourceBindFlags.Vertex) usage |= GPUBufferUsage.VERTEX;
    if (bindFlags & ResourceBindFlags.Index) usage |= GPUBufferUsage.INDEX;
    if (bindFlags & ResourceBindFlags.Constant) usage |= GPUBufferUsage.UNIFORM;
    if (bindFlags & (ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess)) usage |= GPUBufferUsage.STORAGE;
    if (bindFlags & ResourceBindFlags.IndirectArg) usage |= GPUBufferUsage.INDIRECT;
    if (bindFlags & ResourceBindFlags.AccelerationStructure) usage |= GPUBufferUsage.STORAGE;

    switch (memoryType) {
        case MemoryType.DeviceLocal:
            usage |= GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
            break;
        case MemoryType.Upload:
            // Mappable-for-write staging; WebGPU forbids MAP_WRITE with most usages.
            usage |= GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC;
            break;
        case MemoryType.ReadBack:
            usage |= GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
            break;
    }
    return usage;
}

/** Texture usage flags for a given bind-flag combination. */
export function bindFlagsToTextureUsage(bindFlags: ResourceBindFlags): GPUTextureUsageFlags {
    let usage: GPUTextureUsageFlags = GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    if (bindFlags & ResourceBindFlags.ShaderResource) usage |= GPUTextureUsage.TEXTURE_BINDING;
    if (bindFlags & ResourceBindFlags.UnorderedAccess) usage |= GPUTextureUsage.STORAGE_BINDING;
    if (bindFlags & (ResourceBindFlags.RenderTarget | ResourceBindFlags.DepthStencil)) usage |= GPUTextureUsage.RENDER_ATTACHMENT;
    return usage;
}

/** Mirrors Falcor::Resource::Type. */
export enum ResourceType {
    Buffer,
    Texture1D,
    Texture2D,
    Texture3D,
    TextureCube,
    Texture2DMultisample,
}

/** Comparison functions (Falcor::ComparisonFunc). */
export enum ComparisonFunc {
    Disabled,
    Never,
    Always,
    Less,
    Equal,
    NotEqual,
    LessEqual,
    Greater,
    GreaterEqual,
}

export function toGpuCompareFunction(func: ComparisonFunc): GPUCompareFunction | undefined {
    switch (func) {
        case ComparisonFunc.Disabled: return undefined;
        case ComparisonFunc.Never: return "never";
        case ComparisonFunc.Always: return "always";
        case ComparisonFunc.Less: return "less";
        case ComparisonFunc.Equal: return "equal";
        case ComparisonFunc.NotEqual: return "not-equal";
        case ComparisonFunc.LessEqual: return "less-equal";
        case ComparisonFunc.Greater: return "greater";
        case ComparisonFunc.GreaterEqual: return "greater-equal";
    }
}
