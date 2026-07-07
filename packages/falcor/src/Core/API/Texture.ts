/**
 * GPU texture mirroring Falcor/Core/API/Texture.h.
 * Creation via Device.createTexture1D/2D/3D/Cube (Falcor 8 factory methods).
 */

import { Resource } from "./Resource.js";
import { ResourceBindFlags, ResourceType, bindFlagsToTextureUsage } from "./Types.js";
import { ResourceFormat, getFormatBytesPerBlock, isCompressedFormat, isDepthFormat, toGpuTextureFormat } from "./Formats.js";
import { ArgumentError, RuntimeError } from "../Error.js";
import type { Device } from "./Device.js";

/** Mirrors Falcor's kMaxPossible constant for full mip chains. */
export const kMaxPossible = 0xffffffff;

export interface TextureDesc {
    type: ResourceType;
    width: number;
    height?: number;
    depth?: number;
    arraySize?: number;
    mipLevels?: number;
    sampleCount?: number;
    format: ResourceFormat;
    bindFlags?: ResourceBindFlags;
    name?: string;
}

export class Texture extends Resource {
    readonly gpuTexture: GPUTexture;
    readonly gpuFormat: GPUTextureFormat;
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    readonly arraySize: number;
    readonly mipCount: number;
    readonly sampleCount: number;
    readonly format: ResourceFormat;

    constructor(device: Device, desc: TextureDesc) {
        super(device, desc.type, desc.bindFlags ?? ResourceBindFlags.ShaderResource);
        this.width = desc.width;
        this.height = desc.height ?? 1;
        this.depth = desc.depth ?? 1;
        this.arraySize = desc.arraySize ?? 1;
        this.sampleCount = desc.sampleCount ?? 1;
        this.format = desc.format;
        this.name = desc.name ?? "";

        if (this.width === 0 || this.height === 0 || this.depth === 0) throw new ArgumentError("Texture dimensions must be > 0");

        const gpuFormat = toGpuTextureFormat(desc.format);
        if (!gpuFormat) {
            throw new RuntimeError(
                `ResourceFormat ${ResourceFormat[desc.format]} has no WebGPU texture format (see DESIGN.md §Formats)`,
            );
        }
        this.gpuFormat = gpuFormat;

        const maxMips = 1 + Math.floor(Math.log2(Math.max(this.width, this.height, this.depth)));
        const requested = desc.mipLevels ?? kMaxPossible;
        this.mipCount = this.sampleCount > 1 ? 1 : Math.min(requested, maxMips);

        const is3D = desc.type === ResourceType.Texture3D;
        const isCube = desc.type === ResourceType.TextureCube;
        this.gpuTexture = device.gpuDevice.createTexture({
            label: this.name,
            size: {
                width: this.width,
                height: this.height,
                depthOrArrayLayers: is3D ? this.depth : this.arraySize * (isCube ? 6 : 1),
            },
            mipLevelCount: this.mipCount,
            sampleCount: this.sampleCount,
            dimension: desc.type === ResourceType.Texture1D ? "1d" : is3D ? "3d" : "2d",
            format: gpuFormat,
            usage: bindFlagsToTextureUsage(this.bindFlags),
        });
    }

    /** Creates a view; cached by descriptor key (mirrors Falcor's view caching). */
    private viewCache = new Map<string, GPUTextureView>();

    getView(mostDetailedMip = 0, mipCount?: number, firstArraySlice = 0, arraySize?: number, dimension?: GPUTextureViewDimension): GPUTextureView {
        const mips = mipCount ?? this.mipCount - mostDetailedMip;
        const layers = arraySize ?? (this.type === ResourceType.Texture3D ? 1 : this.gpuTexture.depthOrArrayLayers - firstArraySlice);
        const dim =
            dimension ??
            (this.type === ResourceType.Texture1D ? "1d"
            : this.type === ResourceType.Texture3D ? "3d"
            : this.type === ResourceType.TextureCube ? "cube"
            : layers > 1 ? "2d-array" : "2d");
        const key = `${mostDetailedMip}/${mips}/${firstArraySlice}/${layers}/${dim}`;
        let view = this.viewCache.get(key);
        if (!view) {
            view = this.gpuTexture.createView({
                baseMipLevel: mostDetailedMip,
                mipLevelCount: mips,
                baseArrayLayer: this.type === ResourceType.Texture3D ? 0 : firstArraySlice,
                arrayLayerCount: this.type === ResourceType.Texture3D ? undefined : layers,
                dimension: dim,
            });
            this.viewCache.set(key, view);
        }
        return view;
    }

    /** Mirrors Texture::getSRV/getRTV/getDSV/getUAV at the view level. */
    getSRV(mostDetailedMip = 0, mipCount?: number, firstArraySlice = 0, arraySize?: number): GPUTextureView {
        return this.getView(mostDetailedMip, mipCount, firstArraySlice, arraySize);
    }
    getUAV(mipLevel = 0, firstArraySlice = 0, arraySize?: number): GPUTextureView {
        return this.getView(mipLevel, 1, firstArraySlice, arraySize);
    }
    getRTV(mipLevel = 0, firstArraySlice = 0, arraySize = 1): GPUTextureView {
        return this.getView(mipLevel, 1, firstArraySlice, arraySize);
    }
    getDSV(mipLevel = 0, firstArraySlice = 0, arraySize = 1): GPUTextureView {
        if (!isDepthFormat(this.format)) throw new RuntimeError("getDSV on non-depth texture");
        return this.getView(mipLevel, 1, firstArraySlice, arraySize);
    }

    /** Mirrors Texture::generateMips: box-filter downsample chain via blits
     *  (linear sampling at destination texel centers == 2x2 average). */
    generateMips(ctx: { blit(src: Texture, dst: Texture, filter?: GPUFilterMode, srcMip?: number, dstMip?: number): void }): void {
        for (let mip = 1; mip < this.mipCount; mip++) {
            ctx.blit(this, this, "linear", mip - 1, mip);
        }
    }

    /** Mirrors Texture::setSubresourceBlob: uploads one mip of one array slice. */
    setSubresourceBlob(mipLevel: number, arraySlice: number, data: ArrayBufferView | ArrayBuffer): void {
        const w = Math.max(1, this.width >> mipLevel);
        const h = Math.max(1, this.height >> mipLevel);
        const d = this.type === ResourceType.Texture3D ? Math.max(1, this.depth >> mipLevel) : 1;
        const bpb = getFormatBytesPerBlock(this.format);
        const blockDim = isCompressedFormat(this.format) ? 4 : 1;
        const blocksW = Math.ceil(w / blockDim);
        const blocksH = Math.ceil(h / blockDim);
        const view =
            data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        // writeTexture to a compressed format requires the copy extent to be a
        // multiple of the block size (4) — the sub-4 tail mips (2x2, 1x1) round
        // up to one block; the texture's physical mip is block-padded to match.
        this.device.gpuDevice.queue.writeTexture(
            { texture: this.gpuTexture, mipLevel, origin: { x: 0, y: 0, z: arraySlice } },
            view as Uint8Array<ArrayBuffer>,
            { bytesPerRow: blocksW * bpb, rowsPerImage: blocksH },
            { width: blocksW * blockDim, height: blocksH * blockDim, depthOrArrayLayers: d },
        );
    }

    override destroy(): void {
        this.gpuTexture.destroy();
    }
}
