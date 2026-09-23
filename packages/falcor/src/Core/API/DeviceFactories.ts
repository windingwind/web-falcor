/**
 * Device resource-factory methods mirroring Falcor 8's Device::create* API.
 * Kept in a separate module and attached to Device to avoid circular imports
 * between Device and the resource classes.
 */

import { Device } from "./Device.js";
import { Buffer, type BufferDesc } from "./Buffer.js";
import { Texture, kMaxPossible } from "./Texture.js";
import { Sampler, type SamplerDesc } from "./Sampler.js";
import { Fence } from "./Fence.js";
import { GpuTimer } from "./GpuTimer.js";
import { MemoryType, ResourceBindFlags, ResourceType } from "./Types.js";
import { ResourceFormat, getFormatBindFlags, getFormatBytesPerBlock } from "./Formats.js";

declare module "./Device.js" {
    interface Device {
        /** Mirrors Device::createBuffer (raw buffer). */
        createBuffer(size: number, bindFlags?: ResourceBindFlags, memoryType?: MemoryType, initData?: ArrayBufferView): Buffer;
        /** Mirrors Device::createTypedBuffer. */
        createTypedBuffer(format: ResourceFormat, elementCount: number, bindFlags?: ResourceBindFlags, initData?: ArrayBufferView): Buffer;
        /** Mirrors Device::createStructuredBuffer. */
        createStructuredBuffer(
            structSize: number,
            elementCount: number,
            bindFlags?: ResourceBindFlags,
            initData?: ArrayBufferView,
            createCounter?: boolean,
        ): Buffer;
        createBufferFromDesc(desc: BufferDesc): Buffer;
        /** Mirrors Device::getFormatBindFlags: bind flags supported by a texture format. */
        getFormatBindFlags(format: ResourceFormat): ResourceBindFlags;
        /** Mirrors Device::createTexture1D/2D/3D/Cube. */
        createTexture2D(
            width: number,
            height: number,
            format: ResourceFormat,
            arraySize?: number,
            mipLevels?: number,
            initData?: ArrayBufferView,
            bindFlags?: ResourceBindFlags,
        ): Texture;
        /** Mirrors Device::createTexture2DMS. §9: WebGPU multisamples at 1 or 4 samples only. */
        createTexture2DMS(width: number, height: number, format: ResourceFormat, sampleCount: number, arraySize?: number, bindFlags?: ResourceBindFlags): Texture;
        createTexture3D(width: number, height: number, depth: number, format: ResourceFormat, mipLevels?: number, bindFlags?: ResourceBindFlags): Texture;
        createTextureCube(width: number, height: number, format: ResourceFormat, arraySize?: number, mipLevels?: number, bindFlags?: ResourceBindFlags): Texture;
        /** Mirrors Device::createSampler. */
        createSampler(desc?: SamplerDesc): Sampler;
        /** Mirrors Device::createFence. */
        createFence(): Fence;
        createGpuTimer(): GpuTimer;
    }
}

const kDefaultBufferFlags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;

Device.prototype.createBuffer = function (size, bindFlags = kDefaultBufferFlags, memoryType = MemoryType.DeviceLocal, initData) {
    const buffer = new Buffer(this, { size, bindFlags, memoryType });
    if (initData) buffer.setBlob(initData);
    return buffer;
};

Device.prototype.createTypedBuffer = function (format, elementCount, bindFlags = kDefaultBufferFlags, initData) {
    const buffer = new Buffer(this, { size: elementCount * getFormatBytesPerBlock(format), format, bindFlags });
    if (initData) buffer.setBlob(initData);
    return buffer;
};

Device.prototype.createStructuredBuffer = function (structSize, elementCount, bindFlags = kDefaultBufferFlags, initData, createCounter = false) {
    const buffer = new Buffer(this, { size: structSize * elementCount, structSize, bindFlags, createCounter });
    if (initData) buffer.setBlob(initData);
    return buffer;
};

Device.prototype.createBufferFromDesc = function (desc) {
    return new Buffer(this, desc);
};

Device.prototype.getFormatBindFlags = function (format) {
    return getFormatBindFlags(format, (f) => this.hasFeature(f as GPUFeatureName));
};

Device.prototype.createTexture2D = function (width, height, format, arraySize = 1, mipLevels = kMaxPossible, initData, bindFlags = ResourceBindFlags.ShaderResource) {
    const texture = new Texture(this, { type: ResourceType.Texture2D, width, height, format, arraySize, mipLevels, bindFlags });
    if (initData) texture.setSubresourceBlob(0, 0, initData);
    return texture;
};

Device.prototype.createTexture2DMS = function (width, height, format, sampleCount, arraySize = 1, bindFlags = ResourceBindFlags.ShaderResource) {
    return new Texture(this, { type: ResourceType.Texture2DMultisample, width, height, format, sampleCount, arraySize, mipLevels: 1, bindFlags });
};

Device.prototype.createTexture3D = function (width, height, depth, format, mipLevels = kMaxPossible, bindFlags = ResourceBindFlags.ShaderResource) {
    return new Texture(this, { type: ResourceType.Texture3D, width, height, depth, format, mipLevels, bindFlags });
};

Device.prototype.createTextureCube = function (width, height, format, arraySize = 1, mipLevels = kMaxPossible, bindFlags = ResourceBindFlags.ShaderResource) {
    return new Texture(this, { type: ResourceType.TextureCube, width, height, format, arraySize, mipLevels, bindFlags });
};

Device.prototype.createSampler = function (desc = {}) {
    return new Sampler(this, desc);
};

Device.prototype.createFence = function () {
    return new Fence(this.gpuDevice.queue);
};

Device.prototype.createGpuTimer = function () {
    return new GpuTimer(this);
};
