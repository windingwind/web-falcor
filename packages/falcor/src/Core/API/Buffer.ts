/**
 * GPU buffer mirroring Falcor/Core/API/Buffer.h.
 *
 * Creation goes through Device.createBuffer/createTypedBuffer/createStructuredBuffer
 * (mirroring Falcor 8's Device factory methods). UAV counters (structured-buffer
 * append/consume) are emulated with a separate 4-byte counter buffer — WebGPU has
 * no D3D-style UAV counters.
 */

import { Resource } from "./Resource.js";
import { MemoryType, ResourceBindFlags, ResourceType, bindFlagsToBufferUsage } from "./Types.js";
import { ResourceFormat } from "./Formats.js";
import { ArgumentError, RuntimeError } from "../Error.js";
import type { Device } from "./Device.js";

export interface BufferDesc {
    /** Size in bytes. */
    size: number;
    /** Struct stride in bytes for structured buffers, 0 otherwise. */
    structSize?: number;
    /** Element format for typed buffers, Unknown otherwise. */
    format?: ResourceFormat;
    bindFlags?: ResourceBindFlags;
    memoryType?: MemoryType;
    /** Create the hidden counter resource (structured buffers only). */
    createCounter?: boolean;
    name?: string;
}

export class Buffer extends Resource {
    readonly gpuBuffer: GPUBuffer;
    readonly size: number;
    readonly structSize: number;
    readonly format: ResourceFormat;
    readonly memoryType: MemoryType;
    /** Emulated UAV counter (see class docs). */
    readonly counterBuffer: Buffer | undefined;

    constructor(device: Device, desc: BufferDesc) {
        super(device, ResourceType.Buffer, desc.bindFlags ?? ResourceBindFlags.ShaderResource);
        this.size = desc.size;
        this.structSize = desc.structSize ?? 0;
        this.format = desc.format ?? ResourceFormat.Unknown;
        this.memoryType = desc.memoryType ?? MemoryType.DeviceLocal;
        this.name = desc.name ?? "";

        if (this.size === 0) throw new ArgumentError("Buffer size must be > 0");
        if (this.structSize > 0 && this.size % this.structSize !== 0) {
            throw new ArgumentError(`Buffer size (${this.size}) must be a multiple of structSize (${this.structSize})`);
        }
        if (this.memoryType === MemoryType.Upload && (this.bindFlags & ~(ResourceBindFlags.None)) !== 0 && this.bindFlags !== ResourceBindFlags.None) {
            // WebGPU restricts MAP_WRITE to COPY_SRC only; shader-visible upload buffers
            // are not representable (Falcor allows them on UMA). Documented divergence.
            throw new RuntimeError("Upload buffers cannot have bind flags in WebGPU; use DeviceLocal + setBlob()");
        }

        this.gpuBuffer = device.gpuDevice.createBuffer({
            label: this.name,
            // WebGPU requires 4-byte-aligned sizes for most operations; round up like Falcor aligns to CB requirements.
            size: Math.ceil(this.size / 4) * 4,
            usage: bindFlagsToBufferUsage(this.bindFlags, this.memoryType),
        });

        if (desc.createCounter) {
            this.counterBuffer = new Buffer(device, {
                size: 4,
                bindFlags: ResourceBindFlags.UnorderedAccess,
                name: `${this.name}:counter`,
            });
        }
    }

    get elementCount(): number {
        return this.structSize > 0 ? this.size / this.structSize : 0;
    }

    /**
     * Mirrors Buffer::setBlob: schedules an upload of data at byte offset,
     * ordered with respect to commands already recorded on the immediate context.
     */
    setBlob(data: ArrayBufferView | ArrayBuffer, offset = 0): void {
        this.device.renderContext.updateBuffer(this, data, offset);
    }

    /**
     * Mirrors Buffer::getBlob (async divergence, docs §9): reads back buffer
     * contents via the device's readback heap.
     */
    async getBlob(offset = 0, size?: number): Promise<Uint8Array> {
        return this.device.renderContext.readBuffer(this, offset, size);
    }

    override destroy(): void {
        this.counterBuffer?.destroy();
        this.gpuBuffer.destroy();
    }
}
