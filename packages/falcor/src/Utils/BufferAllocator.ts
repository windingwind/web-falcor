/**
 * Mirrors Falcor/Utils/BufferAllocator: a CPU byte buffer with aligned,
 * cache-line-aware sub-allocation, uploaded to a GPU buffer on demand with
 * dirty-range tracking. Typed pushBack/emplaceBack take the bytes of the value.
 */

import type { Device } from "../Core/API/Device.js";
import type { Buffer } from "../Core/API/Buffer.js";
import { MemoryType, ResourceBindFlags } from "../Core/API/Types.js";
import { RuntimeError } from "../Core/Error.js";

const isPowerOf2 = (v: number) => v > 0 && (v & (v - 1)) === 0;
const bytesOf = (data: ArrayBufferView | ArrayBuffer) => (data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength));

export class BufferAllocator {
    private data = new Uint8Array(0);
    private size = 0;
    private dirty = { start: 0, end: 0 };
    private gpuBuffer: Buffer | null = null;

    constructor(
        private readonly alignment: number,
        private readonly elementSize: number,
        private readonly cacheLineSize = 128,
        private readonly bindFlags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
    ) {
        if (!(alignment === 0 || isPowerOf2(alignment))) throw new RuntimeError("Alignment must be a power of two.");
        if (!(cacheLineSize === 0 || isPowerOf2(cacheLineSize))) throw new RuntimeError("Cache line size must be a power of two.");
        if (!(cacheLineSize === 0 || alignment <= cacheLineSize)) throw new RuntimeError("Alignment must be smaller or equal to the cache line size.");
        if (elementSize > 0 && alignment > 0 && Math.max(alignment, elementSize) % Math.min(alignment, elementSize) !== 0)
            throw new RuntimeError("Alignment and element size needs to be integer multiples.");
    }

    /** Mirrors allocate(byteSize): returns the byte offset. */
    allocate(byteSize: number): number {
        this.computeAndAllocatePadding(byteSize);
        return this.allocInternal(byteSize);
    }

    /** Mirrors pushBack / emplaceBack: appends the value's bytes and returns their offset. */
    pushBack(value: ArrayBufferView | ArrayBuffer): number {
        const bytes = bytesOf(value);
        this.computeAndAllocatePadding(bytes.length);
        const offset = this.allocInternal(bytes.length);
        this.data.set(bytes, offset);
        this.markAsDirty(offset, offset + bytes.length);
        return offset;
    }

    emplaceBack(value: ArrayBufferView | ArrayBuffer): number {
        return this.pushBack(value);
    }

    /** Mirrors setBlob / set<T>. */
    setBlob(value: ArrayBufferView | ArrayBuffer, byteOffset: number): void {
        const bytes = bytesOf(value);
        if (byteOffset + bytes.length > this.size) throw new RuntimeError("Memory region is out of range.");
        this.data.set(bytes, byteOffset);
        this.markAsDirty(byteOffset, byteOffset + bytes.length);
    }

    set(byteOffset: number, value: ArrayBufferView | ArrayBuffer): void {
        this.setBlob(value, byteOffset);
    }

    /** Mirrors modified(): marks bytes written through getStartPointer() for upload. */
    modified(byteOffset: number, byteSize: number): void {
        if (byteOffset + byteSize > this.size) throw new RuntimeError("Memory region is out of range.");
        this.markAsDirty(byteOffset, byteOffset + byteSize);
    }

    /** Mirrors getStartPointer(): a view of the CPU copy (valid until the next allocation). */
    getStartPointer(): Uint8Array {
        return this.data.subarray(0, this.size);
    }

    getSize(): number {
        return this.size;
    }

    clear(): void {
        this.size = 0;
        this.dirty = { start: 0, end: 0 };
    }

    /** Mirrors getGPUBuffer: (re)creates the buffer when it is too small and uploads the dirty range. */
    getGPUBuffer(device: Device): Buffer | null {
        if (this.size === 0) return null;
        const elemSize = this.elementSize > 0 ? this.elementSize : 4;
        const bufSize = Math.ceil(this.size / elemSize) * elemSize;
        if (!this.gpuBuffer || this.gpuBuffer.size < bufSize) {
            this.gpuBuffer =
                this.elementSize > 0
                    ? device.createStructuredBuffer(this.elementSize, bufSize / this.elementSize, this.bindFlags)
                    : device.createBuffer(bufSize, this.bindFlags, MemoryType.DeviceLocal);
            this.dirty = { start: 0, end: this.size };
        }
        if (this.dirty.start < this.dirty.end) {
            this.gpuBuffer.setBlob(this.data.subarray(this.dirty.start, this.dirty.end), this.dirty.start);
            this.dirty = { start: this.size, end: 0 };
        }
        return this.gpuBuffer;
    }

    private computeAndAllocatePadding(byteSize: number): void {
        let offset = this.size;
        if (this.alignment > 0 && offset % this.alignment > 0) offset += this.alignment - (offset % this.alignment);
        if (this.cacheLineSize > 0) {
            const lineOffset = offset % this.cacheLineSize;
            if (byteSize <= this.cacheLineSize && lineOffset + byteSize > this.cacheLineSize) offset += this.cacheLineSize - lineOffset;
        }
        if (offset > this.size) this.allocInternal(offset - this.size);
    }

    private allocInternal(byteSize: number): number {
        const offset = this.size;
        const needed = offset + byteSize;
        if (needed > this.data.length) {
            const grown = new Uint8Array(Math.max(needed, this.data.length * 2, 64));
            grown.set(this.data.subarray(0, this.size));
            this.data = grown;
        }
        this.data.fill(0, offset, needed);
        this.size = needed;
        return offset;
    }

    private markAsDirty(start: number, end: number): void {
        this.dirty.start = Math.min(this.dirty.start, start);
        this.dirty.end = Math.max(this.dirty.end, end);
    }
}
