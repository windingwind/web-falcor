/**
 * Copy/transfer command context mirroring Falcor/Core/API/CopyContext.h.
 *
 * Wraps a lazily-created GPUCommandEncoder. Falcor's resource barriers
 * (resourceBarrier/uavBarrier/textureBarrier) are no-ops kept for API parity —
 * WebGPU synchronizes automatically (DESIGN.md §9).
 */

import { ArgumentError } from "../Error.js";
import type { Device } from "./Device.js";
import type { Buffer } from "./Buffer.js";
import type { Texture } from "./Texture.js";
import { getFormatBytesPerBlock, isCompressedFormat } from "./Formats.js";
import { ResourceType } from "./Types.js";

export class CopyContext {
    protected encoder: GPUCommandEncoder | null = null;

    constructor(public readonly device: Device) {}

    /** Lazily begins command recording (mirrors LowLevelContextData's open command list). */
    getEncoder(): GPUCommandEncoder {
        this.encoder ??= this.device.gpuDevice.createCommandEncoder();
        return this.encoder;
    }

    /** Mirrors CopyContext::submit — finishes and submits recorded work. */
    submit(): void {
        if (this.encoder) {
            this.device.gpuDevice.queue.submit([this.encoder.finish()]);
            this.encoder = null;
        }
    }

    /** Mirrors submit(wait=true) as an async variant. */
    async submitAndWait(): Promise<void> {
        this.submit();
        await this.device.gpuDevice.queue.onSubmittedWorkDone();
    }

    // --- barriers: API parity no-ops ---
    resourceBarrier(_resource: unknown, _newState: unknown): boolean { return false; }
    uavBarrier(_resource: unknown): void {}
    textureBarrier(_texture: unknown, _newState: unknown): void {}

    /**
     * Mirrors CopyContext::updateBuffer. Uses a transient mapped staging buffer +
     * ordered copy on the encoder (Falcor's upload-heap pattern) so the update
     * respects command order relative to already-recorded work — queue.writeBuffer
     * would execute before this context's pending encoder submits.
     */
    updateBuffer(buffer: Buffer, data: ArrayBufferView | ArrayBuffer, offset = 0): void {
        const view =
            data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        if (offset + view.byteLength > buffer.size) throw new ArgumentError("updateBuffer out of range");
        const alignedSize = Math.ceil(view.byteLength / 4) * 4;
        const staging = this.device.gpuDevice.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint8Array(staging.getMappedRange()).set(view);
        staging.unmap();
        this.getEncoder().copyBufferToBuffer(staging, 0, buffer.gpuBuffer, offset, alignedSize);
    }

    /** Mirrors CopyContext::copyResource for buffers. */
    copyBuffer(dst: Buffer, src: Buffer): void {
        this.getEncoder().copyBufferToBuffer(src.gpuBuffer, 0, dst.gpuBuffer, 0, Math.min(src.size, dst.size));
    }

    /** Mirrors CopyContext::copyBufferRegion. */
    copyBufferRegion(dst: Buffer, dstOffset: number, src: Buffer, srcOffset: number, size: number): void {
        this.getEncoder().copyBufferToBuffer(src.gpuBuffer, srcOffset, dst.gpuBuffer, dstOffset, size);
    }

    /** Mirrors CopyContext::copyResource for textures (all subresources). */
    copyTexture(dst: Texture, src: Texture): void {
        for (let mip = 0; mip < Math.min(src.mipCount, dst.mipCount); mip++) {
            this.getEncoder().copyTextureToTexture(
                { texture: src.gpuTexture, mipLevel: mip },
                { texture: dst.gpuTexture, mipLevel: mip },
                {
                    width: Math.max(1, src.width >> mip),
                    height: Math.max(1, src.height >> mip),
                    depthOrArrayLayers: src.type === ResourceType.Texture3D ? Math.max(1, src.depth >> mip) : src.gpuTexture.depthOrArrayLayers,
                },
            );
        }
    }

    /** Mirrors CopyContext::copySubresource. */
    copySubresource(dst: Texture, dstMip: number, dstSlice: number, src: Texture, srcMip: number, srcSlice: number): void {
        this.getEncoder().copyTextureToTexture(
            { texture: src.gpuTexture, mipLevel: srcMip, origin: { x: 0, y: 0, z: srcSlice } },
            { texture: dst.gpuTexture, mipLevel: dstMip, origin: { x: 0, y: 0, z: dstSlice } },
            { width: Math.max(1, src.width >> srcMip), height: Math.max(1, src.height >> srcMip), depthOrArrayLayers: 1 },
        );
    }

    /**
     * Mirrors CopyContext::readBuffer (async divergence, DESIGN.md §9).
     * Flushes pending work, copies into a transient readback buffer, maps it.
     */
    async readBuffer(buffer: Buffer, offset = 0, size?: number): Promise<Uint8Array> {
        const byteSize = size ?? buffer.size - offset;
        if (offset + byteSize > buffer.size) throw new ArgumentError("readBuffer out of range");
        const alignedSize = Math.ceil(byteSize / 4) * 4;
        const staging = this.device.gpuDevice.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.getEncoder().copyBufferToBuffer(buffer.gpuBuffer, offset, staging, 0, alignedSize);
        this.submit();
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Uint8Array(staging.getMappedRange().slice(0, byteSize));
        staging.destroy();
        return data;
    }

    /** Mirrors CopyContext::readTextureSubresource (async). Returns tightly packed rows. */
    async readTextureSubresource(texture: Texture, mipLevel = 0, arraySlice = 0): Promise<Uint8Array> {
        const w = Math.max(1, texture.width >> mipLevel);
        const h = Math.max(1, texture.height >> mipLevel);
        const d = texture.type === ResourceType.Texture3D ? Math.max(1, texture.depth >> mipLevel) : 1;
        const bpb = getFormatBytesPerBlock(texture.format);
        const blockDim = isCompressedFormat(texture.format) ? 4 : 1;
        const blocksW = Math.ceil(w / blockDim);
        const blocksH = Math.ceil(h / blockDim);
        const tightBytesPerRow = blocksW * bpb;
        const alignedBytesPerRow = Math.ceil(tightBytesPerRow / 256) * 256; // WebGPU row alignment
        const staging = this.device.gpuDevice.createBuffer({
            size: alignedBytesPerRow * blocksH * d,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.getEncoder().copyTextureToBuffer(
            { texture: texture.gpuTexture, mipLevel, origin: { x: 0, y: 0, z: arraySlice } },
            { buffer: staging, bytesPerRow: alignedBytesPerRow, rowsPerImage: blocksH },
            { width: w, height: h, depthOrArrayLayers: d },
        );
        this.submit();
        await staging.mapAsync(GPUMapMode.READ);
        const mapped = new Uint8Array(staging.getMappedRange());
        const out = new Uint8Array(tightBytesPerRow * blocksH * d);
        for (let z = 0; z < d; z++) {
            for (let row = 0; row < blocksH; row++) {
                const srcOff = (z * blocksH + row) * alignedBytesPerRow;
                out.set(mapped.subarray(srcOff, srcOff + tightBytesPerRow), (z * blocksH + row) * tightBytesPerRow);
            }
        }
        staging.destroy();
        return out;
    }
}
