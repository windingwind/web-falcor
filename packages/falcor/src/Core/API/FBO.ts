/**
 * Framebuffer object mirroring Falcor/Core/API/FBO.h.
 * Lowered to GPURenderPassDescriptor attachments at draw time.
 */

import type { Texture } from "./Texture.js";
import { isDepthFormat } from "./Formats.js";
import { ArgumentError } from "../Error.js";
import type { Device } from "./Device.js";
import { ResourceBindFlags, ResourceType } from "./Types.js";
import { ResourceFormat } from "./Formats.js";
import { Texture as TextureClass } from "./Texture.js";

interface Attachment {
    texture: Texture | null;
    mipLevel: number;
    firstArraySlice: number;
    arraySize: number;
}

const kEmpty: Attachment = { texture: null, mipLevel: 0, firstArraySlice: 0, arraySize: 1 };

export class Fbo {
    private colorAttachments: Attachment[] = [];
    private depthAttachment: Attachment = { ...kEmpty };

    /** Mirrors Fbo::attachColorTarget. */
    attachColorTarget(texture: Texture | null, rtIndex: number, mipLevel = 0, firstArraySlice = 0, arraySize = 1): this {
        while (this.colorAttachments.length <= rtIndex) this.colorAttachments.push({ ...kEmpty });
        this.colorAttachments[rtIndex] = { texture, mipLevel, firstArraySlice, arraySize };
        return this;
    }

    /** Mirrors Fbo::attachDepthStencilTarget. */
    attachDepthStencilTarget(texture: Texture | null, mipLevel = 0, firstArraySlice = 0, arraySize = 1): this {
        if (texture && !isDepthFormat(texture.format)) throw new ArgumentError("Depth attachment must have a depth format");
        this.depthAttachment = { texture, mipLevel, firstArraySlice, arraySize };
        return this;
    }

    getColorTexture(index: number): Texture | null {
        return this.colorAttachments[index]?.texture ?? null;
    }
    getDepthStencilTexture(): Texture | null {
        return this.depthAttachment.texture;
    }
    getColorAttachmentCount(): number {
        return this.colorAttachments.length;
    }

    get width(): number {
        const t = this.anyTexture();
        return t ? Math.max(1, t.texture!.width >> t.mipLevel) : 0;
    }
    get height(): number {
        const t = this.anyTexture();
        return t ? Math.max(1, t.texture!.height >> t.mipLevel) : 0;
    }
    get sampleCount(): number {
        return this.anyTexture()?.texture!.sampleCount ?? 1;
    }

    private anyTexture(): Attachment | null {
        for (const a of this.colorAttachments) if (a.texture) return a;
        return this.depthAttachment.texture ? this.depthAttachment : null;
    }

    /** Color formats for pipeline creation (null slots preserved). */
    getGpuColorFormats(): (GPUTextureFormat | null)[] {
        return this.colorAttachments.map((a) => a.texture?.gpuFormat ?? null);
    }
    getGpuDepthFormat(): GPUTextureFormat | undefined {
        return this.depthAttachment.texture?.gpuFormat;
    }

    /** Builds render-pass attachments. loadOp "load" preserves contents (clears are explicit like Falcor). */
    getGpuRenderPassDescriptor(): GPURenderPassDescriptor {
        const colorAttachments: (GPURenderPassColorAttachment | null)[] = this.colorAttachments.map((a) =>
            a.texture
                ? {
                      view: a.texture.getRTV(a.mipLevel, a.firstArraySlice, a.arraySize),
                      loadOp: "load" as const,
                      storeOp: "store" as const,
                  }
                : null,
        );
        const desc: GPURenderPassDescriptor = { colorAttachments };
        if (this.depthAttachment.texture) {
            const hasStencil = this.depthAttachment.texture.gpuFormat.includes("stencil");
            desc.depthStencilAttachment = {
                view: this.depthAttachment.texture.getDSV(this.depthAttachment.mipLevel, this.depthAttachment.firstArraySlice, this.depthAttachment.arraySize),
                depthLoadOp: "load",
                depthStoreOp: "store",
                ...(hasStencil ? { stencilLoadOp: "load" as const, stencilStoreOp: "store" as const } : {}),
            };
        }
        return desc;
    }

    /** Mirrors Fbo::create2D convenience factory. */
    static create2D(device: Device, width: number, height: number, colorFormat: ResourceFormat, depthFormat?: ResourceFormat): Fbo {
        const fbo = new Fbo();
        const color = new TextureClass(device, {
            type: ResourceType.Texture2D,
            width,
            height,
            format: colorFormat,
            mipLevels: 1,
            bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        });
        fbo.attachColorTarget(color, 0);
        if (depthFormat !== undefined) {
            const depth = new TextureClass(device, {
                type: ResourceType.Texture2D,
                width,
                height,
                format: depthFormat,
                mipLevels: 1,
                bindFlags: ResourceBindFlags.DepthStencil,
            });
            fbo.attachDepthStencilTarget(depth);
        }
        return fbo;
    }
}
