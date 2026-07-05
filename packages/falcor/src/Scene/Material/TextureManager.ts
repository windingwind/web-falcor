/**
 * Material texture manager mirroring Falcor/Utils/Image/TextureManager.h in
 * role, adapted to the §6.2 packing design: WGSL has no binding arrays, so all
 * material textures live in one texture_2d_array (layer == textureID).
 *
 * v1: single rgba8unorm-srgb array sized to the largest texture; smaller
 * textures are blit-resized to the layer size (uvScale stays 1 so wrap
 * addressing keeps working; slight resample divergence recorded). Mip chains
 * and size-class arrays are follow-ups.
 */

import type { Device } from "../../Core/API/Device.js";
import { Texture } from "../../Core/API/Texture.js";
import { ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";

export interface TextureSource {
    bitmap: ImageBitmap;
    srgb: boolean;
}

export class TextureManager {
    private sources: TextureSource[] = [];

    /** Registers a texture; returns its textureID (array layer). */
    addTexture(source: TextureSource): number {
        this.sources.push(source);
        return this.sources.length - 1;
    }

    get count(): number {
        return this.sources.length;
    }

    /**
     * Builds the packed array texture (at least one layer; a white fallback
     * when the scene has no textures) plus per-texture uv scales.
     */
    build(device: Device): { array: Texture; uvScale: Float32Array } {
        const layers = Math.max(this.sources.length, 1);
        const width = Math.max(1, ...this.sources.map((s) => s.bitmap.width));
        const height = Math.max(1, ...this.sources.map((s) => s.bitmap.height));

        const array = new Texture(device, {
            type: ResourceType.Texture2D,
            width,
            height,
            arraySize: layers,
            mipLevels: 1,
            format: ResourceFormat.RGBA8UnormSrgb,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget,
            name: "TextureManager::materialTexturesArray",
        });

        if (this.sources.length === 0) {
            device.gpuDevice.queue.writeTexture(
                { texture: array.gpuTexture, origin: { x: 0, y: 0, z: 0 } },
                new Uint8Array([255, 255, 255, 255]),
                { bytesPerRow: 4 },
                { width: 1, height: 1, depthOrArrayLayers: 1 },
            );
        }

        this.sources.forEach((source, layer) => {
            // copyExternalImageToTexture resizes nothing: sizes must match. For
            // mismatched sizes, draw-based resize would go here; v1 requires
            // uniform dimensions per scene or accepts top-left placement.
            const w = Math.min(source.bitmap.width, width);
            const h = Math.min(source.bitmap.height, height);
            device.gpuDevice.queue.copyExternalImageToTexture(
                { source: source.bitmap },
                { texture: array.gpuTexture, origin: { x: 0, y: 0, z: layer } },
                { width: w, height: h, depthOrArrayLayers: 1 },
            );
        });

        const uvScale = new Float32Array(layers * 2);
        this.sources.forEach((source, layer) => {
            uvScale[layer * 2] = source.bitmap.width / width;
            uvScale[layer * 2 + 1] = source.bitmap.height / height;
        });
        if (this.sources.length === 0) {
            uvScale[0] = 1;
            uvScale[1] = 1;
        }
        return { array, uvScale };
    }
}
