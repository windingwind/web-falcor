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
     * Builds two packed array textures (sRGB and linear colorspaces; each at
     * least one layer with a white fallback) plus per-texture info: uv scale
     * (textures smaller than the layer sit top-left), array selector and
     * layer index. Mirrors native per-slot sRGB semantics — normal/specular
     * maps must NOT be sRGB-decoded on sample.
     */
    build(device: Device): { array: Texture; arrayLinear: Texture; texInfo: Float32Array } {
        const makeArray = (sources: { source: TextureSource; id: number }[], srgb: boolean, name: string) => {
            const layers = Math.max(sources.length, 1);
            const width = Math.max(1, ...sources.map((s) => s.source.bitmap.width));
            const height = Math.max(1, ...sources.map((s) => s.source.bitmap.height));
            const array = new Texture(device, {
                type: ResourceType.Texture2D,
                width,
                height,
                arraySize: layers,
                mipLevels: 1,
                format: srgb ? ResourceFormat.RGBA8UnormSrgb : ResourceFormat.RGBA8Unorm,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget,
                name,
            });
            if (sources.length === 0) {
                device.gpuDevice.queue.writeTexture(
                    { texture: array.gpuTexture, origin: { x: 0, y: 0, z: 0 } },
                    new Uint8Array([255, 255, 255, 255]),
                    { bytesPerRow: 4 },
                    { width: 1, height: 1, depthOrArrayLayers: 1 },
                );
            }
            sources.forEach(({ source }, layer) => {
                const w = Math.min(source.bitmap.width, width);
                const h = Math.min(source.bitmap.height, height);
                device.gpuDevice.queue.copyExternalImageToTexture(
                    { source: source.bitmap },
                    { texture: array.gpuTexture, origin: { x: 0, y: 0, z: layer } },
                    { width: w, height: h, depthOrArrayLayers: 1 },
                );
            });
            return { array, width, height };
        };

        const srgbSources = this.sources.map((source, id) => ({ source, id })).filter((s) => s.source.srgb);
        const linearSources = this.sources.map((source, id) => ({ source, id })).filter((s) => !s.source.srgb);
        const srgbArr = makeArray(srgbSources, true, "TextureManager::materialTexturesArray");
        const linArr = makeArray(linearSources, false, "TextureManager::materialTexturesArrayLinear");

        // Per-texture info (indexed by textureID): uvScale.xy, arraySelector
        // (0 = sRGB, 1 = linear), layer index within its array.
        const texInfo = new Float32Array(Math.max(this.sources.length, 1) * 4);
        texInfo.set([1, 1, 0, 0]);
        srgbSources.forEach(({ source, id }, layer) => {
            texInfo[id * 4] = source.bitmap.width / srgbArr.width;
            texInfo[id * 4 + 1] = source.bitmap.height / srgbArr.height;
            texInfo[id * 4 + 2] = 0;
            texInfo[id * 4 + 3] = layer;
        });
        linearSources.forEach(({ source, id }, layer) => {
            texInfo[id * 4] = source.bitmap.width / linArr.width;
            texInfo[id * 4 + 1] = source.bitmap.height / linArr.height;
            texInfo[id * 4 + 2] = 1;
            texInfo[id * 4 + 3] = layer;
        });
        return { array: srgbArr.array, arrayLinear: linArr.array, texInfo };
    }
}
