/**
 * Material texture manager mirroring Falcor/Utils/Image/TextureManager.h in
 * role, adapted to WGSL, which has no binding arrays (§6.2): material textures
 * live in up to kMaxTextureBuckets texture_2d_arrays, one per GPU format and
 * size ("buckets"), and a per-texture info row selects bucket and layer.
 *
 * BC textures (DDS) keep their compressed full-resolution mip chains, as native
 * uploads them; buckets are keyed by exact format, block-padded size and mip
 * count, so no tiling is needed. Decoded images share RGBA8 (sRGB or linear)
 * buckets per power-of-two size class; a smaller texture tiles its layer, so
 * bilinear filtering at its edges wraps like repeat addressing and, when the
 * tiles fill the layer exactly (power-of-two sizes), the layer's mips are the
 * texture's own mips; samplers take LODs from the texture's own size
 * (texInfo uvScale). Buckets split at the array-layer limit; beyond the binding
 * budget, compressed textures fall back to their decoded image.
 */

import type { Device } from "../../Core/API/Device.js";
import { Texture } from "../../Core/API/Texture.js";
import { ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import { RuntimeError } from "../../Core/Error.js";
import { AssetCategory, AssetResolver } from "../../Core/AssetResolver.js";
import { Logger } from "../../Utils/Logger.js";

/** Decodes one image file; the default handles what createImageBitmap reads. */
export type ImageDecoder = (bytes: Uint8Array, blob: Blob, url: string) => Promise<ImageBitmap>;
const decodeWithBrowser: ImageDecoder = (_bytes, blob) => createImageBitmap(blob, { colorSpaceConversion: "none" });

export interface TextureSource {
    /** Decoded image (CPU consumers: alpha analysis, readback; the GPU fallback for compressed data). */
    bitmap: ImageBitmap;
    srgb: boolean;
    /** Original compressed bytes when available (lossless SceneCache path). */
    bytes?: Uint8Array;
    /** Block-compressed full-resolution mip chain, uploaded as is (DDS). */
    compressed?: { format: ResourceFormat; width: number; height: number; levels: Uint8Array[] };
    /** The source file was a DDS (the SceneCache stores `bytes` as DDS, not PNG). */
    dds?: boolean;
    /** Mip levels 1.. of a `<MIP>` file set (bitmap is mip 0), uploaded as is. */
    mips?: ImageBitmap[];
}

/** TextureAnalyzer::Result subset: constancy per channel mask (R 1, G 2, B 4, A 8) and the value. */
export interface TextureAnalysis {
    value: [number, number, number, number];
    minAlpha: number;
    maxAlpha: number;
    hasAlpha: boolean;
    isConstant(mask: number): boolean;
}

/** Texture arrays bound to the material system (the shader declares this many). */
export const kMaxTextureBuckets = 16;

/** One material texture array and whether its mips must be generated. */
export interface TextureBucket {
    texture: Texture;
    generateMips: boolean;
}

export class TextureManager {
    private sources: TextureSource[] = [];
    private analyses = new Map<number, TextureAnalysis>();

    /** Registers a texture; returns its textureID (array layer). */
    addTexture(source: TextureSource): number {
        this.sources.push(source);
        return this.sources.length - 1;
    }

    /**
     * Mirrors TextureManager::loadTexture: a `<MIP>` path loads mip0, mip1, ... until a file is
     * missing, as one texture with those levels. Returns the texture ID, or undefined (with native's
     * warning) when nothing is found. §9: single images always get generated mips (texture-LOD).
     */
    async loadTexture(path: string, _generateMipLevels: boolean, loadAsSRGB: boolean, resolver = AssetResolver.getDefaultResolver(), baseUrl = "", decode = decodeWithBrowser): Promise<number | undefined> {
        const load = async (p: string): Promise<ImageBitmap | null> => {
            const url = (await resolver.resolvePath(p, AssetCategory.Any)) || (baseUrl ? `${baseUrl}/${p}` : p);
            const res = await fetch(url).catch(() => null);
            if (!res?.ok) return null;
            const blob = await res.blob();
            return decode(new Uint8Array(await blob.arrayBuffer()), blob, url);
        };
        const levels: ImageBitmap[] = [];
        if (path.includes("<MIP>")) {
            for (let bitmap; (bitmap = await load(path.replace("<MIP>", `mip${levels.length}`))); ) levels.push(bitmap);
        } else {
            const bitmap = await load(path);
            if (bitmap) levels.push(bitmap);
        }
        if (levels.length === 0) {
            Logger.warning(`Can't find texture file '${path}'.`);
            return undefined;
        }
        return this.addTexture({ bitmap: levels[0]!, srgb: loadAsSRGB, mips: levels.length > 1 ? levels.slice(1) : undefined });
    }

    get count(): number {
        return this.sources.length;
    }

    /**
     * Reads a texture back as linear float RGB (sRGB-decoded when flagged) for
     * CPU-side integration (mirrors GPU nearest-sampling of mip 0).
     */
    /** Returns the raw source (displacement needs a standalone texture). */
    getSource(textureID: number): TextureSource | undefined {
        return this.sources[textureID];
    }

    readLinearPixels(textureID: number): { width: number; height: number; rgb: Float32Array } | null {
        const source = this.sources[textureID];
        if (!source) return null;
        const { bitmap, srgb } = source;
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const c2d = canvas.getContext("2d", { willReadFrequently: true })!;
        c2d.drawImage(bitmap, 0, 0);
        const bytes = c2d.getImageData(0, 0, bitmap.width, bitmap.height).data;
        const rgb = new Float32Array(bitmap.width * bitmap.height * 3);
        const decode = (b: number) => {
            const c = b / 255;
            return srgb ? (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)) : c;
        };
        for (let i = 0; i < bitmap.width * bitmap.height; i++) {
            rgb[i * 3] = decode(bytes[i * 4]!);
            rgb[i * 3 + 1] = decode(bytes[i * 4 + 1]!);
            rgb[i * 3 + 2] = decode(bytes[i * 4 + 2]!);
        }
        return { width: bitmap.width, height: bitmap.height, rgb };
    }

    /**
     * TextureAnalyzer's result for a texture: per-channel constancy over the image, the constant
     * value (the first texel, sRGB-decoded like a GPU load) and whether the format has alpha.
     * Cached per texture.
     */
    analyze(textureID: number): TextureAnalysis | null {
        const cached = this.analyses.get(textureID);
        if (cached) return cached;
        const source = this.sources[textureID];
        if (!source) return null;
        const { bitmap, srgb, compressed } = source;
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const c2d = canvas.getContext("2d", { willReadFrequently: true })!;
        c2d.drawImage(bitmap, 0, 0);
        const bytes = c2d.getImageData(0, 0, bitmap.width, bitmap.height).data;
        const min = [255, 255, 255, 255];
        const max = [0, 0, 0, 0];
        for (let i = 0; i < bytes.length; i += 4) {
            for (let c = 0; c < 4; c++) {
                const v = bytes[i + c]!;
                if (v < min[c]!) min[c] = v;
                if (v > max[c]!) max[c] = v;
            }
        }
        // BC1/BC4/BC5 carry no alpha (doesFormatHaveAlpha); decoded images read back as RGBA.
        const noAlpha = compressed !== undefined && [ResourceFormat.BC1Unorm, ResourceFormat.BC1UnormSrgb, ResourceFormat.BC4Unorm, ResourceFormat.BC4Snorm, ResourceFormat.BC5Unorm, ResourceFormat.BC5Snorm].includes(compressed.format);
        const toLinear = (b: number) => {
            const c = b / 255;
            return srgb ? (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)) : c;
        };
        let nonConstant = 0;
        for (let c = 0; c < 4; c++) if (min[c] !== max[c] && !(c === 3 && noAlpha)) nonConstant |= 1 << c;
        const analysis: TextureAnalysis = {
            value: [toLinear(bytes[0] ?? 0), toLinear(bytes[1] ?? 0), toLinear(bytes[2] ?? 0), noAlpha ? 1 : (bytes[3] ?? 255) / 255],
            minAlpha: noAlpha ? 1 : min[3]! / 255,
            maxAlpha: noAlpha ? 1 : max[3]! / 255,
            hasAlpha: !noAlpha,
            isConstant: (mask: number) => (nonConstant & mask) === 0,
        };
        this.analyses.set(textureID, analysis);
        return analysis;
    }

    /**
     * Min/max of the alpha channel (mirrors the TextureAnalyzer result used by
     * BasicMaterial::optimizeTexture to decide the alpha mode). Cached per texture.
     */
    getAlphaRange(textureID: number): [number, number] | null {
        const a = this.analyze(textureID);
        return a ? [a.minAlpha, a.maxAlpha] : null;
    }

    /**
     * Builds the bucket arrays (at least one, with a white fallback) plus per-texture
     * info: uv scale (textures smaller than the layer sit top-left), bucket * 2 + 1
     * when the tiles fill the layer exactly, and the layer. Mirrors native per-slot
     * sRGB semantics — normal/specular maps must NOT be sRGB-decoded on sample.
     */
    build(device: Device): { buckets: TextureBucket[]; texInfo: Float32Array } {
        const maxLayers = device.gpuDevice.limits.maxTextureArrayLayers;
        // compressed: a DDS chain; mipSet: a `<MIP>` file set. Both keep their own levels at their exact size.
        type Bucket = { key: string; compressed: boolean; mipSet: boolean; format: ResourceFormat; width: number; height: number; mips: number; members: number[] };
        const byKey = new Map<string, Bucket[]>();
        const place = (id: number, useCompressed: boolean) => {
            const src = this.sources[id]!;
            const c = useCompressed ? src.compressed : undefined;
            const m = useCompressed && !c ? src.mips : undefined;
            const pow2 = (n: number) => 1 << Math.ceil(Math.log2(Math.max(1, n)));
            const size = c ? [Math.ceil(c.width / 4) * 4, Math.ceil(c.height / 4) * 4] : m ? [src.bitmap.width, src.bitmap.height] : [pow2(Math.max(src.bitmap.width, src.bitmap.height)), 0];
            const format = c ? c.format : src.srgb ? ResourceFormat.RGBA8UnormSrgb : ResourceFormat.RGBA8Unorm;
            const mips = c ? c.levels.length : m ? m.length + 1 : 0;
            const key = `${format}|${size[0]}x${size[1]}|${mips}${m ? "|set" : ""}`;
            const list = byKey.get(key) ?? [];
            byKey.set(key, list);
            let b = list.at(-1);
            if (!b || b.members.length >= maxLayers) {
                b = { key, compressed: !!c, mipSet: !!m, format, width: size[0]!, height: size[1]!, mips, members: [] };
                list.push(b);
            }
            b.members.push(id);
        };
        this.sources.forEach((s, id) => place(id, !!s.compressed || !!s.mips));
        // Over the binding budget: move the smallest compressed buckets to the decoded images.
        const all = () => [...byKey.values()].flat();
        const remove = (b: Bucket) => {
            const list = byKey.get(b.key)!;
            list.splice(list.indexOf(b), 1);
            if (list.length === 0) byKey.delete(b.key);
        };
        while (all().length > kMaxTextureBuckets) {
            const smallest = all().filter((b) => b.compressed || b.mipSet).sort((a, b) => a.members.length - b.members.length)[0];
            if (smallest) {
                remove(smallest);
                for (const id of smallest.members) place(id, false);
                continue;
            }
            // Then fold the smallest RGBA size class into the next larger one of its format (tiled).
            const rgba = all().filter((b) => !b.compressed && !b.mipSet).sort((a, b) => a.width - b.width);
            const from = rgba.find((b) => rgba.some((o) => o !== b && o.format === b.format && o.width > b.width && o.members.length + b.members.length <= maxLayers));
            if (!from) break;
            const into = rgba.find((o) => o !== from && o.format === from.format && o.width > from.width && o.members.length + from.members.length <= maxLayers)!;
            remove(from);
            into.members.push(...from.members);
        }
        const buckets = all();
        if (buckets.length > kMaxTextureBuckets) throw new RuntimeError(`TextureManager: ${buckets.length} texture sizes/formats exceed the ${kMaxTextureBuckets} material texture arrays`);

        const texInfo = new Float32Array(Math.max(this.sources.length, 1) * 4);
        texInfo.set([1, 1, 0, 0]);
        const out: TextureBucket[] = buckets.map((b, bucketIndex) => {
            // RGBA buckets: the size class's square layer; tiles repeat smaller textures.
            const own = b.compressed || b.mipSet;
            const width = b.width;
            const height = own ? b.height : b.width;
            const texture = new Texture(device, {
                type: ResourceType.Texture2D,
                width,
                height,
                arraySize: b.members.length,
                // Compressed: the file's own chain. RGBA: the full chain, generated post-upload
                // (texture-LOD modes need real mips).
                mipLevels: own ? b.mips : Math.floor(Math.log2(Math.max(width, height))) + 1,
                format: b.format,
                // Image uploads (copyExternalImageToTexture) need RenderAttachment.
                bindFlags: b.compressed ? ResourceBindFlags.ShaderResource : ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget,
                name: `TextureManager::materialTextures${bucketIndex}`,
            });
            b.members.forEach((id, layer) => {
                const src = this.sources[id]!;
                if (b.compressed) {
                    const c = src.compressed!;
                    c.levels.forEach((data, mip) => texture.setSubresourceBlob(mip, layer, data));
                    texInfo.set([c.width / width, c.height / height, bucketIndex * 2 + (c.width === width && c.height === height ? 1 : 0), layer], id * 4);
                    return;
                }
                if (b.mipSet) {
                    [src.bitmap, ...src.mips!].forEach((bitmap, mip) =>
                        device.gpuDevice.queue.copyExternalImageToTexture({ source: bitmap }, { texture: texture.gpuTexture, mipLevel: mip, origin: { x: 0, y: 0, z: layer } }, { width: bitmap.width, height: bitmap.height, depthOrArrayLayers: 1 }),
                    );
                    texInfo.set([1, 1, bucketIndex * 2 + 1, layer], id * 4);
                    return;
                }
                const { width: tw, height: th } = src.bitmap;
                for (let y = 0; y < height; y += th)
                    for (let x = 0; x < width; x += tw)
                        device.gpuDevice.queue.copyExternalImageToTexture(
                            { source: src.bitmap },
                            { texture: texture.gpuTexture, origin: { x, y, z: layer } },
                            { width: Math.min(tw, width - x), height: Math.min(th, height - y), depthOrArrayLayers: 1 },
                        );
                const exact = width % tw === 0 && height % th === 0;
                texInfo.set([tw / width, th / height, bucketIndex * 2 + (exact ? 1 : 0), layer], id * 4);
            });
            return { texture, generateMips: !own };
        });
        if (out.length === 0) {
            const texture = new Texture(device, { type: ResourceType.Texture2D, width: 1, height: 1, arraySize: 1, mipLevels: 1, format: ResourceFormat.RGBA8UnormSrgb, bindFlags: ResourceBindFlags.ShaderResource, name: "TextureManager::materialTextures0" });
            device.gpuDevice.queue.writeTexture({ texture: texture.gpuTexture }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, { width: 1, height: 1 });
            out.push({ texture, generateMips: false });
        }
        return { buckets: out, texInfo };
    }
}
