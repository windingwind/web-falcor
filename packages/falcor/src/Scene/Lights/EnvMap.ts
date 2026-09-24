/**
 * Environment map mirroring Falcor/Scene/Lights/EnvMap.{h,cpp}: a lat-long
 * radiance texture plus EnvMapData (rotation transform, tint, intensity).
 *
 * v1 loads Radiance .hdr (RGBE) images into an rgba32float texture with a
 * single mip level. Native generates a full mip chain, but EnvMap.eval
 * samples at lod 0 so results match; mip chains come with the texture-LOD
 * work.
 */

import type { Device } from "../../Core/API/Device.js";
import { Texture } from "../../Core/API/Texture.js";
import { Sampler, TextureAddressingMode, TextureFilteringMode } from "../../Core/API/Sampler.js";
import { ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { RuntimeError } from "../../Core/Error.js";
import { decodeHdr, type HdrImage } from "../../Utils/Image/HDRDecoder.js";
import { convertEqualAreaOctToLatLong } from "./EnvMapConverter.js";
import { decodePfm, isPfm } from "../../Utils/Image/PFMDecoder.js";

export interface EnvMapLoadOptions {
    /** The file is an equal-area octahedral map (pbrt-v4) to convert to lat-long. */
    equalAreaOctahedral?: boolean;
}
import { decodeExr } from "../../Utils/Image/EXRDecoder.js";
import { float4x4, inverse, matrixFromRotationXYZ } from "../../Utils/Math/Matrix.js";

/** Row-major 3x4 rows of a float4x4 (EnvMapData transform layout). */
function toRows3x4(m: float4x4): number[] {
    const a = m.toArray();
    return [...a.slice(0, 4), ...a.slice(4, 8), ...a.slice(8, 12)].map((v) => v);
}

export class EnvMap {
    readonly texture: Texture;
    readonly sampler: Sampler;
    intensity = 1;
    /** Source file (EnvMap::getPath); "" when built from memory. */
    path = "";
    tint: [number, number, number] = [1, 1, 1];
    private transform = float4x4.identity();
    private invTransform = float4x4.identity();
    /** Original encoded file (retained by createFromUrl/Bytes for the scene cache). */
    sourceBytes: Uint8Array | null = null;
    sourceIsExr = false;
    /** The source was an equal-area octahedral map (pbrt-v4), converted on load. */
    sourceEqualAreaOctahedral = false;
    /** Last rotation set (degrees), for cache snapshots. */
    rotationDeg: [number, number, number] = [0, 0, 0];

    /** Python `envMap.rotation`: XYZ angles in degrees. */
    get rotation(): { x: number; y: number; z: number } {
        return { x: this.rotationDeg[0], y: this.rotationDeg[1], z: this.rotationDeg[2] };
    }
    set rotation(v: { x: number; y: number; z: number }) {
        this.setRotation([Number(v.x), Number(v.y), Number(v.z)]);
    }

    /** Mirrors EnvMap::setRotation(degreesXYZ). */
    setRotation(degreesXYZ: [number, number, number]): void {
        this.rotationDeg = [...degreesXYZ];
        const r = (d: number) => (d * Math.PI) / 180;
        this.transform = matrixFromRotationXYZ(r(degreesXYZ[0]), r(degreesXYZ[1]), r(degreesXYZ[2]));
        this.invTransform = inverse(this.transform);
    }

    constructor(device: Device, image: HdrImage) {
        this.texture = new Texture(device, {
            type: ResourceType.Texture2D,
            width: image.width,
            height: image.height,
            arraySize: 1,
            mipLevels: 1,
            format: ResourceFormat.RGBA32Float,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "EnvMap::texture",
        });
        device.gpuDevice.queue.writeTexture(
            { texture: this.texture.gpuTexture },
            image.data as Float32Array<ArrayBuffer>,
            { bytesPerRow: image.width * 16 },
            { width: image.width, height: image.height, depthOrArrayLayers: 1 },
        );
        // The lat-long map wraps around horizontally, but not vertically (EnvMap.cpp).
        this.sampler = new Sampler(device, {
            magFilter: TextureFilteringMode.Linear,
            minFilter: TextureFilteringMode.Linear,
            mipFilter: TextureFilteringMode.Linear,
            addressModeU: TextureAddressingMode.Wrap,
            addressModeV: TextureAddressingMode.Clamp,
            addressModeW: TextureAddressingMode.Clamp,
        });
    }

    static async createFromUrl(device: Device, url: string, options: EnvMapLoadOptions = {}): Promise<EnvMap> {
        const res = await fetch(url);
        if (!res.ok) throw new RuntimeError(`Failed to fetch env map '${url}' (${res.status})`);
        const env = EnvMap.createFromBytes(device, new Uint8Array(await res.arrayBuffer()), url.toLowerCase().endsWith(".exr"), options);
        env.path = url;
        return env;
    }

    /** Decodes an encoded .hdr/.exr file; retains the bytes for the scene cache. */
    static createFromBytes(device: Device, bytes: Uint8Array, isExr: boolean, options: EnvMapLoadOptions = {}): EnvMap {
        // PFM is recognised by its signature (pbrt-v4 env maps); EXR/HDR by the flag.
        let image = isPfm(bytes) ? decodePfm(bytes) : isExr ? decodeExr(bytes.slice().buffer as ArrayBuffer) : decodeHdr(bytes);
        if (options.equalAreaOctahedral) image = convertEqualAreaOctToLatLong(image);
        const env = new EnvMap(device, image);
        env.sourceBytes = bytes;
        env.sourceIsExr = isExr;
        env.sourceEqualAreaOctahedral = !!options.equalAreaOctahedral;
        return env;
    }

    /** Binds to gScene.envMap. */
    bindShaderData(var_: ShaderVar): void {
        const data = var_["data"] as ShaderVar;
        data["transform"] = toRows3x4(this.transform);
        data["invTransform"] = toRows3x4(this.invTransform);
        data["tint"] = this.tint;
        data["intensity"] = this.intensity;
        var_["envMap"] = this.texture;
        var_["envSampler"] = this.sampler;
    }
}
