/**
 * LightBVH emissive sampler host mirroring Rendering/Lights/LightBVH.{h,cpp} +
 * LightBVHSampler.{h,cpp}: builds the BVH on the CPU (LightBVHBuilder.ts),
 * uploads nodes/triangleIndices/triangleBitmasks, and provides the sampler
 * defines + bindings (_lightBVH member of EmissiveLightSampler when
 * _EMISSIVE_LIGHT_SAMPLER_TYPE == LIGHT_BVH).
 */

import type { Device } from "../../Core/API/Device.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import { DefineList } from "../../Core/Program/DefineList.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { buildLightBVH, kDefaultLightBVHOptions, type EmissiveTriangleInput, type LightBVHOptions } from "./LightBVHBuilder.js";

export class LightBVHSampler {
    private readonly nodes: Buffer;
    private readonly triangleIndices: Buffer;
    private readonly triangleBitmasks: Buffer;

    constructor(device: Device, triangles: EmissiveTriangleInput[], options: LightBVHOptions = kDefaultLightBVHOptions) {
        const result = buildLightBVH(triangles, options);

        const make = (name: string, data: ArrayBufferView | ArrayBuffer, structSize: number) => {
            const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            const buf = new Buffer(device, {
                size: Math.max(bytes.byteLength, structSize),
                structSize,
                bindFlags: ResourceBindFlags.ShaderResource,
                memoryType: MemoryType.DeviceLocal,
                name: `LightBVH::${name}`,
            });
            buf.setBlob(bytes);
            return buf;
        };
        this.nodes = make("nodes", result.nodes, 32);
        this.triangleIndices = make("triangleIndices", result.triangleIndices, 4);
        this.triangleBitmasks = make("triangleBitmasks", result.triangleBitmasks, 8);
    }

    /** Mirrors LightBVHSampler::getDefines (default options). */
    getDefines(): DefineList {
        return new DefineList()
            .add("_EMISSIVE_LIGHT_SAMPLER_TYPE", "1") // EMISSIVE_LIGHT_SAMPLER_LIGHT_BVH
            .add("_USE_BOUNDING_CONE", "1")
            .add("_USE_LIGHTING_CONE", "1")
            .add("_DISABLE_NODE_FLUX", "0")
            .add("_USE_UNIFORM_TRIANGLE_SAMPLING", "1")
            .add("_ACTUAL_MAX_TRIANGLES_PER_NODE", "10")
            .add("_SOLID_ANGLE_BOUND_METHOD", "3"); // SolidAngleBoundMethod::Sphere
    }

    /** Binds under emissiveSampler (fields live at _lightBVH.*). */
    bindShaderData(var_: ShaderVar): void {
        const bvh = var_["_lightBVH"] as ShaderVar;
        bvh["nodes"] = this.nodes;
        bvh["triangleIndices"] = this.triangleIndices;
        bvh["triangleBitmasks"] = this.triangleBitmasks;
    }
}
