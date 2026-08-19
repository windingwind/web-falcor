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

/** Mirrors SolidAngleBoundMethod (LightBVHSamplerSharedDefinitions.slang). */
export const kSolidAngleBoundMethods: Record<string, number> = { BoxToCenter: 1, BoxToAverage: 2, Sphere: 3 };

/** Mirrors LightBVHSampler::Options (buildOptions nested like the native serialization). */
export interface LightBVHSamplerOptions {
    buildOptions: LightBVHOptions;
    useBoundingCone: boolean;
    useLightingCone: boolean;
    disableNodeFlux: boolean;
    useUniformTriangleSampling: boolean;
    solidAngleBoundMethod: number; // SolidAngleBoundMethod
}

export const kDefaultLightBVHSamplerOptions: LightBVHSamplerOptions = {
    buildOptions: kDefaultLightBVHOptions,
    useBoundingCone: true,
    useLightingCone: true,
    disableNodeFlux: false,
    useUniformTriangleSampling: true,
    solidAngleBoundMethod: kSolidAngleBoundMethods["Sphere"]!,
};

export class LightBVHSampler {
    private readonly nodes: Buffer;
    private readonly triangleIndices: Buffer;
    private readonly triangleBitmasks: Buffer;
    private readonly options: LightBVHSamplerOptions;

    constructor(device: Device, triangles: EmissiveTriangleInput[], options: LightBVHSamplerOptions = kDefaultLightBVHSamplerOptions) {
        this.options = options;
        const result = buildLightBVH(triangles, options.buildOptions);

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

    /** Mirrors LightBVHSampler::getDefines. */
    getDefines(): DefineList {
        const o = this.options;
        return new DefineList()
            .add("_EMISSIVE_LIGHT_SAMPLER_TYPE", "1") // EMISSIVE_LIGHT_SAMPLER_LIGHT_BVH
            .add("_USE_BOUNDING_CONE", o.useBoundingCone ? "1" : "0")
            .add("_USE_LIGHTING_CONE", o.useLightingCone ? "1" : "0")
            .add("_DISABLE_NODE_FLUX", o.disableNodeFlux ? "1" : "0")
            .add("_USE_UNIFORM_TRIANGLE_SAMPLING", o.useUniformTriangleSampling ? "1" : "0")
            .add("_ACTUAL_MAX_TRIANGLES_PER_NODE", String(o.buildOptions.maxTriangleCountPerLeaf))
            .add("_SOLID_ANGLE_BOUND_METHOD", String(o.solidAngleBoundMethod));
    }

    /** Binds under emissiveSampler (fields live at _lightBVH.*). */
    bindShaderData(var_: ShaderVar): void {
        const bvh = var_["_lightBVH"] as ShaderVar;
        bvh["nodes"] = this.nodes;
        bvh["triangleIndices"] = this.triangleIndices;
        bvh["triangleBitmasks"] = this.triangleBitmasks;
    }
}
