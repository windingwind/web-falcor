/**
 * GPU device abstraction mirroring Falcor/Core/API/Device.h.
 *
 * Wraps GPUAdapter/GPUDevice. Falcor's Device owns the GFX backend device,
 * upload/readback heaps, the program manager and profiler; those subsystems
 * attach here as they are implemented.
 */

import { RuntimeError } from "../Error.js";
import { Logger } from "../../Utils/Logger.js";

/** Mirrors Falcor::Device::Type (D3D12/Vulkan). The web backend is WebGPU; WebGL2 fallback is raster-only. */
export enum DeviceType {
    Default,
    WebGPU,
    WebGL2,
}

/** Subset of Falcor::Device::Desc relevant on the web platform. */
export interface DeviceDesc {
    type?: DeviceType;
    /** Features to require if available (mirrors Desc::requiredFeatures). */
    requiredFeatures?: GPUFeatureName[];
    enableDebugLayer?: boolean;
    powerPreference?: GPUPowerPreference;
}

/** Mirrors Falcor::Device::SupportedFeatures, translated to WebGPU capability queries. */
export interface SupportedFeatures {
    /** Hardware ray tracing pipelines: not available in WebGPU; software fallback is used (DESIGN.md §RayTracing). */
    raytracing: boolean;
    /** Inline ray queries: same as above. */
    raytracingQuery: boolean;
    shaderFloat16: boolean;
    timestampQueries: boolean;
    textureCompressionBC: boolean;
    indirectFirstInstance: boolean;
    subgroups: boolean;
}

export class Device {
    private constructor(
        public readonly adapter: GPUAdapter,
        public readonly gpuDevice: GPUDevice,
        public readonly desc: DeviceDesc,
    ) {}

    /** Async factory (WebGPU device acquisition is async, unlike Falcor's constructor). */
    static async create(desc: DeviceDesc = {}): Promise<Device> {
        if (!("gpu" in navigator)) {
            throw new RuntimeError(
                "WebGPU is not available in this browser. web-falcor requires WebGPU (WebGL2 fallback covers raster-only paths).",
            );
        }
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: desc.powerPreference ?? "high-performance",
        });
        if (!adapter) throw new RuntimeError("Failed to acquire a WebGPU adapter.");

        const wanted = desc.requiredFeatures ?? [
            "timestamp-query",
            "shader-f16",
            "texture-compression-bc",
            "float32-filterable",
            "indirect-first-instance",
        ];
        const features = wanted.filter((f) => adapter.features.has(f));

        const gpuDevice = await adapter.requestDevice({
            requiredFeatures: features,
            requiredLimits: {
                maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            },
        });

        gpuDevice.lost.then((info) => {
            Logger.error(`GPU device lost: ${info.reason} - ${info.message}`);
        });

        const device = new Device(adapter, gpuDevice, desc);
        Logger.info(`Created WebGPU device (features: ${features.join(", ") || "none"})`);
        return device;
    }

    get limits(): GPUSupportedLimits {
        return this.gpuDevice.limits;
    }

    get queue(): GPUQueue {
        return this.gpuDevice.queue;
    }

    hasFeature(feature: GPUFeatureName): boolean {
        return this.gpuDevice.features.has(feature);
    }

    getSupportedFeatures(): SupportedFeatures {
        return {
            raytracing: false,
            raytracingQuery: false,
            shaderFloat16: this.hasFeature("shader-f16"),
            timestampQueries: this.hasFeature("timestamp-query"),
            textureCompressionBC: this.hasFeature("texture-compression-bc"),
            indirectFirstInstance: this.hasFeature("indirect-first-instance"),
            subgroups: this.hasFeature("subgroups" as GPUFeatureName),
        };
    }

    /** Mirrors Device::endFrame()/flushAndSync: waits for all submitted GPU work. */
    async wait(): Promise<void> {
        await this.gpuDevice.queue.onSubmittedWorkDone();
    }

    destroy(): void {
        this.gpuDevice.destroy();
    }
}
