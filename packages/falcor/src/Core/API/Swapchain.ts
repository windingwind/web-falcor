/**
 * Swapchain mirroring Falcor/Core/API/Swapchain.h, backed by a canvas WebGPU context.
 * No vsync/refresh control on the web (parity matrix §8.1); presentation happens
 * implicitly at the end of the browser frame.
 */

import type { Device } from "./Device.js";
import { RuntimeError } from "../Error.js";

export interface SwapchainDesc {
    format?: GPUTextureFormat;
    /** 'premultiplied' enables compositing over the page; Falcor default is opaque. */
    alphaMode?: GPUCanvasAlphaMode;
    colorSpace?: PredefinedColorSpace;
    toneMappingMode?: "standard" | "extended";
}

export class Swapchain {
    readonly context: GPUCanvasContext;
    readonly format: GPUTextureFormat;

    constructor(
        device: Device,
        public readonly canvas: HTMLCanvasElement | OffscreenCanvas,
        desc: SwapchainDesc = {},
    ) {
        const context = canvas.getContext("webgpu");
        if (!context) throw new RuntimeError("Failed to acquire WebGPU canvas context");
        this.context = context;
        this.format = desc.format ?? navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: device.gpuDevice,
            format: this.format,
            alphaMode: desc.alphaMode ?? "opaque",
            colorSpace: desc.colorSpace ?? "srgb",
            toneMapping: desc.toneMappingMode ? { mode: desc.toneMappingMode } : undefined,
        });
    }

    /** Mirrors Swapchain::acquireNextImage. */
    acquireNextImage(): GPUTexture {
        return this.context.getCurrentTexture();
    }

    /** Mirrors Swapchain::resize (canvas size drives the backing texture). */
    resize(width: number, height: number): void {
        this.canvas.width = width;
        this.canvas.height = height;
    }
}
