/**
 * Vertex array object + vertex layout mirroring Falcor/Core/API/VAO.h and
 * VertexLayout.h. Lowered to GPUVertexBufferLayout[] at pipeline creation.
 */

import type { Buffer } from "./Buffer.js";
import { ResourceFormat, toGpuTextureFormat } from "./Formats.js";
import { ArgumentError } from "../Error.js";

export enum Topology {
    Undefined,
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

export function toGpuTopology(topology: Topology): GPUPrimitiveTopology {
    switch (topology) {
        case Topology.PointList: return "point-list";
        case Topology.LineList: return "line-list";
        case Topology.LineStrip: return "line-strip";
        case Topology.TriangleList: return "triangle-list";
        case Topology.TriangleStrip: return "triangle-strip";
        default: throw new ArgumentError("Undefined topology");
    }
}

export enum InputClass {
    PerVertexData,
    PerInstanceData,
}

/** One vertex attribute inside a buffer layout (mirrors VertexBufferLayout::addElement). */
export interface VertexElement {
    name: string; // semantic name, e.g. "POSITION" (matched to shader inputs via reflection in M2)
    offset: number;
    format: ResourceFormat;
    arraySize: number;
    shaderLocation: number;
}

/** Vertex format of a ResourceFormat (subset of formats valid as vertex attributes). */
function toGpuVertexFormat(format: ResourceFormat): GPUVertexFormat {
    switch (format) {
        case ResourceFormat.R32Float: return "float32";
        case ResourceFormat.RG32Float: return "float32x2";
        case ResourceFormat.RGB32Float: return "float32x3";
        case ResourceFormat.RGBA32Float: return "float32x4";
        case ResourceFormat.R16Float: return "float16";
        case ResourceFormat.RG16Float: return "float16x2";
        case ResourceFormat.RGBA16Float: return "float16x4";
        case ResourceFormat.RGBA8Unorm: return "unorm8x4";
        case ResourceFormat.RGBA8Snorm: return "snorm8x4";
        case ResourceFormat.RG8Unorm: return "unorm8x2";
        case ResourceFormat.RGBA8Uint: return "uint8x4";
        case ResourceFormat.RGBA8Int: return "sint8x4";
        case ResourceFormat.R32Uint: return "uint32";
        case ResourceFormat.RG32Uint: return "uint32x2";
        case ResourceFormat.RGBA32Uint: return "uint32x4";
        case ResourceFormat.R32Int: return "sint32";
        case ResourceFormat.RG16Unorm: return "unorm16x2";
        case ResourceFormat.RGBA16Unorm: return "unorm16x4";
        case ResourceFormat.RGBA16Uint: return "uint16x4";
        default: throw new ArgumentError(`Format ${ResourceFormat[format]} is not a valid vertex format`);
    }
}

/** Mirrors Falcor::VertexBufferLayout. */
export class VertexBufferLayout {
    readonly elements: VertexElement[] = [];
    stride = 0;
    inputClass = InputClass.PerVertexData;
    instanceStepRate = 0;

    addElement(name: string, offset: number, format: ResourceFormat, arraySize = 1, shaderLocation?: number): this {
        this.elements.push({ name, offset, format, arraySize, shaderLocation: shaderLocation ?? this.elements.length });
        return this;
    }
    setInputClass(inputClass: InputClass, stepRate = 0): this {
        this.inputClass = inputClass;
        this.instanceStepRate = stepRate;
        return this;
    }

    getGpuLayout(): GPUVertexBufferLayout {
        return {
            arrayStride: this.stride,
            stepMode: this.inputClass === InputClass.PerInstanceData ? "instance" : "vertex",
            attributes: this.elements.map((e) => ({
                format: toGpuVertexFormat(e.format),
                offset: e.offset,
                shaderLocation: e.shaderLocation,
            })),
        };
    }
}

/** Mirrors Falcor::VertexLayout (one VertexBufferLayout per bound vertex buffer). */
export class VertexLayout {
    readonly bufferLayouts: VertexBufferLayout[] = [];

    addBufferLayout(index: number, layout: VertexBufferLayout): this {
        this.bufferLayouts[index] = layout;
        return this;
    }

    getGpuLayouts(): GPUVertexBufferLayout[] {
        return this.bufferLayouts.map((l) => l.getGpuLayout());
    }
}

/** Mirrors Falcor::Vao. */
export class Vao {
    constructor(
        public readonly topology: Topology,
        public readonly vertexLayout: VertexLayout | null,
        public readonly vertexBuffers: Buffer[],
        public readonly indexBuffer: Buffer | null = null,
        public readonly indexFormat: ResourceFormat = ResourceFormat.R32Uint,
    ) {
        if (indexBuffer && indexFormat !== ResourceFormat.R32Uint && indexFormat !== ResourceFormat.R16Uint) {
            throw new ArgumentError("Index buffer format must be R16Uint or R32Uint");
        }
    }

    getGpuIndexFormat(): GPUIndexFormat {
        return this.indexFormat === ResourceFormat.R16Uint ? "uint16" : "uint32";
    }
}

export { toGpuTextureFormat };
