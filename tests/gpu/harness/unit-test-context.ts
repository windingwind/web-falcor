/**
 * Mirrors Falcor/Testing/UnitTest.h GPUUnitTestContext: the helper native
 * FalcorTest GPU tests drive kernels through (createProgram, ctx["var"] = ...,
 * allocateStructuredBuffer, runProgram, readBuffer). Transplanted tests under
 * tests/gpu/suites/falcortest/ keep native's shaders (Tests/...cs.slang, served
 * from Falcor/Source/Tools/FalcorTest) and assertions.
 *
 * §9: readBuffer is async (WebGPU readback); structured-buffer strides come
 * from the WGSL layout Slang emits (float3 elements are 16 bytes, not 12).
 */

import { Buffer, ComputePass, DefineList, ResourceBindFlags, type Device, type ProgramReflection, type ShaderModuleDesc, type ShaderVar, type TypeConformance } from "@web-falcor/falcor";

type TypedArray = Float32Array | Uint32Array | Int32Array | Uint16Array | Uint8Array | Float64Array;
type TypedArrayCtor<T extends TypedArray> = { new (buffer: ArrayBuffer): T };

interface ReflectionType {
    kind?: string;
    scalarType?: string;
    elementCount?: number;
    rowCount?: number;
    columnCount?: number;
    elementType?: ReflectionType;
    resultType?: ReflectionType;
    fields?: { name: string; type?: ReflectionType; binding?: { offset?: number; size?: number } }[];
}

const kScalarBytes: Record<string, number> = { float16: 2, int16: 2, uint16: 2, float64: 8, int64: 8, uint64: 8 };

/** WGSL (std430-like) size and alignment of a reflected type. */
function sizeAlign(t: ReflectionType | undefined): { size: number; align: number } {
    if (!t) return { size: 4, align: 4 };
    if (t.kind === "scalar") {
        const s = kScalarBytes[t.scalarType ?? ""] ?? 4;
        return { size: s, align: s };
    }
    if (t.kind === "vector") {
        const e = sizeAlign(t.elementType).size;
        const n = t.elementCount ?? 1;
        return { size: n * e, align: (n === 3 ? 4 : n) * e };
    }
    if (t.kind === "matrix") {
        // Column vectors of rowCount lanes (Slang emits row-major Falcor matrices transposed).
        const col = sizeAlign({ kind: "vector", elementCount: t.rowCount, elementType: t.elementType });
        return { size: col.align * (t.columnCount ?? 1), align: col.align };
    }
    if (t.kind === "array") {
        const e = sizeAlign(t.elementType);
        const stride = Math.ceil(e.size / e.align) * e.align;
        return { size: stride * (t.elementCount ?? 1), align: e.align };
    }
    if (t.kind === "struct") {
        let size = 0;
        let align = 1;
        for (const f of t.fields ?? []) {
            const fa = sizeAlign(f.type);
            const offset = f.binding?.offset ?? Math.ceil(size / fa.align) * fa.align;
            size = Math.max(size, offset + (f.binding?.size ?? fa.size));
            align = Math.max(align, fa.align);
        }
        return { size: Math.ceil(size / align) * align, align };
    }
    return { size: 4, align: 4 };
}

export class GPUUnitTestContext {
    private pass: ComputePass | null = null;
    private buffers = new Map<string, Buffer>();

    constructor(readonly device: Device) {}

    getDevice(): Device {
        return this.device;
    }
    getRenderContext() {
        return this.device.renderContext;
    }

    /** Mirrors createProgram(path, csEntry, defines). */
    createProgram(path: string, csEntry = "main", defines: DefineList | Record<string, string | number | boolean> = {}, typeConformances: TypeConformance[] = []): void {
        this.pass = ComputePass.create(this.device, { path, csEntry, defines, typeConformances });
        this.buffers.clear();
    }

    /** Mirrors createProgram(const ProgramDesc&, defines): shader modules from files and strings. */
    createProgramFromModules(modules: ShaderModuleDesc[], csEntry = "main", defines: DefineList | Record<string, string | number | boolean> = {}, typeConformances: TypeConformance[] = []): void {
        this.pass = ComputePass.create(this.device, { modules, csEntry, defines, typeConformances });
        this.buffers.clear();
    }

    /** Mirrors vars().getRootVar() / operator[]: `ctx.vars()["name"] = value`. */
    vars(): ShaderVar {
        if (!this.pass) throw new Error("GPUUnitTestContext: program not created");
        return this.pass.getRootVar();
    }

    /** Mirrors getProgram()->getReflector(). */
    getReflector(): ProgramReflection {
        return this.pass!.getReflector();
    }

    /** Thread-group size from reflection (mirrors mThreadGroupSize). */
    getThreadGroupSize(): [number, number, number] {
        return this.pass!.getThreadGroupSize();
    }

    /** Element stride of the structured buffer `name`, from the program's reflection. */
    getStructSize(name: string): number {
        const reflection = this.pass!.program.getActiveVersion().reflection as unknown as { json?: { parameters?: { name: string; type?: ReflectionType }[] } };
        const param = reflection.json?.parameters?.find((p) => p.name === name);
        if (!param) throw new Error(`GPUUnitTestContext: no shader parameter '${name}'`);
        const { size, align } = sizeAlign(param.type?.resultType ?? param.type?.elementType);
        return Math.ceil(size / align) * align;
    }

    /** Mirrors allocateStructuredBuffer(name, nElements, initData). */
    allocateStructuredBuffer(name: string, nElements: number, initData?: ArrayBufferView): Buffer {
        const structSize = this.getStructSize(name);
        const buffer = this.device.createStructuredBuffer(structSize, nElements, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
        if (initData) {
            if (initData.byteLength !== structSize * nElements) throw new Error(`StructuredBuffer '${name}' initial data size mismatch (${initData.byteLength} != ${structSize * nElements})`);
            buffer.setBlob(initData);
        }
        this.buffers.set(name, buffer);
        return buffer;
    }

    getBuffer(name: string): Buffer {
        const b = this.buffers.get(name);
        if (!b) throw new Error(`${name}: couldn't find buffer to map`);
        return b;
    }

    /** Mirrors runProgram(dimensions): total threads, rounded up to whole groups. */
    runProgram(x: number, y = 1, z = 1): void {
        const root = this.vars();
        for (const [name, buffer] of this.buffers) root[name] = buffer;
        const [gx, gy, gz] = this.getThreadGroupSize();
        const limit = this.device.gpuDevice.limits.maxComputeWorkgroupsPerDimension;
        if (Math.ceil(x / gx) > limit || Math.ceil(y / gy) > limit || Math.ceil(z / gz) > limit) throw new Error("GPUUnitTestContext::runProgram() - Dispatch dimension exceeds maximum.");
        this.pass!.execute(this.device.renderContext, x, y, z);
    }

    /** Mirrors readBuffer<T>(name) (async). */
    async readBuffer<T extends TypedArray>(name: string, ctor: TypedArrayCtor<T>): Promise<T> {
        const bytes = await this.getBuffer(name).getBlob();
        return new ctor(bytes.slice().buffer as ArrayBuffer);
    }
}

/** Reads a whole buffer as a typed array (Buffer::getElements<T>). */
export async function getElements<T extends TypedArray>(buffer: Buffer, ctor: TypedArrayCtor<T>): Promise<T> {
    const bytes = await buffer.getBlob();
    return new ctor(bytes.slice().buffer as ArrayBuffer);
}
