/**
 * Parameter block mirroring Falcor/Core/API/ParameterBlock.h.
 *
 * Supports flat globals and nested ParameterBlock<T> hierarchies (e.g. gScene):
 * - Each parameter block gets its own bind-group space; its uniform members fold
 *   into an implicit std140 buffer at binding 0 of that space (Slang WGSL emission).
 * - Resource members flatten to globals named by the member path joined with '_'
 *   (e.g. gScene_materials_materialTexturesArray_0), matched against the
 *   parsed WGSL bindings.
 * - ShaderVar proxies navigate paths; assignment dispatches on value type
 *   (resource object vs uniform value).
 */

import type { Device } from "../API/Device.js";
import { Buffer } from "../API/Buffer.js";
import { Texture } from "../API/Texture.js";
import { Sampler } from "../API/Sampler.js";
import { ResourceBindFlags, MemoryType } from "../API/Types.js";
import { ProgramReflection, ReflectionVar, type WgslBinding } from "./ProgramReflection.js";
import { RuntimeError, ArgumentError } from "../Error.js";
import type { SlangReflectionType, SlangReflectionParameter } from "./SlangCompiler.js";

type BindableResource = Buffer | Texture | Sampler | GPUTextureView;

interface CBufferSlot {
    kind: "cbuffer";
    binding: WgslBinding;
    /** Struct type whose uniform-field offsets address into cpuData. */
    elementType: SlangReflectionType;
    cpuData: ArrayBuffer;
    view: DataView;
    gpuBuffer: Buffer;
    dirty: boolean;
}

interface ResourceSlot {
    kind: "resource";
    binding: WgslBinding;
    resource: BindableResource | null;
}

type Slot = CBufferSlot | ResourceSlot;

/** Strips Slang's WGSL suffix (gOut_0 -> gOut). */
function demangle(name: string): string {
    return name.replace(/_\d+$/, "");
}

function computeStructSize(type: SlangReflectionType): number {
    let size = 16;
    for (const f of type.fields ?? []) {
        const b = f.binding as { kind?: string; offset?: number; size?: number } | undefined;
        if (b?.kind === "uniform") size = Math.max(size, (b.offset ?? 0) + (b.size ?? 0));
    }
    return Math.ceil(size / 16) * 16;
}

function isResourceKind(type: SlangReflectionType | undefined): boolean {
    const kind = type?.kind;
    return kind === "resource" || kind === "samplerState" || kind === "constantBuffer";
}

function hasUniformContent(type: SlangReflectionType): boolean {
    return (type.fields ?? []).some((f) => (f.binding as { kind?: string } | undefined)?.kind === "uniform");
}

export class ParameterBlock {
    /** Slots keyed by dot-joined path ("gOut", "gScene", "gScene.vertices.data0"). */
    private slots = new Map<string, Slot>();
    private wgslByName = new Map<string, WgslBinding>();
    private groups = new Map<number, { layout: GPUBindGroupLayout; bindGroup: GPUBindGroup | null; generation: number }>();
    private generation = 0;
    /** Reflection element types for cbuffer-slot paths (uniform member lookup). */
    private topLevel = new Map<string, ReflectionVar>();

    constructor(
        public readonly device: Device,
        public readonly reflection: ProgramReflection,
        public readonly wgslBindings: WgslBinding[],
    ) {
        for (const wb of wgslBindings) this.wgslByName.set(demangle(wb.name), wb);
        for (const p of reflection.json.parameters ?? []) {
            this.topLevel.set(p.name, new ReflectionVar(p.name, p.type ?? { kind: "unknown" }, null));
            this.registerParameter(p, [p.name]);
        }

        const groupIndices = new Set(wgslBindings.map((b) => b.group));
        for (const g of groupIndices) {
            const layout = device.gpuDevice.createBindGroupLayout({
                entries: wgslBindings.filter((b) => b.group === g).map((b) => b.layoutEntry),
            });
            this.groups.set(g, { layout, bindGroup: null, generation: -1 });
        }
    }

    private registerParameter(p: SlangReflectionParameter, path: string[]): void {
        const type = p.type ?? { kind: "unknown" };
        const flatName = path.join("_");
        const key = path.join(".");

        if (type.kind === "parameterBlock" || type.kind === "constantBuffer") {
            // Reflection JSON shape differs between slang-wasm and native slangc:
            // fields live on elementType or under elementVarLayout.type.
            const element = (type.elementType?.fields ? type.elementType : type.elementVarLayout?.type) ?? type.elementType ?? { kind: "struct" };
            // Implicit uniform buffer for the block's uniform members. The exact
            // std140 size (incl. trailing padding) comes from elementVarLayout.
            const wb = this.wgslByName.get(flatName);
            if (wb && wb.layoutEntry.buffer?.type === "uniform" && hasUniformContent(element)) {
                const uniformBinding = type.elementVarLayout?.bindings?.find((b) => b.kind === "uniform");
                this.addCBufferSlot(key, wb, element, uniformBinding?.size);
            }
            // Recurse into element fields for resources / nested blocks.
            for (const f of element.fields ?? []) {
                this.registerParameter(f as SlangReflectionParameter, [...path, f.name]);
            }
            return;
        }

        if (type.kind === "struct") {
            // Struct containing resources (e.g. SplitVertexBuffer) or pure uniforms (handled by parent cbuffer).
            for (const f of type.fields ?? []) {
                if (isResourceKind(f.type) || f.type?.kind === "struct" || f.type?.kind === "parameterBlock") {
                    this.registerParameter(f as SlangReflectionParameter, [...path, f.name]);
                }
            }
            return;
        }

        const wb = this.wgslByName.get(flatName);
        if (!wb) return; // statically unused or uniform member (lives in parent cbuffer)
        if (wb.layoutEntry.buffer?.type === "uniform" && type.kind !== "resource") return;
        this.slots.set(key, { kind: "resource", binding: wb, resource: null });
    }

    private addCBufferSlot(key: string, wb: WgslBinding, elementType: SlangReflectionType, exactSize?: number): void {
        const size = exactSize && exactSize > 0 ? Math.ceil(exactSize / 16) * 16 : computeStructSize(elementType);
        const cpuData = new ArrayBuffer(size);
        this.slots.set(key, {
            kind: "cbuffer",
            binding: wb,
            elementType,
            cpuData,
            view: new DataView(cpuData),
            gpuBuffer: new Buffer(this.device, { size, bindFlags: ResourceBindFlags.Constant, memoryType: MemoryType.DeviceLocal, name: `cb:${key}` }),
            dirty: true,
        });
    }

    getSlotNames(): string[] {
        return [...this.slots.keys()];
    }

    /** Resource assignment at a path (["gScene","vertices","data0"]). */
    setResourceByPath(path: string[], resource: BindableResource): void {
        const key = path.join(".");
        const slot = this.slots.get(key);
        if (!slot) {
            if (this.pathExistsInReflection(path)) return; // statically unused
            throw new ArgumentError(`No shader parameter at path '${key}'`);
        }
        if (slot.kind !== "resource") throw new ArgumentError(`'${key}' is a constant buffer, not a resource`);
        slot.resource = resource;
        this.generation++;
    }

    /** Uniform write: longest slot prefix is the containing cbuffer; rest addresses members. */
    setUniformByPath(path: string[], value: unknown): void {
        for (let prefixLen = path.length - 1; prefixLen >= 1; prefixLen--) {
            const key = path.slice(0, prefixLen).join(".");
            const slot = this.slots.get(key);
            if (slot?.kind === "cbuffer") {
                let v: ReflectionVar | undefined = new ReflectionVar(key, { kind: "struct", fields: slot.elementType.fields }, null);
                for (const part of path.slice(prefixLen)) {
                    v = v.findMember(part);
                    if (!v) throw new ArgumentError(`No member '${path.join(".")}'`);
                }
                this.writeValue(slot, v, value);
                slot.dirty = true;
                this.generation++;
                return;
            }
        }
        if (this.pathExistsInReflection(path)) return; // statically unused
        throw new ArgumentError(`No constant buffer containing '${path.join(".")}'`);
    }

    /** Legacy flat API used by earlier passes. */
    setResource(name: string, resource: BindableResource): void {
        this.setResourceByPath([name], resource);
    }
    setUniform(cbufferName: string, memberPath: string[], value: unknown): void {
        this.setUniformByPath([cbufferName, ...memberPath], value);
    }

    private pathExistsInReflection(path: string[]): boolean {
        let v = this.topLevel.get(path[0]!);
        if (!v) return false;
        for (const part of path.slice(1)) {
            const next: ReflectionVar | undefined = v.findMember(part);
            if (!next) return false;
            v = next;
        }
        return true;
    }

    private writeValue(slot: CBufferSlot, member: ReflectionVar, value: unknown): void {
        const type = member.type;
        const offset = member.byteOffset;
        const write = (off: number, scalarType: string, val: number | boolean) => {
            const n = typeof val === "boolean" ? (val ? 1 : 0) : val;
            switch (scalarType) {
                case "float32": slot.view.setFloat32(off, n as number, true); break;
                case "uint32": case "bool": slot.view.setUint32(off, n as number, true); break;
                case "int32": slot.view.setInt32(off, n as number, true); break;
                default: throw new RuntimeError(`Unsupported scalar type '${scalarType}'`);
            }
        };
        if (type.kind === "scalar") {
            if (typeof value !== "number" && typeof value !== "boolean") throw new ArgumentError(`Expected scalar for '${member.name}'`);
            write(offset, type.scalarType ?? "float32", value);
        } else if (type.kind === "vector") {
            const arr = value as ArrayLike<number>;
            const scalarType = type.elementType?.scalarType ?? "float32";
            const count = type.elementCount ?? 0;
            if (arr.length !== count) throw new ArgumentError(`Expected ${count} components for '${member.name}', got ${arr.length}`);
            for (let i = 0; i < count; i++) write(offset + i * 4, scalarType, arr[i]!);
        } else if (type.kind === "matrix") {
            // Host is row-major; Slang WGSL stores square matrices as vec4-aligned rows,
            // non-square transposed (columns as elements) — disambiguated by size (GPU-verified).
            const maybe = value as { toArray?: () => Float32Array };
            const arr = (typeof maybe?.toArray === "function" ? maybe.toArray() : value) as ArrayLike<number>;
            const rows = type.rowCount ?? 4;
            const cols = type.columnCount ?? 4;
            if (arr.length !== rows * cols) throw new ArgumentError(`Expected ${rows * cols} floats for '${member.name}', got ${arr.length}`);
            const size = member.byteSize > 0 ? member.byteSize : rows * 16;
            if (size === rows * 16) {
                for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) write(offset + r * 16 + c * 4, "float32", arr[r * cols + c]!);
            } else if (size === cols * 16) {
                for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) write(offset + c * 16 + r * 4, "float32", arr[r * cols + c]!);
            } else {
                throw new RuntimeError(`Unexpected matrix layout for '${member.name}': float${rows}x${cols} in ${size} bytes`);
            }
        } else {
            throw new RuntimeError(`Setting '${type.kind}' uniforms not implemented yet`);
        }
    }

    /** Uploads dirty cbuffers, returns the bind group for a group index. */
    getBindGroup(group: number): GPUBindGroup {
        const g = this.groups.get(group);
        if (!g) throw new RuntimeError(`No bind group ${group} in this parameter block`);
        for (const slot of this.slots.values()) {
            if (slot.kind === "cbuffer" && slot.dirty) {
                slot.gpuBuffer.setBlob(new Uint8Array(slot.cpuData));
                slot.dirty = false;
            }
        }
        if (!g.bindGroup || g.generation !== this.generation) {
            const entries: GPUBindGroupEntry[] = [];
            for (const [key, slot] of this.slots) {
                if (slot.binding.group !== group) continue;
                if (slot.kind === "cbuffer") {
                    entries.push({ binding: slot.binding.binding, resource: { buffer: slot.gpuBuffer.gpuBuffer } });
                } else {
                    if (!slot.resource) throw new RuntimeError(`Shader parameter '${key}' is not bound`);
                    entries.push({ binding: slot.binding.binding, resource: this.lowerResource(slot) });
                }
            }
            g.bindGroup = this.device.gpuDevice.createBindGroup({ layout: g.layout, entries });
            g.generation = this.generation;
        }
        return g.bindGroup;
    }

    getBindGroupLayout(group: number): GPUBindGroupLayout | undefined {
        return this.groups.get(group)?.layout;
    }

    getGroupIndices(): number[] {
        return [...this.groups.keys()].sort((a, b) => a - b);
    }

    private lowerResource(slot: ResourceSlot): GPUBindingResource {
        const r = slot.resource!;
        if (r instanceof Buffer) return { buffer: r.gpuBuffer };
        if (r instanceof Texture) {
            // View dimension must match the layout's declaration (e.g. a one-layer
            // texture bound as texture_2d_array still needs a 2d-array view).
            const entry = slot.binding.layoutEntry;
            const dim = entry.storageTexture?.viewDimension ?? entry.texture?.viewDimension;
            if (entry.storageTexture) return r.getView(0, 1, 0, undefined, dim);
            return r.getView(0, undefined, 0, undefined, dim);
        }
        if (r instanceof Sampler) return r.gpuSampler;
        return r;
    }
}

/**
 * ShaderVar mirroring Falcor/Core/Program/ShaderVar.h: proxy path access with
 * value-type dispatch (resources vs uniform values).
 */
export type ShaderVar = {
    [key: string]: any;
};

function isBindable(value: unknown): value is BindableResource {
    return value instanceof Buffer || value instanceof Texture || value instanceof Sampler || (typeof GPUTextureView !== "undefined" && value instanceof GPUTextureView);
}

export function makeRootVar(block: ParameterBlock): ShaderVar {
    const makeProxy = (path: string[]): any =>
        new Proxy(Object.create(null), {
            get: (_t, prop: string) => makeProxy([...path, prop]),
            set: (_t, prop: string, value) => {
                const full = [...path, prop];
                if (isBindable(value)) block.setResourceByPath(full, value);
                else block.setUniformByPath(full, value);
                return true;
            },
        });
    return makeProxy([]);
}
