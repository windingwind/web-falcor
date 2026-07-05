/**
 * Parameter block mirroring Falcor/Core/API/ParameterBlock.h.
 *
 * Owns a CPU mirror of each constant buffer plus a resource table, both driven
 * by reflection. getBindGroup() lazily uploads dirty uniform data and builds
 * WebGPU bind groups against explicit reflection-derived layouts.
 */

import type { Device } from "../API/Device.js";
import { Buffer } from "../API/Buffer.js";
import { Texture } from "../API/Texture.js";
import { Sampler } from "../API/Sampler.js";
import { ResourceBindFlags, MemoryType } from "../API/Types.js";
import { ProgramReflection, ReflectionVar, type WgslBinding } from "./ProgramReflection.js";
import { RuntimeError, ArgumentError } from "../Error.js";
import type { SlangReflectionType } from "./SlangCompiler.js";

type BindableResource = Buffer | Texture | Sampler | GPUTextureView;

interface CBufferSlot {
    kind: "cbuffer";
    binding: WgslBinding;
    reflectionVar: ReflectionVar;
    cpuData: ArrayBuffer;
    view: DataView;
    gpuBuffer: Buffer;
    dirty: boolean;
}

interface ResourceSlot {
    kind: "resource";
    binding: WgslBinding;
    reflectionVar: ReflectionVar | null;
    resource: BindableResource | null;
}

type Slot = CBufferSlot | ResourceSlot;

/** Strips Slang's WGSL name suffix (gOut_0 -> gOut). */
function demangle(name: string): string {
    return name.replace(/_\d+$/, "");
}

function computeStructSize(type: SlangReflectionType): number {
    const el = type.kind === "constantBuffer" || type.kind === "parameterBlock" ? type.elementType : type;
    let size = 16;
    for (const f of el?.fields ?? []) {
        const b = f.binding as { kind?: string; offset?: number; size?: number } | undefined;
        if (b?.kind === "uniform") size = Math.max(size, (b.offset ?? 0) + (b.size ?? 0));
    }
    return Math.ceil(size / 16) * 16;
}

export class ParameterBlock {
    private slots = new Map<string, Slot>();
    private groups = new Map<number, { layout: GPUBindGroupLayout; bindGroup: GPUBindGroup | null; generation: number }>();
    private generation = 0;

    constructor(
        public readonly device: Device,
        public readonly reflection: ProgramReflection,
        public readonly wgslBindings: WgslBinding[],
    ) {
        for (const wb of wgslBindings) {
            const name = demangle(wb.name);
            const reflectionVar = reflection.findParameter(name) ?? null;
            if (wb.layoutEntry.buffer?.type === "uniform" && reflectionVar) {
                const size = computeStructSize(reflectionVar.type);
                const cpuData = new ArrayBuffer(size);
                this.slots.set(name, {
                    kind: "cbuffer",
                    binding: wb,
                    reflectionVar,
                    cpuData,
                    view: new DataView(cpuData),
                    gpuBuffer: new Buffer(device, { size, bindFlags: ResourceBindFlags.Constant, memoryType: MemoryType.DeviceLocal, name: `cb:${name}` }),
                    dirty: true,
                });
            } else {
                this.slots.set(name, { kind: "resource", binding: wb, reflectionVar, resource: null });
            }
        }
        const groupIndices = new Set(wgslBindings.map((b) => b.group));
        for (const g of groupIndices) {
            const layout = device.gpuDevice.createBindGroupLayout({
                entries: wgslBindings.filter((b) => b.group === g).map((b) => b.layoutEntry),
            });
            this.groups.set(g, { layout, bindGroup: null, generation: -1 });
        }
    }

    getSlotNames(): string[] {
        return [...this.slots.keys()];
    }

    /**
     * Mirrors ShaderVar resource assignment. Parameters present in program
     * reflection but statically unused by this kernel's WGSL are accepted as
     * no-ops (Falcor reflection is program-level; backends bind what's used).
     */
    setResource(name: string, resource: BindableResource): void {
        const slot = this.slots.get(name);
        if (!slot) {
            if (this.reflection.findParameter(name)) return;
            throw new ArgumentError(`No shader parameter named '${name}'`);
        }
        if (slot.kind !== "resource") throw new ArgumentError(`'${name}' is a constant buffer, not a resource`);
        slot.resource = resource;
        this.generation++;
    }

    /** Writes a uniform value at a member path inside a cbuffer. */
    setUniform(cbufferName: string, memberPath: string[], value: unknown): void {
        const slot = this.slots.get(cbufferName);
        if (!slot && this.reflection.findParameter(cbufferName)) return;
        if (!slot || slot.kind !== "cbuffer") throw new ArgumentError(`No constant buffer named '${cbufferName}'`);
        let v: ReflectionVar | undefined = slot.reflectionVar;
        for (const part of memberPath) {
            v = v.findMember(part);
            if (!v) throw new ArgumentError(`No member '${memberPath.join(".")}' in cbuffer '${cbufferName}'`);
        }
        this.writeValue(slot, v, value);
        slot.dirty = true;
        this.generation++;
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
                default: throw new RuntimeError(`Unsupported scalar type '${scalarType}' (f16/f64 handled in M3)`);
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
            // Host convention is row-major (Falcor). Slang's WGSL emission stores
            // HLSL floatRxC as an array of R vec4-aligned rows and swaps mul()
            // operand order itself — so bytes go out row-major with 16-byte row
            // stride (GPU-verified against mul(M, v) for float4x4; the _ColMajor
            // suffix in emitted type names refers to the transposed WGSL-side
            // matrix dims, not the byte order).
            const maybe = value as { toArray?: () => Float32Array };
            const arr = (typeof maybe?.toArray === "function" ? maybe.toArray() : value) as ArrayLike<number>;
            const rows = type.rowCount ?? 4;
            const cols = type.columnCount ?? 4;
            if (arr.length !== rows * cols) {
                throw new ArgumentError(`Expected ${rows * cols} floats for '${member.name}' (float${rows}x${cols}), got ${arr.length}`);
            }
            // Element layout by size signature: square/row-count-many elements store
            // rows (GPU-verified for float4x4); otherwise elements are columns
            // (e.g. float3x4 -> 4 elements x 16B = 64B, columns of 3).
            const size = member.byteSize > 0 ? member.byteSize : rows * 16;
            if (size === rows * 16) {
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) write(offset + r * 16 + c * 4, "float32", arr[r * cols + c]!);
                }
            } else if (size === cols * 16) {
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) write(offset + c * 16 + r * 4, "float32", arr[r * cols + c]!);
                }
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
            for (const slot of this.slots.values()) {
                if (slot.binding.group !== group) continue;
                if (slot.kind === "cbuffer") {
                    entries.push({ binding: slot.binding.binding, resource: { buffer: slot.gpuBuffer.gpuBuffer } });
                } else {
                    if (!slot.resource) throw new RuntimeError(`Shader parameter '${demangle(slot.binding.name)}' is not bound`);
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
            return slot.binding.layoutEntry.storageTexture ? r.getUAV() : r.getSRV();
        }
        if (r instanceof Sampler) return r.gpuSampler;
        return r; // GPUTextureView
    }
}

/**
 * ShaderVar mirroring Falcor/Core/Program/ShaderVar.h: proxy-based path access
 * (rootVar.PerFrame.gFrameIndex = 5; rootVar.gOutput = buffer).
 */
export type ShaderVar = {
    [key: string]: any;
};

export function makeRootVar(block: ParameterBlock): ShaderVar {
    const makeMemberProxy = (cbufferName: string, path: string[]): any =>
        new Proxy(Object.create(null), {
            get: (_t, prop: string) => makeMemberProxy(cbufferName, [...path, prop]),
            set: (_t, prop: string, value) => {
                block.setUniform(cbufferName, [...path, prop], value);
                return true;
            },
        });

    return new Proxy(Object.create(null), {
        get: (_t, prop: string) => {
            const slotNames = block.getSlotNames();
            if (slotNames.includes(prop)) return makeMemberProxy(prop, []);
            // Allow root.cbuffer.member without naming the cbuffer: search all cbuffers (Falcor's implicit global block).
            return makeMemberProxy(prop, []);
        },
        set: (_t, prop: string, value) => {
            block.setResource(prop, value);
            return true;
        },
    });
}
