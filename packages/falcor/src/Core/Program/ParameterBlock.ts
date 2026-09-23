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

/** Slot key of Slang's implicit buffer for loose global uniforms. */
const kGlobalParamsKey = "$globalParams";

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
    // Fields mixing resources and uniforms expose a `bindings` array instead
    // of a single `binding` (same shape as in ReflectionVar.findMember).
    return (type.fields ?? []).some(
        (f) =>
            (f.binding as { kind?: string } | undefined)?.kind === "uniform" ||
            (f as { bindings?: { kind: string }[] }).bindings?.some((b) => b.kind === "uniform"),
    );
}

export class ParameterBlock {
    /** Slots keyed by dot-joined path ("gOut", "gScene", "gScene.vertices.data0"). */
    private slots = new Map<string, Slot>();
    private wgslByName = new Map<string, WgslBinding>();
    private groups = new Map<number, { layout: GPUBindGroupLayout; bindGroup: GPUBindGroup | null; generation: number }>();
    private generation = 0;
    /** Reflection element types for cbuffer-slot paths (uniform member lookup). */
    private topLevel = new Map<string, ReflectionVar>();

    /** This block's own copy of the bindings (retargetStorageFormats edits them; kernels are shared). */
    public readonly wgslBindings: WgslBinding[];

    constructor(
        public readonly device: Device,
        public readonly reflection: ProgramReflection,
        wgslBindings: WgslBinding[],
    ) {
        this.wgslBindings = wgslBindings = wgslBindings.map((b) => ({ ...b, layoutEntry: structuredClone(b.layoutEntry) }));
        for (const wb of wgslBindings) this.wgslByName.set(demangle(wb.name), wb);
        for (const p of reflection.json.parameters ?? []) {
            this.topLevel.set(p.name, new ReflectionVar(p.name, p.type ?? { kind: "unknown" }, null));
            this.registerParameter(p, [p.name]);
        }
        // Loose global uniforms (`uniform uint g_count;`) live in Slang's implicit globalParams buffer.
        const globals = (reflection.json.parameters ?? []).filter((p) => (p.binding as { kind?: string } | undefined)?.kind === "uniform");
        const globalsBinding = this.lookupWgsl("globalParams");
        if (globals.length > 0 && globalsBinding) {
            const size = Math.max(...globals.map((p) => ((p.binding as { offset?: number }).offset ?? 0) + ((p.binding as { size?: number }).size ?? 0)));
            this.addCBufferSlot(kGlobalParamsKey, globalsBinding, { kind: "struct", fields: globals as never }, size);
        }

        const groupIndices = new Set(wgslBindings.map((b) => b.group));
        for (const g of groupIndices) {
            const layout = device.gpuDevice.createBindGroupLayout({
                entries: wgslBindings.filter((b) => b.group === g).map((b) => b.layoutEntry),
            });
            this.groups.set(g, { layout, bindGroup: null, generation: -1 });
        }

        // Every WGSL binding must have a slot, or the bind group can never be
        // completed (layout counts all bindings). Unmatched bindings indicate a
        // reflection-path mapping gap — surface them loudly.
        const slotBindings = new Set([...this.slots.values()].map((s) => s.binding));
        for (const wb of wgslBindings) {
            if (!slotBindings.has(wb)) {
                console.error(`ParameterBlock: WGSL binding '${wb.name}' (group ${wb.group}, binding ${wb.binding}) has no reflection match; bind group will be incomplete`);
            }
        }
    }

    /** Fields with leading underscores (e.g. `_emissivePower`) collapse in
     *  Slang's WGSL name mangling: `a__b` emits as `a_b`. */
    private lookupWgsl(flatName: string): WgslBinding | undefined {
        return this.wgslByName.get(flatName) ?? this.wgslByName.get(flatName.replace(/_+/g, "_"));
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
            const wb = this.lookupWgsl(flatName);
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

        const wb = this.lookupWgsl(flatName);
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

    /** Mirrors reading a resource back through a ShaderVar (`ref<Buffer> b = var["x"]`). */
    getResourceByPath(path: string[]): BindableResource | null {
        const slot = this.slots.get(path.join("."));
        return slot?.kind === "resource" ? slot.resource : null;
    }

    /** Uniform write: longest slot prefix is the containing cbuffer; rest addresses members. */
    setUniformByPath(path: string[], value: unknown): void {
        for (let prefixLen = path.length - 1; prefixLen >= 0; prefixLen--) {
            // Prefix length 0: a loose global uniform, in the implicit globalParams buffer.
            const key = prefixLen === 0 ? kGlobalParamsKey : path.slice(0, prefixLen).join(".");
            const slot = this.slots.get(key);
            if (prefixLen === 0 && slot?.kind === "cbuffer" && !slot.elementType.fields?.some((f) => f.name === path[0])) break;
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
            throw new RuntimeError(`Setting '${type.kind}' uniforms not implemented yet (member '${member.name}')`);
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

    /**
     * Native UAVs take their resource's format; WGSL storage textures declare one (Slang infers
     * it from the element type, e.g. rgba32float for float4). Moves storage-texture bindings to
     * the bound textures' formats, within the same texel type (float/uint/sint), and rebuilds the
     * layouts. Returns the retargeted binding names and formats, or null when nothing changed.
     */
    retargetStorageFormats(): Map<string, GPUTextureFormat> | null {
        const texelType = (f: string) => (f.endsWith("uint") ? "u" : f.endsWith("sint") ? "i" : "f");
        const changed = new Map<string, GPUTextureFormat>();
        for (const slot of this.slots.values()) {
            if (slot.kind !== "resource" || !(slot.resource instanceof Texture)) continue;
            const st = slot.binding.layoutEntry.storageTexture;
            const format = slot.resource.gpuFormat;
            if (!st || !st.format || st.format === format || texelType(st.format) !== texelType(format)) continue;
            if (st.access === "read-write" && !/^r32(float|uint|sint)$/.test(format)) continue;
            st.format = format;
            changed.set(slot.binding.name, format);
        }
        if (changed.size === 0) return null;
        for (const [g, entry] of this.groups) {
            entry.layout = this.device.gpuDevice.createBindGroupLayout({ entries: this.wgslBindings.filter((b) => b.group === g).map((b) => b.layoutEntry) });
            entry.bindGroup = null;
        }
        this.generation++;
        return changed;
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

/**
 * Mirrors ParameterBlock::create(device, reflection): a block filled on its own and
 * bound later with `var["gBlock"] = block`. §9: program parameter blocks are laid
 * out per program here, so the block records its values and forwards them to every
 * program it is bound to (later writes included, as native blocks are shared).
 */
export class StandaloneParameterBlock {
    private values = new Map<string, { path: string[]; value: unknown }>();
    private targets: { block: ParameterBlock; prefix: string[] }[] = [];
    private readonly root: ReflectionVar;

    private constructor(
        public readonly device: Device,
        reflection: ReflectionVar,
    ) {
        this.root = reflection;
    }

    /** `reflection` is a ParameterBlock<T> parameter, e.g. `getReflector().getParameterBlock("gBlock")`. */
    static create(device: Device, reflection: ReflectionVar): StandaloneParameterBlock {
        if (reflection.type.kind !== "parameterBlock") throw new ArgumentError(`'${reflection.name}' is not a parameter block`);
        return new StandaloneParameterBlock(device, reflection);
    }

    getRootVar(): ShaderVar {
        const makeProxy = (path: string[]): any =>
            new Proxy(Object.create(null), {
                get: (_t, prop: string) => (prop === "getBuffer" || prop === "getTexture" || prop === "getResource" ? () => this.get(path) : makeProxy([...path, prop])),
                set: (_t, prop: string, value) => (this.set([...path, prop], value), true),
            });
        return makeProxy([]);
    }

    setBuffer(name: string, buffer: Buffer): void {
        this.set([name], buffer);
    }

    getBuffer(name: string): Buffer | null {
        const v = this.get([name]);
        return v instanceof Buffer ? v : null;
    }

    private get(path: string[]): unknown {
        return this.values.get(path.join("."))?.value ?? null;
    }

    private set(path: string[], value: unknown): void {
        let v: ReflectionVar | undefined = this.root;
        for (const part of path) if (!(v = v.findMember(part))) throw new ArgumentError(`No member '${path.join(".")}' in parameter block '${this.root.name}'`);
        this.values.set(path.join("."), { path, value });
        for (const t of this.targets) write(t.block, [...t.prefix, ...path], value);
    }

    /** Binds the block's values at `prefix` of a program block (ShaderVar assignment). */
    bindTo(block: ParameterBlock, prefix: string[]): void {
        if (!this.targets.some((t) => t.block === block && t.prefix.join(".") === prefix.join("."))) this.targets.push({ block, prefix });
        for (const { path, value } of this.values.values()) write(block, [...prefix, ...path], value);
    }
}

function write(block: ParameterBlock, path: string[], value: unknown): void {
    if (isBindable(value)) block.setResourceByPath(path, value);
    else block.setUniformByPath(path, value);
}

function isBindable(value: unknown): value is BindableResource {
    return value instanceof Buffer || value instanceof Texture || value instanceof Sampler || (typeof GPUTextureView !== "undefined" && value instanceof GPUTextureView);
}

export function makeRootVar(block: ParameterBlock): ShaderVar {
    const makeProxy = (path: string[]): any =>
        new Proxy(Object.create(null), {
            get: (_t, prop: string) => (prop === "getBuffer" || prop === "getTexture" || prop === "getResource" ? () => block.getResourceByPath(path) : makeProxy([...path, prop])),
            set: (_t, prop: string, value) => {
                const full = [...path, prop];
                if (value instanceof StandaloneParameterBlock) value.bindTo(block, full);
                else write(block, full, value);
                return true;
            },
        });
    return makeProxy([]);
}
