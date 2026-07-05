/**
 * Program reflection mirroring Falcor/Core/Program/ProgramReflection.h.
 *
 * Wraps Slang's reflection JSON (ProgramLayout.toJsonObject()) into a typed
 * tree used by ParameterBlock/ShaderVar for offset lookup and bind-group
 * construction. Binding model: Slang 'descriptorTableSlot' index -> @binding,
 * space -> @group (WGSL emission and reflection are consistent by construction).
 */

import type { SlangReflectionJson, SlangReflectionParameter, SlangReflectionType } from "./SlangCompiler.js";
import { RuntimeError } from "../Error.js";

export interface BindingInfo {
    /** @binding index within the group. */
    index: number;
    /** @group index. */
    space: number;
    kind: string; // descriptorTableSlot | uniform | ...
}

/** One shader-visible resource or cbuffer parameter. */
export class ReflectionVar {
    constructor(
        public readonly name: string,
        public readonly type: SlangReflectionType,
        public readonly binding: BindingInfo | null,
        /** Byte offset for uniform-kind members inside their parent cbuffer. */
        public readonly byteOffset = 0,
        public readonly byteSize = 0,
    ) {}

    /** Struct member lookup (constantBuffer elementType or plain struct). */
    findMember(name: string): ReflectionVar | undefined {
        const struct = this.getStructType();
        const field = struct?.fields?.find((f) => f.name === name);
        if (!field) return undefined;
        const b = field.binding as { kind: string; offset?: number; size?: number } | undefined;
        return new ReflectionVar(
            field.name,
            field.type as SlangReflectionType,
            null,
            this.byteOffset + (b?.offset ?? 0),
            b?.size ?? 0,
        );
    }

    private getStructType(): SlangReflectionType | undefined {
        if (this.type.kind === "struct") return this.type;
        if (this.type.kind === "constantBuffer" || this.type.kind === "parameterBlock") {
            const el = this.type.elementType;
            if (el?.kind === "struct") return el;
        }
        return undefined;
    }
}

export interface EntryPointReflection {
    name: string;
    stage: string;
    threadGroupSize: [number, number, number];
}

export class ProgramReflection {
    readonly parameters: ReflectionVar[] = [];
    readonly entryPoints: EntryPointReflection[] = [];

    constructor(public readonly json: SlangReflectionJson) {
        for (const p of json.parameters ?? []) {
            this.parameters.push(ProgramReflection.makeVar(p));
        }
        for (const ep of json.entryPoints ?? []) {
            const tgs = (ep.threadGroupSize ?? [1, 1, 1]) as number[];
            this.entryPoints.push({
                name: ep.name,
                stage: ep.stage ?? "compute",
                threadGroupSize: [tgs[0] ?? 1, tgs[1] ?? 1, tgs[2] ?? 1],
            });
        }
    }

    private static makeVar(p: SlangReflectionParameter): ReflectionVar {
        const b = p.binding;
        const binding: BindingInfo | null =
            b && b.kind === "descriptorTableSlot"
                ? { index: b.index ?? 0, space: b.space ?? 0, kind: b.kind }
                : b
                  ? { index: b.index ?? 0, space: b.space ?? 0, kind: b.kind }
                  : null;
        return new ReflectionVar(p.name, p.type ?? { kind: "unknown" }, binding, b?.offset ?? 0, b?.size ?? 0);
    }

    findParameter(name: string): ReflectionVar | undefined {
        return this.parameters.find((p) => p.name === name);
    }

    getEntryPoint(name: string): EntryPointReflection {
        const ep = this.entryPoints.find((e) => e.name === name) ?? this.entryPoints[0];
        if (!ep) throw new RuntimeError(`Entry point '${name}' not found in reflection`);
        return ep;
    }
}

/** A WGSL-declared binding parsed from emitted code (used for explicit bind group layouts). */
export interface WgslBinding {
    group: number;
    binding: number;
    name: string;
    layoutEntry: GPUBindGroupLayoutEntry;
}

/**
 * Parses @group/@binding declarations out of Slang-emitted WGSL to build explicit
 * bind-group layouts. Explicit layouts (not 'auto') mirror Falcor's reflection-built
 * root signatures and avoid WebGPU auto-layout's unused-binding stripping.
 */
export function parseWgslBindings(wgsl: string, visibility: GPUShaderStageFlags): WgslBinding[] {
    const bindings: WgslBinding[] = [];
    const re = /@binding\((\d+)\)\s*@group\((\d+)\)\s*var\s*(?:<([^>]+)>)?\s*([A-Za-z0-9_]+)\s*:\s*([^;]+);/g;
    for (const m of wgsl.matchAll(re)) {
        const [, bindingStr, groupStr, addressSpace, name, typeStrRaw] = m;
        const typeStr = typeStrRaw!.trim();
        const entry: GPUBindGroupLayoutEntry = { binding: Number(bindingStr), visibility };

        if (addressSpace?.startsWith("uniform")) {
            entry.buffer = { type: "uniform" };
        } else if (addressSpace?.startsWith("storage")) {
            const readOnly = !addressSpace.includes("read_write");
            entry.buffer = { type: readOnly ? "read-only-storage" : "storage" };
        } else if (typeStr.startsWith("texture_storage_")) {
            const fm = /texture_storage_(\w+)<([a-z0-9]+),\s*(\w+)>/.exec(typeStr);
            if (!fm) continue;
            const [, dim, format, access] = fm;
            entry.storageTexture = {
                access: access === "read" ? "read-only" : access === "read_write" ? "read-write" : "write-only",
                format: format as GPUTextureFormat,
                viewDimension: wgslDimToView(dim!),
            };
        } else if (typeStr.startsWith("texture_depth")) {
            const dim = /texture_depth_([a-z0-9_]+)/.exec(typeStr)?.[1] ?? "2d";
            entry.texture = { sampleType: "depth", viewDimension: wgslDimToView(dim.replace(/_multisampled/, "")) };
        } else if (typeStr.startsWith("texture_")) {
            const tm = /texture(?:_multisampled)?_(\w+)<([a-z0-9]+)>/.exec(typeStr);
            const dim = tm?.[1] ?? "2d";
            const scalar = tm?.[2] ?? "f32";
            entry.texture = {
                sampleType: scalar === "f32" ? "float" : scalar === "u32" ? "uint" : scalar === "i32" ? "sint" : "float",
                viewDimension: wgslDimToView(dim),
                multisampled: typeStr.includes("multisampled"),
            };
        } else if (typeStr.startsWith("sampler_comparison")) {
            entry.sampler = { type: "comparison" };
        } else if (typeStr.startsWith("sampler")) {
            entry.sampler = { type: "filtering" };
        } else {
            // Unhandled binding type; skip (acceleration structures etc. don't exist in WGSL).
            continue;
        }
        bindings.push({ group: Number(groupStr), binding: Number(bindingStr), name: name!, layoutEntry: entry });
    }
    return bindings;
}

/**
 * Merges per-entry-point WGSL binding lists (e.g. VS + PS of one program) into
 * one list with OR'd visibility — linked Slang programs emit consistent
 * @group/@binding assignments across entry points.
 */
export function mergeWgslBindings(...lists: WgslBinding[][]): WgslBinding[] {
    const merged = new Map<string, WgslBinding>();
    for (const list of lists) {
        for (const b of list) {
            const key = `${b.group}:${b.binding}`;
            const existing = merged.get(key);
            if (existing) {
                existing.layoutEntry.visibility |= b.layoutEntry.visibility;
            } else {
                merged.set(key, { ...b, layoutEntry: { ...b.layoutEntry } });
            }
        }
    }
    return [...merged.values()];
}

function wgslDimToView(dim: string): GPUTextureViewDimension {
    switch (dim) {
        case "1d": return "1d";
        case "2d": return "2d";
        case "2d_array": return "2d-array";
        case "3d": return "3d";
        case "cube": return "cube";
        case "cube_array": return "cube-array";
        default: return "2d";
    }
}
