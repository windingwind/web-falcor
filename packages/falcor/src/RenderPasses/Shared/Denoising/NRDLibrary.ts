/**
 * The NRD 3.1.0 library (wasm/nrd.{mjs,wasm}, built by scripts/build-nrd-wasm.mjs) behind the
 * nrd:: calls NRDPass makes. Each instance loads its own module, so several NRD passes keep
 * separate denoisers. Settings structs live in wasm memory and are edited by field name
 * through the layouts the wrapper reports (offsetof in the wasm build).
 */

import { RuntimeError } from "../../../Core/Error.js";

/** Mirrors nrd::Method. */
export enum NRDMethod {
    REBLUR_DIFFUSE,
    REBLUR_DIFFUSE_OCCLUSION,
    REBLUR_SPECULAR,
    REBLUR_SPECULAR_OCCLUSION,
    REBLUR_DIFFUSE_SPECULAR,
    REBLUR_DIFFUSE_SPECULAR_OCCLUSION,
    REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION,
    SIGMA_SHADOW,
    SIGMA_SHADOW_TRANSLUCENCY,
    RELAX_DIFFUSE,
    RELAX_SPECULAR,
    RELAX_DIFFUSE_SPECULAR,
    REFERENCE,
    SPECULAR_REFLECTION_MV,
    SPECULAR_DELTA_MV,
}

/** Mirrors nrd::ResourceType. */
export enum NRDResourceType {
    IN_MV,
    IN_NORMAL_ROUGHNESS,
    IN_VIEWZ,
    IN_DIFF_RADIANCE_HITDIST,
    IN_SPEC_RADIANCE_HITDIST,
    IN_DIFF_HITDIST,
    IN_SPEC_HITDIST,
    IN_DIFF_DIRECTION_HITDIST,
    IN_DIFF_DIRECTION_PDF,
    IN_SPEC_DIRECTION_PDF,
    IN_DIFF_CONFIDENCE,
    IN_SPEC_CONFIDENCE,
    IN_SHADOWDATA,
    IN_SHADOW_TRANSLUCENCY,
    IN_RADIANCE,
    IN_DELTA_PRIMARY_POS,
    IN_DELTA_SECONDARY_POS,
    OUT_DIFF_RADIANCE_HITDIST,
    OUT_SPEC_RADIANCE_HITDIST,
    OUT_DIFF_HITDIST,
    OUT_SPEC_HITDIST,
    OUT_DIFF_DIRECTION_HITDIST,
    OUT_SHADOW_TRANSLUCENCY,
    OUT_RADIANCE,
    OUT_REFLECTION_MV,
    OUT_DELTA_MV,
    TRANSIENT_POOL,
    PERMANENT_POOL,
}

/** Mirrors nrd::DescriptorType. */
export enum NRDDescriptorType {
    TEXTURE,
    STORAGE_TEXTURE,
}

/** Mirrors nrd::Sampler. */
export enum NRDSampler {
    NEAREST_CLAMP,
    NEAREST_MIRRORED_REPEAT,
    LINEAR_CLAMP,
    LINEAR_MIRRORED_REPEAT,
}

export interface NRDTextureDesc {
    format: number;
    width: number;
    height: number;
    mipNum: number;
}

/** Mirrors nrd::DenoiserDesc (pipelines with their descriptor ranges, samplers, pools). */
export interface NRDDenoiserDesc {
    pipelines: { shader: string; entry: string; hasConstantData: boolean; ranges: { type: NRDDescriptorType; base: number; count: number }[] }[];
    samplers: { sampler: NRDSampler; register: number }[];
    permanentPool: NRDTextureDesc[];
    transientPool: NRDTextureDesc[];
    constantBuffer: { maxDataSize: number; register: number };
}

/** Mirrors nrd::Resource. */
export interface NRDResource {
    stateNeeded: NRDDescriptorType;
    type: NRDResourceType;
    indexInPool: number;
    mipOffset: number;
    mipNum: number;
}

/** Mirrors nrd::DispatchDesc (constants copied out of wasm memory). */
export interface NRDDispatch {
    name: string;
    pipelineIndex: number;
    gridWidth: number;
    gridHeight: number;
    constants: Uint8Array;
    resources: NRDResource[];
}

type FieldType = "f32" | "u32" | "u8" | "bool";
type Layout = { size: number; fields: [name: string, offset: number, type: FieldType, count: number][] };

interface NRDModule {
    HEAPU8: Uint8Array;
    UTF8ToString(ptr: number): string;
    stringToNewUTF8(s: string): number;
    _free(ptr: number): void;
    _nrdw_fields(): number;
    _nrdw_settings(name: number): number;
    _nrdw_library(): number;
    _nrdw_create(method: number, width: number, height: number): number;
    _nrdw_set_method_settings(method: number): number;
    _nrdw_denoiser_desc(): number;
    _nrdw_dispatches(): number;
    _nrdw_destroy(): void;
}

/** A settings struct in wasm memory, read and written by field name. */
export class NRDSettings {
    private readonly byName = new Map<string, Layout["fields"][number]>();

    constructor(
        private readonly module: NRDModule,
        private readonly ptr: number,
        readonly layout: Layout,
    ) {
        for (const f of layout.fields) this.byName.set(f[0], f);
    }

    has(name: string): boolean {
        return this.byName.has(name);
    }

    get fieldNames(): string[] {
        return [...this.byName.keys()];
    }

    get(name: string): number | boolean | number[] {
        const f = this.byName.get(name);
        if (!f) throw new RuntimeError(`NRD settings: no field '${name}'`);
        const values = Array.from({ length: f[3] }, (_, i) => this.read(f[1] + i * this.width(f[2]), f[2]));
        return f[3] === 1 ? values[0]! : (values as number[]);
    }

    set(name: string, value: number | boolean | ArrayLike<number>): void {
        const f = this.byName.get(name);
        if (!f) throw new RuntimeError(`NRD settings: no field '${name}'`);
        const values = typeof value === "object" ? Array.from(value) : [value];
        values.forEach((v, i) => this.write(f[1] + i * this.width(f[2]), f[2], v));
    }

    private width(t: FieldType): number {
        return t === "u8" || t === "bool" ? 1 : 4;
    }

    private read(offset: number, t: FieldType): number | boolean {
        const view = new DataView(this.module.HEAPU8.buffer);
        const at = this.ptr + offset;
        switch (t) {
            case "f32": return view.getFloat32(at, true);
            case "u32": return view.getUint32(at, true);
            case "u8": return view.getUint8(at);
            default: return view.getUint8(at) !== 0;
        }
    }

    private write(offset: number, t: FieldType, v: number | boolean): void {
        const view = new DataView(this.module.HEAPU8.buffer);
        const at = this.ptr + offset;
        const n = typeof v === "boolean" ? (v ? 1 : 0) : v;
        switch (t) {
            case "f32": view.setFloat32(at, n, true); break;
            case "u32": view.setUint32(at, n >>> 0, true); break;
            default: view.setUint8(at, n); break;
        }
    }
}

export class NRDLibrary {
    readonly version: [number, number, number];
    private readonly layouts: Record<string, Layout>;

    private constructor(private readonly module: NRDModule) {
        this.layouts = JSON.parse(module.UTF8ToString(module._nrdw_fields())) as Record<string, Layout>;
        this.version = (JSON.parse(module.UTF8ToString(module._nrdw_library())) as { version: [number, number, number] }).version;
    }

    /** Loads a fresh library instance (its own denoiser and settings). */
    static async load(): Promise<NRDLibrary> {
        const url = new URL("../../../../wasm/nrd.mjs", import.meta.url).href;
        const { default: factory } = (await import(/* @vite-ignore */ url)) as { default: (opts: object) => Promise<NRDModule> };
        const module = await factory({ locateFile: (f: string) => new URL(`../../../../wasm/${f}`, import.meta.url).href });
        return new NRDLibrary(module);
    }

    /** One of CommonSettings, RelaxDiffuseSpecularSettings, RelaxDiffuseSettings, ReblurSettings, SpecularReflectionMvSettings, SpecularDeltaMvSettings. */
    settings(struct: string): NRDSettings {
        const layout = this.layouts[struct];
        if (!layout) throw new RuntimeError(`NRD: unknown settings struct '${struct}'`);
        const name = this.module.stringToNewUTF8(struct);
        const ptr = this.module._nrdw_settings(name);
        this.module._free(name);
        return new NRDSettings(this.module, ptr, layout);
    }

    /** Mirrors nrd::CreateDenoiser with one method. */
    createDenoiser(method: NRDMethod, width: number, height: number): void {
        const res = this.module._nrdw_create(method, width, height);
        if (res !== 0) throw new RuntimeError(`NRD: failed to create the denoiser (nrd::Result ${res})`);
    }

    /** Mirrors nrd::SetMethodSettings with this library's settings struct for `method`. */
    setMethodSettings(method: NRDMethod): void {
        const res = this.module._nrdw_set_method_settings(method);
        if (res !== 0) throw new RuntimeError(`NRD: SetMethodSettings failed (${res})`);
    }

    /** Mirrors nrd::GetDenoiserDesc. */
    getDenoiserDesc(): NRDDenoiserDesc {
        return JSON.parse(this.module.UTF8ToString(this.module._nrdw_denoiser_desc())) as NRDDenoiserDesc;
    }

    /** Mirrors nrd::GetComputeDispatches with the CommonSettings struct. */
    getComputeDispatches(): NRDDispatch[] {
        const raw = JSON.parse(this.module.UTF8ToString(this.module._nrdw_dispatches())) as
            | { error: number }
            | { name: string; pipeline: number; grid: [number, number]; constants: [number, number]; resources: [number, number, number, number, number][] }[];
        if (!Array.isArray(raw)) throw new RuntimeError(`NRD: GetComputeDispatches failed (${raw.error})`);
        return raw.map((d) => ({
            name: d.name,
            pipelineIndex: d.pipeline,
            gridWidth: d.grid[0],
            gridHeight: d.grid[1],
            constants: this.module.HEAPU8.slice(d.constants[0], d.constants[0] + d.constants[1]),
            resources: d.resources.map(([stateNeeded, type, indexInPool, mipOffset, mipNum]) => ({ stateNeeded, type, indexInPool, mipOffset, mipNum })),
        }));
    }

    destroy(): void {
        this.module._nrdw_destroy();
    }
}
