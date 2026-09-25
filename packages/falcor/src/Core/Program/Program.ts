/**
 * Program & ProgramManager mirroring Falcor/Core/Program/Program.h and
 * ProgramManager.h. A Program is (source path + entry points + defines);
 * ProgramVersion is the compiled artifact for one define-set: WGSL + shader
 * modules + reflection + parsed bindings.
 */

import { Device } from "../API/Device.js";
import { DefineList } from "./DefineList.js";
import { SlangCompiler, ShaderType, loadSlangRuntime, type SlangRuntime, parenthesizeNegations, type CompileModule, type ShaderSourceResolver, type EntryPointDesc } from "./SlangCompiler.js";
import { kShaderOverrides } from "./ShaderOverrides.js";
import { ProgramReflection, parseWgslBindings, type WgslBinding } from "./ProgramReflection.js";
import { RuntimeError } from "../Error.js";

/** Mirrors ProgramDesc::ShaderSource: a file (shader-root path) or a code string. */
export type ShaderSourceDesc = { file: string } | { string: string; path?: string };

/** Mirrors ProgramDesc::ShaderModule: one translation unit from files and strings; a named module can be imported by name. */
export interface ShaderModuleDesc {
    name?: string;
    sources: ShaderSourceDesc[];
}

export interface ProgramDesc {
    /**
     * Shader module path(s) relative to the shader root, e.g.
     * "RenderPasses/ToneMapper/ToneMapper.cs.slang". Multiple modules mirror
     * Falcor's multi-translation-unit programs; entry points reference modules
     * via moduleIndex.
     */
    path?: string | string[];
    /** Modules composed of files and strings (ProgramDesc::addShaderModule); appended after `path`. */
    modules?: ShaderModuleDesc[];
    /** Mirrors ProgramDesc::addTypeConformances: IDs for createDynamicObject<Interface, T>(id, data). */
    typeConformances?: TypeConformance[];
    entryPoints: EntryPointDesc[];
    /**
     * slang-wasm build to compile with (URL of its slang-wasm.js) instead of the default; it must
     * be loaded first with ProgramManager.loadSlangRuntime. For kernels a newer Slang breaks.
     */
    slangRuntime?: string;
}

/** Mirrors TypeConformance + its conformance ID. */
export interface TypeConformance {
    typeName: string;
    interfaceName: string;
    id: number;
}

export interface EntryPointKernel {
    name: string;
    type: ShaderType;
    wgsl: string;
    module: GPUShaderModule;
    bindings: WgslBinding[];
}

/** Flattens a ProgramDesc into the compiler's module list. */
function programModules(desc: ProgramDesc): CompileModule[] {
    const paths = desc.path === undefined ? [] : typeof desc.path === "string" ? [desc.path] : desc.path;
    const out: CompileModule[] = paths.map((path) => ({ path, sources: [{ file: path }] }));
    (desc.modules ?? []).forEach((m, i) => {
        const first = m.sources[0];
        const path = first && "file" in first ? first.file : (first && "string" in first && first.path) || `${m.name ?? `ShaderStringModule${i}`}.slang`;
        out.push({ path, name: m.name, sources: m.sources });
    });
    return out;
}

/** Compiled program for one define-set (mirrors Falcor::ProgramVersion/ProgramKernels). */
export class ProgramVersion {
    constructor(
        public readonly reflection: ProgramReflection,
        public readonly kernels: EntryPointKernel[],
    ) {}

    getKernel(name: string, type?: ShaderType): EntryPointKernel {
        const k = this.kernels.find((k) => k.name === name && (type === undefined || k.type === type));
        if (!k) throw new RuntimeError(`No kernel '${name}'${type !== undefined ? ` of stage ${ShaderType[type]}` : ""} in program version`);
        return k;
    }
}

export class Program {
    private versions = new Map<string, ProgramVersion>();
    /** Shader generation the cached versions were compiled in. */
    private generation = 0;

    constructor(
        public readonly device: Device,
        public readonly desc: ProgramDesc,
        public readonly defines: DefineList,
    ) {}

    /** Mirrors Program::getActiveVersion: compile-on-miss per define-set. */
    getActiveVersion(): ProgramVersion {
        const manager = this.device.programManager;
        // A shader reload invalidates everything compiled before it.
        if (manager.generation !== this.generation) {
            this.versions.clear();
            this.generation = manager.generation;
        }
        const key = this.defines.key();
        let version = this.versions.get(key);
        if (!version) {
            version = manager.compileProgram(this.desc, this.defines);
            this.versions.set(key, version);
        }
        return version;
    }

    addDefine(name: string, value: string | number | boolean = ""): this {
        this.defines.add(name, value);
        return this;
    }

    removeDefine(name: string): this {
        this.defines.remove(name);
        return this;
    }

    /** Mirrors Program::setDefines: replaces the define set. */
    setDefines(defines: DefineList | Record<string, string | number | boolean>): this {
        this.defines.clear();
        this.defines.addAll(defines);
        return this;
    }

    /** Mirrors Program::getTypeConformances. */
    getTypeConformances(): TypeConformance[] {
        return [...(this.desc.typeConformances ?? [])];
    }

    /** Mirrors Program::setTypeConformances; conformances are part of the compile, so versions are dropped. */
    setTypeConformances(conformances: TypeConformance[]): this {
        this.desc.typeConformances = [...conformances];
        this.versions.clear();
        return this;
    }

    /** Mirrors Program::addTypeConformance: an already listed type/interface pair keeps its ID. */
    addTypeConformance(typeName: string, interfaceName: string, id: number): this {
        const list = this.getTypeConformances();
        if (list.some((c) => c.typeName === typeName && c.interfaceName === interfaceName)) return this;
        return this.setTypeConformances([...list, { typeName, interfaceName, id }]);
    }

    /** Mirrors Program::removeTypeConformance. */
    removeTypeConformance(typeName: string, interfaceName: string): this {
        return this.setTypeConformances(this.getTypeConformances().filter((c) => c.typeName !== typeName || c.interfaceName !== interfaceName));
    }
}

/**
 * Post-emission WGSL fixups for known Slang emission defects:
 * - '@interpolate' is only valid on IO members with '@location'; Slang keeps it
 *   on internal copies of varying structs (Tint rejects them);
 * - conversely, integral entry-IO varyings lose their required
 *   '@interpolate(flat)' in the flattened IO structs;
 * - vector/matrix negation is emitted as `(vecN<T>(0) - x)` without parenthesizing x,
 *   so `-(a + b)` became `0 - a + b` (see parenthesizeNegations).
 */
function fixupWgsl(wgsl: string): string {
    return parenthesizeNegations(wgsl)
        .split("\n")
        .map((line) => {
            if (line.includes("@interpolate") && !line.includes("@location") && !line.includes("@builtin")) {
                return line.replace(/@interpolate\([a-z_, ]+\)\s*/g, "");
            }
            if (line.includes("@location") && !line.includes("@interpolate") && /:\s*(vec[234]<)?(u32|i32)>?\s*,?\s*$/.test(line)) {
                return line.replace("@location", "@interpolate(flat) @location");
            }
            return line;
        })
        .join("\n");
}

function shaderTypeToVisibility(type: ShaderType): GPUShaderStageFlags {
    switch (type) {
        case ShaderType.Compute: return GPUShaderStage.COMPUTE;
        case ShaderType.Vertex: return GPUShaderStage.VERTEX;
        case ShaderType.Pixel: return GPUShaderStage.FRAGMENT;
        default: return GPUShaderStage.COMPUTE;
    }
}

/** Mirrors Falcor::ProgramManager (device-owned program factory + cache + global defines). */
export class ProgramManager {
    readonly globalDefines = new DefineList();
    private compiler: SlangCompiler;
    /** Compilers over other slang-wasm builds, by URL (ProgramDesc.slangRuntime). */
    private altCompilers = new Map<string, { runtime: SlangRuntime; compiler: SlangCompiler }>();
    private resolveSource: ShaderSourceResolver;
    private filePaths: string[];
    /** Bumped by reloadAllPrograms; Programs drop cached versions when it moves. */
    private reloadGeneration = 0;

    constructor(
        private readonly device: Device,
        resolveSource: ShaderSourceResolver,
        filePaths: string[],
    ) {
        this.resolveSource = resolveSource;
        this.filePaths = filePaths;
        this.compiler = this.createCompiler();

        // f16 demotion: when the device lacks shader-f16, map half types to f32 at
        // the token level (WGSL 'enable f16' would fail validation otherwise).
        // Literals like 1.h constant-fold; f16tof32/f32tof16 lower to pack/unpack2x16float.
        // Divergence: f16 storage precision becomes f32 math with f16 rounding only at
        // pack boundaries. 16-bit index buffers are unsupported in this mode.
        if (!device.hasFeature("shader-f16")) {
            this.globalDefines.addAll({
                WEBFALCOR_NO_F16: 1,
                float16_t: "float",
                float16_t2: "float2",
                float16_t3: "float3",
                float16_t4: "float4",
            });
        }
        // WGSL has no 16-bit integers at all (PathState.rejectedHits etc.);
        // demote unconditionally — packing code masks to 16 bits already.
        this.globalDefines.addAll({
            uint16_t: "uint",
            uint16_t2: "uint2",
            uint16_t4: "uint4",
            int16_t: "int",
        });
    }

    /** Substitutes WGSL-incompatible upstream files with WebFalcor overrides (docs §4.3). */
    private createCompiler(runtime?: SlangRuntime): SlangCompiler {
        const resolveWithOverrides: ShaderSourceResolver = (path) => this.resolveSource(kShaderOverrides[path] ?? path);
        // Older builds lack ByteAddressBuffer.Load2/3/4 for WGSL; the generic Load<T> is equivalent.
        const transform = runtime ? (src: string) => src.replace(/\.Load([234])\(/g, ".Load<uint$1>(") : undefined;
        return new SlangCompiler(resolveWithOverrides, this.filePaths, runtime, transform);
    }

    /** Loads another slang-wasm build for programs that name it in ProgramDesc.slangRuntime. */
    async loadSlangRuntime(url: string): Promise<void> {
        if (this.altCompilers.has(url)) return;
        const runtime = await loadSlangRuntime(url);
        if (!this.altCompilers.has(url)) this.altCompilers.set(url, { runtime, compiler: this.createCompiler(runtime) });
    }

    /** The shader sources in use; wrap these to patch a file and reload. */
    getSourceProvider(): { resolveSource: ShaderSourceResolver; filePaths: string[] } {
        return { resolveSource: this.resolveSource, filePaths: this.filePaths };
    }

    /** Counter identifying the current shader generation (see reloadAllPrograms). */
    get generation(): number {
        return this.reloadGeneration;
    }

    /**
     * Mirrors ProgramManager::reloadAllPrograms: throws away every compiled
     * version so the next use recompiles from the current sources. Passes pick
     * the new kernels up on their next `getRootVar`/`execute`, which is where
     * render passes already re-bind, so a live render graph keeps running.
     *
     * The shader sources live in the app (it fetches them), so pass `source`
     * after re-fetching them; without it the existing resolver is re-read,
     * which is enough when the app mutates its own source map in place.
     */
    reloadAllPrograms(source?: { resolveSource: ShaderSourceResolver; filePaths: string[] }): void {
        if (source) {
            this.resolveSource = source.resolveSource;
            this.filePaths = source.filePaths;
        }
        this.compiler.dispose();
        this.compiler = this.createCompiler();
        for (const alt of this.altCompilers.values()) {
            alt.compiler.dispose();
            alt.compiler = this.createCompiler(alt.runtime);
        }
        this.reloadGeneration++;
    }

    /** Adds shader files (e.g. a script's own modules) so imports resolve them; reloads programs if any changed. */
    addShaderFiles(files: Record<string, string>): void {
        const changed = Object.entries(files).filter(([path, source]) => this.resolveSource(path) !== source);
        if (changed.length === 0) return;
        const extra = new Map(changed);
        const prev = this.resolveSource;
        const known = new Set(this.filePaths);
        this.reloadAllPrograms({ resolveSource: (path) => extra.get(path) ?? prev(path), filePaths: [...this.filePaths, ...[...extra.keys()].filter((p) => !known.has(p))] });
    }

    createProgram(desc: ProgramDesc, defines = new DefineList()): Program {
        return new Program(this.device, desc, defines);
    }

    compileProgram(desc: ProgramDesc, defines: DefineList): ProgramVersion {
        const allDefines = this.globalDefines.clone().addAll(defines);
        const alt = desc.slangRuntime ? this.altCompilers.get(desc.slangRuntime) : undefined;
        if (desc.slangRuntime && !alt) throw new RuntimeError(`Slang runtime ${desc.slangRuntime} not loaded (ProgramManager.loadSlangRuntime)`);
        const compiler = alt?.compiler ?? this.compiler;
        const result = compiler.compile(programModules(desc), desc.entryPoints, allDefines, desc.typeConformances ?? []);
        const kernels = desc.entryPoints.map((ep, i) => {
            const wgsl = fixupWgsl(result.entryPointCode[i]!);
            return {
                name: ep.name,
                type: ep.type,
                wgsl,
                module: this.device.gpuDevice.createShaderModule({ label: `${String(desc.path ?? desc.modules?.[0]?.name ?? "")}:${ep.name}`, code: wgsl }),
                bindings: parseWgslBindings(wgsl, shaderTypeToVisibility(ep.type)),
            };
        });
        return new ProgramVersion(new ProgramReflection(result.reflection), kernels);
    }
}

declare module "../API/Device.js" {
    interface Device {
        programManager: ProgramManager;
        /** Initializes the program system (slang-wasm + shader source registry). */
        setProgramManager(manager: ProgramManager): void;
    }
}

Device.prototype.setProgramManager = function (manager: ProgramManager): void {
    Object.defineProperty(this, "programManager", { value: manager, writable: false, configurable: true });
};
