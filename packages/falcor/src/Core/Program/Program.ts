/**
 * Program & ProgramManager mirroring Falcor/Core/Program/Program.h and
 * ProgramManager.h. A Program is (source path + entry points + defines);
 * ProgramVersion is the compiled artifact for one define-set: WGSL + shader
 * modules + reflection + parsed bindings.
 */

import { Device } from "../API/Device.js";
import { DefineList } from "./DefineList.js";
import { SlangCompiler, ShaderType, type ShaderSourceResolver, type EntryPointDesc } from "./SlangCompiler.js";
import { kShaderOverrides } from "./ShaderOverrides.js";
import { ProgramReflection, parseWgslBindings, type WgslBinding } from "./ProgramReflection.js";
import { RuntimeError } from "../Error.js";

export interface ProgramDesc {
    /**
     * Shader module path(s) relative to the shader root, e.g.
     * "RenderPasses/ToneMapper/ToneMapper.cs.slang". Multiple modules mirror
     * Falcor's multi-translation-unit programs; entry points reference modules
     * via moduleIndex.
     */
    path: string | string[];
    entryPoints: EntryPointDesc[];
}

export interface EntryPointKernel {
    name: string;
    type: ShaderType;
    wgsl: string;
    module: GPUShaderModule;
    bindings: WgslBinding[];
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

    constructor(
        public readonly device: Device,
        public readonly desc: ProgramDesc,
        public readonly defines: DefineList,
    ) {}

    /** Mirrors Program::getActiveVersion: compile-on-miss per define-set. */
    getActiveVersion(): ProgramVersion {
        const manager = this.device.programManager;
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

    constructor(
        private readonly device: Device,
        resolveSource: ShaderSourceResolver,
        filePaths: string[],
    ) {
        // Substitute WGSL-incompatible upstream files with WebFalcor overrides (DESIGN.md §4.3).
        const resolveWithOverrides: ShaderSourceResolver = (path) => resolveSource(kShaderOverrides[path] ?? path);
        this.compiler = new SlangCompiler(resolveWithOverrides, filePaths);
    }

    createProgram(desc: ProgramDesc, defines = new DefineList()): Program {
        return new Program(this.device, desc, defines);
    }

    compileProgram(desc: ProgramDesc, defines: DefineList): ProgramVersion {
        const allDefines = this.globalDefines.clone().addAll(defines);
        const result = this.compiler.compile(desc.path, desc.entryPoints, allDefines);
        const kernels = desc.entryPoints.map((ep, i) => {
            const wgsl = result.entryPointCode[i]!;
            return {
                name: ep.name,
                type: ep.type,
                wgsl,
                module: this.device.gpuDevice.createShaderModule({ label: `${String(desc.path)}:${ep.name}`, code: wgsl }),
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
