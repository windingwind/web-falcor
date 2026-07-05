/**
 * Runtime Slang->WGSL compiler wrapping slang-wasm (the same compiler front-end
 * native Falcor embeds via slang-gfx). See DESIGN.md §4.1.
 *
 * Translation-unit semantics: Falcor injects program defines into every
 * translation unit of a compile request. The wasm binding's module API keeps
 * preprocessor state per module, so we reproduce Falcor's behavior by creating
 * one Slang session per define-set (== Falcor's ProgramVersion granularity) and
 * prepending the define header to every module source registered in that
 * session's filesystem.
 */

import { DefineList } from "./DefineList.js";
import { RuntimeError } from "../Error.js";

/** Slang stage ids (slang.h SlangStage). */
export enum ShaderType {
    Vertex = 1,
    Hull = 2,
    Domain = 3,
    Geometry = 4,
    Pixel = 5,
    Compute = 6,
}

export interface EntryPointDesc {
    name: string;
    type: ShaderType;
}

export interface CompileResult {
    /** WGSL source per entry point, in request order. */
    entryPointCode: string[];
    /** Slang reflection (ProgramLayout.toJsonObject()). */
    reflection: SlangReflectionJson;
}

/** Shape of the slang reflection JSON we rely on (subset). */
export interface SlangReflectionJson {
    parameters?: SlangReflectionParameter[];
    entryPoints?: {
        name: string;
        stage?: string;
        threadGroupSize?: number[];
        parameters?: SlangReflectionParameter[];
        bindings?: unknown[];
    }[];
}

export interface SlangReflectionParameter {
    name: string;
    binding?: { kind: string; index?: number; space?: number; offset?: number; size?: number; used?: number };
    type?: SlangReflectionType;
}

export interface SlangReflectionType {
    kind: string;
    name?: string;
    baseShape?: string;
    elementCount?: number;
    scalarType?: string;
    elementType?: SlangReflectionType;
    resultType?: SlangReflectionType;
    fields?: (SlangReflectionParameter & { binding?: { kind: string; offset?: number; size?: number } })[];
    access?: string;
    [key: string]: unknown;
}

/** Source-file registry: shader path (Falcor-style, e.g. "Utils/Math/MathHelpers.slang") -> source. */
export type ShaderSourceResolver = (path: string) => string | undefined;

interface SlangModuleApi {
    findAndCheckEntryPoint(name: string, stage: number): SlangComponentApi | null;
}
interface SlangComponentApi {
    link(): SlangComponentApi;
    getEntryPointCode(entryPointIndex: number, targetIndex: number): string;
    getLayout(targetIndex: number): { toJsonObject(): SlangReflectionJson } | null;
}
interface SlangSessionApi {
    loadModuleFromSource(source: string, name: string, path: string): (SlangModuleApi & SlangComponentApi) | null;
    createCompositeComponentType(components: unknown[]): SlangComponentApi;
}
interface SlangWasmApi {
    createGlobalSession(): { createSession(target: number): SlangSessionApi | null } | null;
    getCompileTargets(): { name: string; value: number }[];
    getLastError(): { type: string; message: string };
    FS: {
        createPath(parent: string, path: string, canRead: boolean, canWrite: boolean): void;
        writeFile(path: string, data: string): void;
        analyzePath(path: string): { exists: boolean };
    };
}

let slangInstance: SlangWasmApi | null = null;
let wgslTargetId = -1;

/** Loads the slang-wasm module (idempotent). `moduleUrl` points at slang-wasm.js. */
export async function initSlang(moduleUrl: string): Promise<void> {
    if (slangInstance) return;
    const { default: factory } = (await import(/* @vite-ignore */ moduleUrl)) as {
        default: () => Promise<SlangWasmApi>;
    };
    slangInstance = await factory();
    const wgsl = slangInstance.getCompileTargets().find((t) => t.name === "WGSL");
    if (!wgsl) throw new RuntimeError("slang-wasm build lacks the WGSL target");
    wgslTargetId = wgsl.value;
}

export function isSlangInitialized(): boolean {
    return slangInstance !== null;
}

export class SlangCompiler {
    /** Session cache keyed by define-set (mirrors ProgramVersion caching). */
    private sessions = new Map<string, SlangSessionApi>();
    private registeredFiles: string[] = [];

    constructor(private readonly resolveSource: ShaderSourceResolver, private readonly filePaths: string[]) {
        if (!slangInstance) throw new RuntimeError("Call initSlang() before constructing SlangCompiler");
        this.registeredFiles = filePaths;
    }

    private getSession(defines: DefineList): SlangSessionApi {
        const key = defines.key();
        let session = this.sessions.get(key) ?? null;
        if (!session) {
            const slang = slangInstance!;
            session = slang.createGlobalSession()?.createSession(wgslTargetId) ?? null;
            if (!session) throw new RuntimeError("Failed to create Slang session");
            const header = defines.toHeader();
            for (const path of this.registeredFiles) {
                const source = this.resolveSource(path);
                if (source === undefined) continue;
                const dir = path.split("/").slice(0, -1).join("/");
                if (dir) slang.FS.createPath("/", dir, true, true);
                // #line keeps diagnostics pointing at the original source.
                slang.FS.writeFile(`/${path}`, `${header}#line 1 "${path}"\n${source}`);
            }
            this.sessions.set(key, session);
        }
        return session;
    }

    /**
     * Compiles a program (single entry module importing others) to WGSL + reflection.
     * Mirrors ProgramManager::createProgramKernels' compile request.
     */
    compile(entrySourcePath: string, entryPoints: EntryPointDesc[], defines = new DefineList()): CompileResult {
        const slang = slangInstance!;
        const session = this.getSession(defines);
        const source = this.resolveSource(entrySourcePath);
        if (source === undefined) throw new RuntimeError(`Shader source not found: ${entrySourcePath}`);
        const moduleName = entrySourcePath.replace(/[/.]/g, "_");
        const header = defines.toHeader();
        const module = session.loadModuleFromSource(`${header}#line 1 "${entrySourcePath}"\n${source}`, moduleName, `/${entrySourcePath}`);
        if (!module) {
            const err = slang.getLastError();
            throw new RuntimeError(`Slang compilation failed for ${entrySourcePath}:\n${err.type}: ${err.message}`);
        }
        const eps: SlangComponentApi[] = [];
        for (const ep of entryPoints) {
            const found = module.findAndCheckEntryPoint(ep.name, ep.type);
            if (!found) {
                const err = slang.getLastError();
                throw new RuntimeError(`Entry point '${ep.name}' not found in ${entrySourcePath}:\n${err.type}: ${err.message}`);
            }
            eps.push(found);
        }
        const composite = session.createCompositeComponentType([module, ...eps]);
        const linked = composite.link();
        const entryPointCode = entryPoints.map((_ep, i) => linked.getEntryPointCode(i, 0));
        const layout = linked.getLayout(0);
        if (!layout) throw new RuntimeError("Slang reflection unavailable");
        return { entryPointCode, reflection: layout.toJsonObject() };
    }
}
