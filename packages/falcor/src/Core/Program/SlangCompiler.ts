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
    /** Index into the program's module list (default 0). */
    moduleIndex?: number;
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
    rowCount?: number;
    columnCount?: number;
    scalarType?: string;
    elementType?: SlangReflectionType;
    resultType?: SlangReflectionType;
    fields?: (SlangReflectionParameter & { binding?: { kind: string; offset?: number; size?: number } })[];
    access?: string;
    /** Present on parameterBlock types: element layout incl. total uniform size. */
    elementVarLayout?: {
        type?: SlangReflectionType;
        bindings?: { kind: string; index?: number; offset?: number; size?: number; count?: number }[];
    };
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
        symlink(target: string, linkPath: string): void;
    };
}

let slangInstance: SlangWasmApi | null = null;
let slangGlobalSession: { createSession(target: number): SlangSessionApi | null } | null = null;
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
    // One global session for the process: each global session loads the full
    // Slang core module and is never freed — creating one per define-set
    // exhausts the wasm heap after ~15 program variants.
    slangGlobalSession = slangInstance.createGlobalSession();
    if (!slangGlobalSession) throw new RuntimeError("Failed to create Slang global session");
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

    /**
     * Rewrites #include directives to absolute MEMFS paths. The wasm binding has
     * no -I search-path API; Falcor includes are shader-root-relative, with
     * occasional same-directory relative includes.
     */
    private rewriteIncludes(source: string, filePath: string): string {
        const dir = filePath.split("/").slice(0, -1).join("/");
        const known = new Set(this.registeredFiles);
        return source.replace(/^(\s*#\s*include\s+")([^"]+)(")/gm, (_m, pre: string, target: string, post: string) => {
            if (target.startsWith("/")) return `${pre}${target}${post}`;
            if (known.has(target)) return `${pre}/${target}${post}`;
            const relative = dir ? `${dir}/${target}` : target;
            if (known.has(relative)) return `${pre}/${relative}${post}`;
            return `${pre}${target}${post}`; // leave unresolved; slang reports the error
        });
    }

    /** Define-set key whose header the (global) MEMFS files currently carry. */
    private fsOwnerKey: string | null = null;

    /** Writes all registered shader files with this define header prepended.
     *  MEMFS is shared across sessions while modules load lazily per session,
     *  so the files must carry the ACTIVE session's defines whenever a compile
     *  may trigger implicit module loads. */
    private writeShaderFiles(defines: DefineList): void {
        const slang = slangInstance!;
        const header = defines.toHeader();
        const dirs = new Set<string>();
        for (const path of this.registeredFiles) {
            const source = this.resolveSource(path);
            if (source === undefined) continue;
            const dir = path.split("/").slice(0, -1).join("/");
            if (dir) {
                slang.FS.createPath("/", dir, true, true);
                dirs.add(dir);
            }
            const rewritten = this.rewriteIncludes(source, path);
            // Prepend program defines to modules only — .slangh files are textually
            // included into units that already carry the defines (avoid redefinition).
            const prefix = path.endsWith(".slangh") || path.endsWith(".h") ? "" : header;
            // #line keeps diagnostics pointing at the original source.
            slang.FS.writeFile(`/${path}`, `${prefix}#line 1 "${path}"\n${rewritten}`);
        }
        // Modules loaded implicitly from the FS are recorded without a leading
        // slash, so their imports resolve relative to their own directory.
        // Falcor imports are shader-root-relative: symlink each top-level root
        // into every directory so dir-relative resolution lands at the root.
        const roots = new Set(this.registeredFiles.map((f) => f.split("/")[0]!).filter((r) => !r.includes(".")));
        for (const dir of dirs) {
            for (const root of roots) {
                try {
                    slang.FS.symlink(`/${root}`, `/${dir}/${root}`);
                } catch {
                    // Path already exists (real directory or prior link) — fine.
                }
            }
        }
        this.fsOwnerKey = defines.key();
    }

    /** Sessions hold compiled-module caches that grow the wasm heap; past
     *  ~20 program variants slang-wasm aborts ("unreachable"). Keep an LRU of
     *  recent sessions and delete() evicted ones (embind teardown). */
    private static readonly kMaxSessions = 6;

    private getSession(defines: DefineList): SlangSessionApi {
        const key = defines.key();
        let session = this.sessions.get(key) ?? null;
        if (session) {
            // LRU touch.
            this.sessions.delete(key);
            this.sessions.set(key, session);
        } else {
            while (this.sessions.size >= SlangCompiler.kMaxSessions) {
                const [oldKey, old] = this.sessions.entries().next().value as [string, SlangSessionApi];
                this.sessions.delete(oldKey);
                (old as { delete?: () => void }).delete?.();
            }
            session = slangGlobalSession!.createSession(wgslTargetId) ?? null;
            if (!session) throw new RuntimeError("Failed to create Slang session");
            this.sessions.set(key, session);
        }
        // A session created for another define set may have stamped the shared
        // MEMFS with its header since this session last compiled; refresh so
        // lazy module loads during the upcoming compile see OUR defines.
        if (this.fsOwnerKey !== key) this.writeShaderFiles(defines);
        return session;
    }

    /**
     * Compiles a program (one or more source modules + entry points) to WGSL +
     * reflection. Mirrors ProgramManager::createProgramKernels' compile request;
     * multiple modules cover Falcor's multi-translation-unit programs (e.g.
     * FullScreenPass.vs.slang + user pixel shader).
     */
    compile(modulePaths: string | string[], entryPoints: EntryPointDesc[], defines = new DefineList()): CompileResult {
        const paths = typeof modulePaths === "string" ? [modulePaths] : modulePaths;
        const slang = slangInstance!;
        const session = this.getSession(defines);
        const header = defines.toHeader();

        const modules: (SlangModuleApi & SlangComponentApi)[] = paths.map((path) => {
            const source = this.resolveSource(path);
            if (source === undefined) throw new RuntimeError(`Shader source not found: ${path}`);
            const rewritten = this.rewriteIncludes(source, path);
            const moduleName = path.replace(/[/.]/g, "_");
            const module = session.loadModuleFromSource(`${header}#line 1 "${path}"\n${rewritten}`, moduleName, `/${path}`);
            if (!module) {
                const err = slang.getLastError();
                throw new RuntimeError(`Slang compilation failed for ${path}:\n${err.type}: ${err.message}`);
            }
            return module;
        });

        const eps: SlangComponentApi[] = entryPoints.map((ep) => {
            const moduleIndex = ep.moduleIndex ?? 0;
            const found = modules[moduleIndex]!.findAndCheckEntryPoint(ep.name, ep.type);
            if (!found) {
                const err = slang.getLastError();
                throw new RuntimeError(`Entry point '${ep.name}' not found in ${paths[moduleIndex]}:\n${err.type}: ${err.message}`);
            }
            return found;
        });

        const composite = session.createCompositeComponentType([...modules, ...eps]);
        const linked = composite.link();
        const entryPointCode = entryPoints.map((_ep, i) => linked.getEntryPointCode(i, 0));
        const layout = linked.getLayout(0);
        if (!layout) throw new RuntimeError("Slang reflection unavailable");
        return { entryPointCode, reflection: layout.toJsonObject() };
    }
}
