/**
 * Converts Python scene scripts (.pyscene), render-graph and Mogwai scripts (.py) to the equivalent JS script
 * modules (JsScripting.ts). Python's own parser (in Pyodide) reads the source; this emits the module. Python
 * semantics JS lacks go through the `py` runtime (PyRuntime.ts), so a converted script behaves as the original.
 */

/** A Python AST node as serialised by kAstDumper. */
export interface PyNode {
    _type: string;
    lineno?: number;
    end_lineno?: number;
    [field: string]: unknown;
}

/** Source text -> { ast, comments }: Python parses and tokenizes (comments aren't in the AST). */
export type PythonParser = (source: string) => { ast: PyNode; comments: [number, string][] };

/** Python code serialising ast.parse(src) and the comment tokens to JSON; run it where Python is (Pyodide). */
export const kAstDumper = `
import ast as _ast, io as _io, json as _json, tokenize as _tok
def _py2js_node(n):
    if isinstance(n, _ast.AST):
        d = {'_type': type(n).__name__}
        for f, v in _ast.iter_fields(n):
            d[f] = _py2js_node(v)
        for a in ('lineno', 'end_lineno'):
            if hasattr(n, a): d[a] = getattr(n, a)
        return d
    if isinstance(n, list): return [_py2js_node(x) for x in n]
    if isinstance(n, bool) or n is None or isinstance(n, str): return n
    if isinstance(n, int): return n if abs(n) < 2**53 else float(n)
    if isinstance(n, float): return {'_type': '_float', 'value': repr(n)}
    if n is Ellipsis: return {'_type': '_ellipsis'}
    return repr(n)
def _py2js_dump(src):
    comments = []
    try:
        for t in _tok.generate_tokens(_io.StringIO(src).readline):
            if t.type == _tok.COMMENT: comments.append([t.start[0], t.string[1:].strip()])
    except Exception:
        pass
    return _json.dumps({'ast': _py2js_node(_ast.parse(src)), 'comments': comments})
`;

export type ConvertKind = "scene" | "script" | "module";

export interface ConvertOptions {
    /** "scene" (.pyscene), "script" (Mogwai/graph .py), or "module" (a helper a script imports). */
    kind: ConvertKind;
    /** The source file's name, for messages and the header. */
    fileName?: string;
    /**
     * A local module import (`from helpers import x`, `import graphs.X`) -> the JS module's path relative to the
     * converted file (e.g. "../helpers.js"), or null if it isn't local. `searchDirs` are sys.path.append arguments.
     */
    resolveModule?: (name: string, level: number, searchDirs: string[]) => string | null;
    /** Rewrites a script path in exec(open(p).read()) and m.script(p) (default: .py -> .js, .pyscene -> .scene.js). */
    rewriteScriptPath?: (path: string) => string;
    /** The Python file's own URL, which `__file__` names (paths built from it then find the same files). */
    sourceUrl?: string;
}

export interface ConvertResult {
    code: string;
    /** Constructs converted only approximately, or not at all (each names its line). */
    warnings: string[];
}

/** The JS file name for a Python script: x.pyscene -> x.scene.js, x.py -> x.js. */
export function jsNameFor(path: string): string {
    return path.replace(/\.pyscene$/i, ".scene.js").replace(/\.py$/i, ".js");
}

// Python keyword arguments: calls taking an options object, and positional signatures of the script API.
const kOptionsCalls = new Set(["Transform", "AABB", "SDFGrid.createSBS", "SDFGrid.createSVS", "SDFGrid.createSVO"]);
const kSignatures: Record<string, string[]> = {
    StandardMaterial: ["name", "model"],
    Material: ["name", "model"],
    ClothMaterial: ["name"],
    HairMaterial: ["name"],
    PBRTDiffuseMaterial: ["name"],
    PBRTConductorMaterial: ["name"],
    PBRTDiffuseTransmissionMaterial: ["name"],
    PBRTDielectricMaterial: ["name"],
    PBRTCoatedConductorMaterial: ["name"],
    PBRTCoatedDiffuseMaterial: ["name"],
    MERLMaterial: ["name", "path"],
    MERLMixMaterial: ["name", "paths"],
    RGLMaterial: ["name", "path"],
    Camera: ["name"],
    PointLight: ["name"],
    DirectionalLight: ["name"],
    DistantLight: ["name"],
    RectLight: ["name"],
    DiscLight: ["name"],
    SphereLight: ["name"],
    GridVolume: ["name"],
    Volume: ["name"],
    Animation: ["name", "nodeID", "duration"],
    EnvMap: ["path"],
    "EnvMap.createFromFile": ["path"],
    "TriangleMesh.createQuad": ["size"],
    "TriangleMesh.createCube": ["size"],
    "TriangleMesh.createSphere": ["radius", "segmentsU", "segmentsV"],
    "TriangleMesh.createDisk": ["radius", "segments"],
    "TriangleMesh.createFromFile": ["path", "smoothNormals", "flags"],
    "Grid.createSphere": ["radius", "voxelSize", "blendRange"],
    "Grid.createBox": ["width", "height", "depth", "voxelSize", "blendRange"],
    "Grid.createFromFile": ["path", "gridname"],
    "SDFGrid.createNDGrid": ["narrowBandThickness"],
    float2: ["x", "y"],
    float3: ["x", "y", "z"],
    float4: ["x", "y", "z", "w"],
    createPass: ["name", "dict"],
    RenderGraph: ["name"],
    addNode: ["name", "transform", "parent"],
    addMeshInstance: ["nodeID", "meshID"],
    addTriangleMesh: ["mesh", "material"],
    loadMaterialTexture: ["material", "slot", "path"],
    loadTexture: ["slot", "path", "useSrgb"],
    createAnimation: ["animatable", "name", "duration"],
    addCustomPrimitive: ["userID", "aabb"],
    addSDFGrid: ["grid", "material"],
    addSDFGridInstance: ["nodeID", "sdfGridID"],
    addPass: ["pass", "name"],
    addEdge: ["src", "dst"],
    markOutput: ["name", "mask"],
    updatePass: ["name", "dict"],
    loadScene: ["path", "buildFlags"],
    resizeFrameBuffer: ["width", "height"],
    resizeSwapChain: ["width", "height"],
    captureFrameTime: ["path"],
    add_search_path: ["path", "priority", "category"],
    resolve_path: ["path", "category"],
};

/** Python builtins -> the runtime (or plain JS). */
const kBuiltins: Record<string, string> = {
    len: "py.len", range: "py.range", str: "py.str", repr: "py.repr", int: "py.int", float: "py.float", bool: "py.bool",
    abs: "py.abs", min: "py.min", max: "py.max", round: "py.round", sum: "py.sum", list: "py.list", tuple: "py.tuple",
    dict: "py.dict", set: "py.set", enumerate: "py.enumerate", zip: "py.zip", sorted: "py.sorted", reversed: "py.reversed",
    any: "py.any", all: "py.all", map: "py.map", filter: "py.filter", isinstance: "py.isinstance", hasattr: "py.hasattr",
    getattr: "py.getattr", setattr: "py.setattr", print: "py.print", exit: "py.exit", quit: "py.exit", pow: "Math.pow",
};
/** Standard modules the runtime provides. */
const kStdModules: Record<string, string> = { math: "py.math", json: "py.json", os: "py.os", "os.path": "py.os.path", sys: "py.sys", glob: "py.glob" };
/** Names only a main script sees (Mogwai's globals, the scene's builder): a module's use raises NameError, as in Python. */
const kScriptOnlyNames = new Set(["m", "t", "fc", "tc", "sceneBuilder"]);

const kReserved = new Set(
    ("break case catch class const continue debugger default delete do else enum export extends false finally for function if implements import in " +
        "instanceof interface let new null package private protected public return static super switch this throw true try typeof var void while with " +
        "yield await arguments eval undefined NaN Infinity").split(" "),
);
const ident = (name: string) => (kReserved.has(name) ? `${name}_` : name);

const kOps: Record<string, string> = { Add: "add", Sub: "sub", Mult: "mul", Div: "div", FloorDiv: "floordiv", Mod: "mod" };
const kNativeOps: Record<string, string> = { BitOr: "|", BitAnd: "&", BitXor: "^", LShift: "<<", RShift: ">>", Pow: "**" };

/** Converts Python source to a JS script module. */
export function convertPython(source: string, parse: PythonParser, options: ConvertOptions): ConvertResult {
    const { ast, comments } = parse(source);
    return new Converter(options, comments).module(ast);
}

class Converter {
    private warnings: string[] = [];
    private imports: string[] = [];
    private importIds = new Map<string, string>();
    private moduleFactories: string[] = [];
    private searchDirs: string[] = [];
    private localDefs = new Map<string, string[]>();
    private comments: [number, string][];
    private tmp = 0;

    constructor(
        private readonly opts: ConvertOptions,
        comments: [number, string][],
    ) {
        this.comments = [...comments];
    }

    private warn(node: PyNode | undefined, msg: string): void {
        this.warnings.push(`${this.opts.fileName ?? "script"}:${node?.lineno ?? "?"}: ${msg}`);
    }

    module(ast: PyNode): ConvertResult {
        const body = ast.body as PyNode[];
        this.collectDefs(body);
        this.collectSearchDirs(body);
        const scope = new Scope(null, new Set());
        const defined = this.definedNames(body, scope.globals);
        const inner = this.block(body, 1, scope, defined);
        const free = [...this.freeNames(ast)]
            .filter((n) => !defined.has(n) && !this.allAssigned.has(n) && !(n in kBuiltins) && n !== "py" && n !== "__file__" && n !== "__name__" && n !== "exec" && n !== "open" && !(this.opts.kind === "module" && kScriptOnlyNames.has(n)))
            .sort();
        const destructure = free.length ? `    const { ${["py", ...free.map((n) => (kReserved.has(n) ? `${n}: ${ident(n)}` : n))].join(", ")} } = ctx;\n` : "    const { py } = ctx;\n";
        const lets = [...defined].filter((n) => !this.functionNames.has(n)).map(ident);
        const fileLine = this.usesFile ? `    const __file__ = ${this.opts.sourceUrl ? JSON.stringify(this.opts.sourceUrl) : "import.meta.url"};\n` : "";
        const letLine = `${fileLine}${lets.length ? `    let ${lets.join(", ")};\n` : ""}`;
        const header = `// Converted from ${this.opts.fileName ?? "a Python script"} by web-falcor's Python-to-JS converter.\n`;
        const imports = this.imports.length ? `${this.imports.join("\n")}\n\n` : "";
        // Star imports of local modules bind before the context's names are read.
        const stars = this.moduleFactories.join("");
        this.moduleFactories = [];
        const cwdLine = this.opts.kind === "script" && this.opts.sourceUrl ? `    py.cwd ??= ${JSON.stringify(this.opts.sourceUrl.slice(0, this.opts.sourceUrl.lastIndexOf("/")))};\n` : "";
        let code: string;
        if (this.opts.kind === "module") {
            const exported = [...defined].filter((n) => !n.startsWith("_")).map(ident);
            code =
                `${header}${imports}export default function (ctx) {\n    return ctx.py.module(import.meta.url, () => {\n` +
                indentBlock(`${stars.replaceAll("    Object", "    Object")}${destructure}${letLine}${inner}    return { ${exported.join(", ")} };\n`, 1) +
                `    });\n}\n`;
        } else {
            code = `${header}${imports}export default async function (ctx) {\n${stars}${destructure}${cwdLine}${letLine}${inner}}\n`;
        }
        const rest = this.comments.map(([, t]) => `// ${t}`).join("\n");
        return { code: rest ? `${code}${rest}\n` : code, warnings: this.warnings };
    }

    // ---- analysis ----

    private functionNames = new Set<string>();
    private usesFile = false;
    private importNames = new Set<string>();
    private allAssigned = new Set<string>();

    /** Every expression each name is bound to (for float inference); null marks a binding of unknown type. */
    private bindings = new Map<string, (PyNode | null)[]>();
    private floatNames: Set<string> | null = null;
    /** Types by program order: the last binding's floatness, for names rebound with different types. */
    private typeEnv = new Map<string, boolean>();

    private bind(name: string, value: PyNode | null): void {
        if (!this.bindings.has(name)) this.bindings.set(name, []);
        this.bindings.get(name)!.push(value);
    }

    private collectBindings(t: PyNode, value: PyNode | null): void {
        if (t._type === "Name") this.bind(t.id as string, value);
        else if (t._type === "Tuple" || t._type === "List") {
            const elts = t.elts as PyNode[];
            const vals = value && (value._type === "Tuple" || value._type === "List") && (value.elts as PyNode[]).length === elts.length ? (value.elts as PyNode[]) : null;
            elts.forEach((e, i) => this.collectBindings(e, vals ? vals[i]! : null));
        }
    }

    /** Whether `n` is statically a Python float (float literals, float arithmetic, names only ever bound to floats). */
    private isFloat(n: PyNode): boolean {
        if (!this.floatNames) {
            // Fixpoint: a name is a float if every binding is.
            this.floatNames = new Set();
            let changed = true;
            while (changed) {
                changed = false;
                for (const [name, vals] of this.bindings) {
                    if (this.floatNames.has(name) || !vals.length) continue;
                    if (vals.every((v) => v !== null && this.isFloatExpr(v))) {
                        this.floatNames.add(name);
                        changed = true;
                    }
                }
            }
        }
        return this.isFloatExpr(n);
    }

    private isFloatExpr(n: PyNode): boolean {
        switch (n._type) {
            case "Constant": return n.value !== null && typeof n.value === "object" && (n.value as PyNode)._type === "_float";
            case "_float": return true;
            case "Name": return this.typeEnv.get(n.id as string) ?? this.floatNames?.has(n.id as string) ?? false;
            case "UnaryOp": return this.isFloatExpr(n.operand as PyNode);
            case "BinOp": {
                const op = (n.op as PyNode)._type;
                if (op === "Div") return true;
                if (!["Add", "Sub", "Mult", "FloorDiv", "Mod", "Pow"].includes(op)) return false;
                return this.isFloatExpr(n.left as PyNode) || this.isFloatExpr(n.right as PyNode);
            }
            case "IfExp": return this.isFloatExpr(n.body as PyNode) && this.isFloatExpr(n.orelse as PyNode);
            case "Call": return dotted(n.func as PyNode) === "float";
            default: return false;
        }
    }

    private collectDefs(body: PyNode[]): void {
        for (const n of walk({ _type: "Module", body })) {
            if (n._type === "Assign") for (const t of n.targets as PyNode[]) this.collectBindings(t, n.value as PyNode);
            if (n._type === "AugAssign") this.collectBindings(n.target as PyNode, { _type: "BinOp", op: n.op, left: n.target, right: n.value });
            if (n._type === "AnnAssign") this.collectBindings(n.target as PyNode, (n.value as PyNode) ?? null);
            if (n._type === "For") {
                // A loop over a literal list binds its elements.
                const it = n.iter as PyNode;
                if (it._type === "List" || it._type === "Tuple") for (const e of it.elts as PyNode[]) this.collectBindings(n.target as PyNode, e);
                else this.collectBindings(n.target as PyNode, null);
            }
            if (n._type === "FunctionDef") for (const a of [...(((n.args as PyNode).posonlyargs as PyNode[]) ?? []), ...((n.args as PyNode).args as PyNode[])]) this.bind(a.arg as string, null);
            if (n._type === "comprehension") this.collectBindings(n.target as PyNode, null);
            if (n._type === "FunctionDef") {
                const a = n.args as PyNode;
                this.localDefs.set(n.name as string, [...((a.posonlyargs as PyNode[]) ?? []), ...(a.args as PyNode[])].map((x) => x.arg as string));
            }
            for (const t of assignTargets(n)) for (const name of targetNames(t)) this.allAssigned.add(name);
        }
    }

    private collectSearchDirs(body: PyNode[]): void {
        for (const n of walk({ _type: "Module", body })) {
            if (n._type !== "Call") continue;
            const f = dotted(n.func as PyNode);
            if (f === "sys.path.append" || f === "sys.path.insert") {
                const arg = (n.args as PyNode[])[f.endsWith("insert") ? 1 : 0];
                if (arg?._type === "Constant" && typeof arg.value === "string") this.searchDirs.push(arg.value);
            }
        }
    }

    /** Names a block binds at its own scope level (assignments, defs, imports, loop/with/except targets). */
    private definedNames(body: PyNode[], globals: Set<string>): Set<string> {
        const out = new Set<string>();
        const visit = (stmts: PyNode[]) => {
            for (const s of stmts) {
                switch (s._type) {
                    case "FunctionDef":
                        out.add(s.name as string);
                        this.functionNames.add(s.name as string);
                        continue; // its body is its own scope
                    case "ClassDef":
                        out.add(s.name as string);
                        continue;
                    case "Import":
                        for (const a of s.names as PyNode[]) {
                            const name = (a.asname as string) ?? (a.name as string).split(".")[0]!;
                            if (name === "falcor" && !a.asname) continue;
                            out.add(name);
                            this.importNames.add(name);
                        }
                        continue;
                    case "ImportFrom":
                        for (const a of s.names as PyNode[]) if (a.name !== "*" && !isFalcorModule(s.module as string)) {
                            const name = (a.asname as string) ?? (a.name as string);
                            out.add(name);
                            this.importNames.add(name);
                        }
                        continue;
                }
                for (const t of assignTargets(s)) for (const name of targetNames(t)) if (!globals.has(name)) out.add(name);
                for (const key of ["body", "orelse", "finalbody"]) if (Array.isArray(s[key])) visit(s[key] as PyNode[]);
                if (Array.isArray(s.handlers)) for (const h of s.handlers as PyNode[]) {
                    if (h.name) out.add(h.name as string);
                    visit(h.body as PyNode[]);
                }
            }
        };
        visit(body);
        return out;
    }

    /** Names read anywhere (Load context), i.e. candidates for the script context. */
    private freeNames(ast: PyNode): Set<string> {
        const out = new Set<string>();
        // Exception types in `except X:` are matched by name (py.matches), not read from the context.
        const exceptionNames = new Set<PyNode>();
        for (const n of walk(ast)) if (n._type === "ExceptHandler" && n.type) for (const t of walk(n.type as PyNode)) exceptionNames.add(t);
        // Comprehension and lambda variables are their own (block) scope.
        const bound = new Set<string>();
        for (const n of walk(ast)) {
            if (n._type === "comprehension") for (const t of targetNames(n.target as PyNode)) bound.add(t);
            if (n._type === "Lambda") for (const a of ((n.args as PyNode).args as PyNode[])) bound.add(a.arg as string);
        }
        for (const n of walk(ast)) if (n._type === "Name" && (n.ctx as PyNode)?._type === "Load" && !exceptionNames.has(n) && !bound.has(n.id as string)) out.add(n.id as string);
        return out;
    }

    // ---- statements ----

    private flushComments(beforeLine: number, depth: number): string {
        let out = "";
        while (this.comments.length && this.comments[0]![0] < beforeLine) out += `${pad(depth)}// ${this.comments.shift()![1]}\n`;
        return out;
    }

    private block(stmts: PyNode[], depth: number, scope: Scope, _defined?: Set<string>): string {
        let out = "";
        for (const s of stmts) {
            out += this.flushComments(s.lineno ?? Infinity, depth);
            out += this.stmt(s, depth, scope);
            // Trailing comments on the statement's own lines.
            while (this.comments.length && this.comments[0]![0] <= (s.end_lineno ?? s.lineno ?? 0) && s._type !== "FunctionDef" && s._type !== "For" && s._type !== "If" && s._type !== "While" && s._type !== "Try" && s._type !== "With")
                out = `${out.replace(/\n$/, "")} // ${this.comments.shift()![1]}\n`;
        }
        return out;
    }

    private stmt(s: PyNode, depth: number, scope: Scope): string {
        const p = pad(depth);
        switch (s._type) {
            case "Expr": {
                const v = s.value as PyNode;
                if (v._type === "Constant" && typeof v.value === "string") return (v.value as string).split("\n").map((l) => `${p}// ${l}`.trimEnd()).join("\n") + "\n";
                if (v._type === "Constant" && (v.value === null || typeof v.value === "number" || typeof v.value === "object")) return ""; // a bare `None` / `...` does nothing
                return `${p}${this.expr(v, scope)};\n`;
            }
            case "Assign": {
                const value = this.expr(s.value as PyNode, scope);
                const targets = s.targets as PyNode[];
                for (const t of targets) if (t._type === "Name") this.typeEnv.set(t.id as string, this.isFloat(s.value as PyNode));
                if (targets.length === 1) return `${p}${this.target(targets[0]!, scope)} = ${value};\n`;
                const t = `__t${++this.tmp}`;
                return `${p}const ${t} = ${value};\n` + targets.map((x) => `${p}${this.target(x, scope)} = ${t};\n`).join("");
            }
            case "AnnAssign":
                return s.value ? `${p}${this.target(s.target as PyNode, scope)} = ${this.expr(s.value as PyNode, scope)};\n` : "";
            case "AugAssign": {
                const target = s.target as PyNode;
                const op = (s.op as PyNode)._type;
                const t = this.target(target, scope);
                const cur = this.expr({ ...target, ctx: { _type: "Load" } }, scope);
                const rhs = this.expr(s.value as PyNode, scope);
                if (kOps[op]) return `${p}${t} = py.${kOps[op]}(${cur}, ${rhs});\n`;
                return `${p}${t} ${kNativeOps[op] ?? "+"}= ${rhs};\n`;
            }
            case "FunctionDef":
                return this.functionDef(s, depth, scope);
            case "Return":
                return `${p}return${s.value ? ` ${this.expr(s.value as PyNode, scope)}` : ""};\n`;
            case "If":
                return this.ifStmt(s, depth, scope, false);
            case "For":
                return this.forStmt(s, depth, scope);
            case "While": {
                if ((s.orelse as PyNode[]).length) this.warn(s, "while/else: the else branch is dropped");
                return `${p}while (${this.test(s.test as PyNode, scope)}) {\n${this.block(s.body as PyNode[], depth + 1, scope)}${p}}\n`;
            }
            case "Break":
                return `${p}break;\n`;
            case "Continue":
                return `${p}continue;\n`;
            case "Pass":
                return "";
            case "Global":
            case "Nonlocal":
                return "";
            case "Import":
                return this.importStmt(s, depth);
            case "ImportFrom":
                return this.importFrom(s, depth);
            case "Try":
                return this.tryStmt(s, depth, scope);
            case "Raise":
                return `${p}throw py.raise(${s.exc ? this.expr(s.exc as PyNode, scope) : "undefined"});\n`;
            case "With":
                return this.withStmt(s, depth, scope);
            case "Assert":
                return `${p}py.assert(${this.expr(s.test as PyNode, scope)}${s.msg ? `, ${this.expr(s.msg as PyNode, scope)}` : ""});\n`;
            case "Delete":
                return (s.targets as PyNode[]).map((t) => `${p}delete ${this.expr({ ...t, ctx: { _type: "Load" } }, scope)};\n`).join("");
            default:
                this.warn(s, `${s._type} is not converted`);
                return `${p}/* py2js: ${s._type} not converted */\n`;
        }
    }

    private functionDef(s: PyNode, depth: number, outer: Scope): string {
        const p = pad(depth);
        const a = s.args as PyNode;
        const positional = [...((a.posonlyargs as PyNode[]) ?? []), ...(a.args as PyNode[])];
        const defaults = a.defaults as PyNode[];
        const firstDefault = positional.length - defaults.length;
        const params = positional.map((x, i) => (i >= firstDefault ? `${ident(x.arg as string)} = ${this.expr(defaults[i - firstDefault]!, outer)}` : ident(x.arg as string)));
        if (a.vararg) params.push(`...${ident((a.vararg as PyNode).arg as string)}`);
        if (a.kwarg) this.warn(s, `**${(a.kwarg as PyNode).arg as string} is not converted`);
        if ((a.kwonlyargs as PyNode[]).length) this.warn(s, "keyword-only parameters are converted as positional ones");
        for (const k of a.kwonlyargs as PyNode[]) params.push(ident(k.arg as string));
        if ((s.decorator_list as PyNode[]).length) this.warn(s, "decorators are dropped");
        const globals = new Set<string>();
        for (const n of s.body as PyNode[]) if (n._type === "Global" || n._type === "Nonlocal") for (const g of n.names as string[]) globals.add(g);
        const scope = new Scope(outer, globals);
        const argNames = new Set([...positional, ...(a.kwonlyargs as PyNode[])].map((x) => x.arg as string));
        if (a.vararg) argNames.add((a.vararg as PyNode).arg as string);
        const locals = [...this.definedNames(s.body as PyNode[], globals)].filter((n) => !argNames.has(n) && !this.functionNamesIn(s.body as PyNode[]).has(n));
        const letLine = locals.length ? `${pad(depth + 1)}let ${locals.map(ident).join(", ")};\n` : "";
        const name = ident(s.name as string);
        const paramNames = [...positional, ...(a.kwonlyargs as PyNode[])].map((x) => JSON.stringify(x.arg));
        return `${p}function ${name}(${params.join(", ")}) {\n${letLine}${this.block(s.body as PyNode[], depth + 1, scope)}${p}}\n${p}py.def(${name}, [${paramNames.join(", ")}]);\n`;
    }

    private functionNamesIn(body: PyNode[]): Set<string> {
        return new Set(body.filter((s) => s._type === "FunctionDef").map((s) => s.name as string));
    }

    private ifStmt(s: PyNode, depth: number, scope: Scope, isElse: boolean): string {
        const p = pad(depth);
        let out = `${isElse ? "" : p}if (${this.test(s.test as PyNode, scope)}) {\n${this.block(s.body as PyNode[], depth + 1, scope)}${p}}`;
        const orelse = s.orelse as PyNode[];
        if (orelse.length === 1 && orelse[0]!._type === "If") out += ` else ${this.ifStmt(orelse[0]!, depth, scope, true).trimStart()}`;
        else if (orelse.length) out += ` else {\n${this.block(orelse, depth + 1, scope)}${p}}\n`;
        else out += "\n";
        return out;
    }

    private forStmt(s: PyNode, depth: number, scope: Scope): string {
        const p = pad(depth);
        if ((s.orelse as PyNode[]).length) this.warn(s, "for/else: the else branch is dropped");
        const target = s.target as PyNode;
        const iterNode = s.iter as PyNode;
        // The loop variable's type in the body: a literal list's element types, else unknown.
        this.setLoopTypes(target, iterNode);
        const body = this.block(s.body as PyNode[], depth + 1, scope);
        // for i in range(...) with a constant positive step: a counting loop (loop variables are function locals in Python).
        if (target._type === "Name" && iterNode._type === "Call" && dotted(iterNode.func as PyNode) === "range" && !(iterNode.keywords as PyNode[]).length) {
            const args = (iterNode.args as PyNode[]).map((x) => this.expr(x, scope));
            const step = (iterNode.args as PyNode[])[2];
            const stepValue = step === undefined ? 1 : constNumber(step);
            if (stepValue !== null && stepValue !== 0 && args.length <= 3) {
                const [start, stop] = args.length === 1 ? ["0", args[0]!] : [args[0]!, args[1]!];
                const v = ident(target.id as string);
                const cmp = stepValue > 0 ? "<" : ">";
                const stopVar = `__n${++this.tmp}`;
                return `${p}const ${stopVar} = ${stop};\n${p}for (${v} = ${start}; ${v} ${cmp} ${stopVar}; ${v} += ${stepValue}) {\n${body}${p}}\n`;
            }
        }
        const loopVar = `__v${++this.tmp}`;
        return `${p}for (const ${loopVar} of py.iter(${this.expr(iterNode, scope)})) {\n${pad(depth + 1)}${this.target(target, scope)} = ${loopVar};\n${body}${p}}\n`;
    }

    private setLoopTypes(target: PyNode, iterNode: PyNode): void {
        const elts = iterNode._type === "List" || iterNode._type === "Tuple" ? (iterNode.elts as PyNode[]) : null;
        const names = (t: PyNode, pick: (e: PyNode) => PyNode | null) => {
            if (t._type === "Name") this.typeEnv.set(t.id as string, !!elts?.length && elts.every((e) => { const v = pick(e); return v !== null && this.isFloat(v); }));
            else if (t._type === "Tuple" || t._type === "List")
                (t.elts as PyNode[]).forEach((sub, i) => names(sub, (e) => { const v = pick(e); return v && (v._type === "Tuple" || v._type === "List") ? ((v.elts as PyNode[])[i] ?? null) : null; }));
        };
        names(target, (e) => e);
    }

    private tryStmt(s: PyNode, depth: number, scope: Scope): string {
        const p = pad(depth);
        let out = `${p}try {\n${this.block(s.body as PyNode[], depth + 1, scope)}${p}}`;
        const handlers = s.handlers as PyNode[];
        if (handlers.length) {
            const e = `__e${++this.tmp}`;
            out += ` catch (${e}) {\n`;
            let chain = "";
            for (const h of handlers) {
                const types = h.type ? ((h.type as PyNode)._type === "Tuple" ? ((h.type as PyNode).elts as PyNode[]) : [h.type as PyNode]) : [];
                const cond = types.length ? types.map((t) => `py.matches(${e}, ${JSON.stringify(dotted(t))})`).join(" || ") : "true";
                const bind = h.name ? `${pad(depth + 2)}${ident(h.name as string)} = ${e};\n` : "";
                chain += `${chain ? " else " : pad(depth + 1)}if (${cond}) {\n${bind}${this.block(h.body as PyNode[], depth + 2, scope)}${pad(depth + 1)}}`;
            }
            out += `${chain} else {\n${pad(depth + 2)}throw ${e};\n${pad(depth + 1)}}\n${p}}`;
        }
        if ((s.orelse as PyNode[]).length) this.warn(s, "try/else: the else branch runs after the try block");
        if ((s.finalbody as PyNode[]).length) out += ` finally {\n${this.block(s.finalbody as PyNode[], depth + 1, scope)}${p}}`;
        out += "\n";
        if ((s.orelse as PyNode[]).length) out += this.block(s.orelse as PyNode[], depth, scope);
        return out;
    }

    private withStmt(s: PyNode, depth: number, scope: Scope): string {
        const p = pad(depth);
        const items = s.items as PyNode[];
        let body = this.block(s.body as PyNode[], depth + 1 + items.length - 1, scope);
        for (let i = items.length - 1; i >= 0; i--) {
            const item = items[i]!;
            const cm = item.context_expr as PyNode;
            const d = depth + i;
            const v = item.optional_vars ? this.target(item.optional_vars as PyNode, scope) : null;
            // m.profiler.event(name): the JS binding takes the body as a callback.
            if (cm._type === "Call" && (cm.func as PyNode)._type === "Attribute" && (cm.func as PyNode).attr === "event")
                body = `${pad(d)}${this.expr(cm, scope).replace(/\)$/, "")}, () => {\n${body}${pad(d)}});\n`;
            else body = `${pad(d)}py.with(${this.expr(cm, scope)}, (__w) => {\n${v ? `${pad(d + 1)}${v} = __w;\n` : ""}${body}${pad(d)}});\n`;
        }
        if (/\breturn\b|\bbreak;|\bcontinue;/.test(body)) this.warn(s, "return/break/continue inside `with` now leave only the with block");
        return body.startsWith(p) ? body : `${p}${body}`;
    }

    private importStmt(s: PyNode, depth: number): string {
        let out = "";
        for (const a of s.names as PyNode[]) {
            const full = a.name as string;
            const as = (a.asname as string | undefined) ?? full.split(".")[0]!;
            if (full === "falcor") {
                if (a.asname) out += `${pad(depth)}${ident(as)} = ctx;\n`;
                continue;
            }
            if (kStdModules[full]) {
                out += `${pad(depth)}${ident(as)} = ${kStdModules[full]};\n`;
                continue;
            }
            const local = this.localModule(full, 0, s);
            if (local) out += `${pad(depth)}${ident(as)} = ${local}(ctx);\n`;
            else {
                this.warn(s, `import ${full}: not a local module or a known standard module`);
                out += `${pad(depth)}${ident(as)} = undefined; // py2js: import ${full}\n`;
            }
        }
        return out;
    }

    private importFrom(s: PyNode, depth: number): string {
        const mod = (s.module as string | null) ?? "";
        const level = (s.level as number) ?? 0;
        const names = s.names as PyNode[];
        if (isFalcorModule(mod)) return ""; // the script context has falcor's names
        if (kStdModules[mod]) {
            return names.map((a) => `${pad(depth)}${ident((a.asname as string) ?? (a.name as string))} = ${kStdModules[mod]}.${a.name as string};\n`).join("");
        }
        const local = this.localModule(mod, level, s);
        if (!local) {
            this.warn(s, `from ${mod} import ...: not a local module`);
            return `${pad(depth)}/* py2js: from ${mod} import ${names.map((a) => a.name).join(", ")} */\n`;
        }
        if (names.some((a) => a.name === "*")) {
            // Star imports bind at the top (the names are only known at run time): the context gains the module's names.
            this.moduleFactories.push(`    Object.assign(ctx, ${local}(ctx));\n`);
            return "";
        }
        const t = `__m${++this.tmp}`;
        return `${pad(depth)}const ${t} = ${local}(ctx);\n` + names.map((a) => `${pad(depth)}${ident((a.asname as string) ?? (a.name as string))} = ${t}.${ident(a.name as string)};\n`).join("");
    }

    /** A local module's factory identifier (adding its static import), or null if it isn't local. */
    private localModule(name: string, level: number, node: PyNode): string | null {
        const path = this.opts.resolveModule?.(name, level, this.searchDirs) ?? null;
        if (!path) return null;
        let id = this.importIds.get(path);
        if (!id) {
            id = `__py_${name.replace(/\W/g, "_") || "module"}`;
            while ([...this.importIds.values()].includes(id)) id += "_";
            this.importIds.set(path, id);
            this.imports.push(`import ${id} from ${JSON.stringify(path)};`);
        }
        void node;
        return id;
    }

    // ---- expressions ----

    private test(n: PyNode, scope: Scope): string {
        return isBoolish(n) ? this.expr(n, scope) : `py.truth(${this.expr(n, scope)})`;
    }

    private target(n: PyNode, scope: Scope): string {
        switch (n._type) {
            case "Name":
                return ident(n.id as string);
            case "Tuple":
            case "List":
                return `[${(n.elts as PyNode[]).map((e) => (e._type === "Starred" ? `...${this.target(e.value as PyNode, scope)}` : this.target(e, scope))).join(", ")}]`;
            case "Attribute":
                return this.expr({ ...n, ctx: { _type: "Load" } }, scope);
            case "Subscript": {
                const sl = n.slice as PyNode;
                const obj = this.expr(n.value as PyNode, scope);
                if (sl._type === "Slice") {
                    this.warn(n, "slice assignment is not converted");
                    return "undefined";
                }
                return `${obj}[${this.expr(sl, scope)}]`;
            }
            default:
                this.warn(n, `assignment to ${n._type} is not converted`);
                return "undefined";
        }
    }

    expr(n: PyNode, scope: Scope): string {
        switch (n._type) {
            case "Constant":
                return constant(n.value);
            case "_float":
                return String(Number(n.value));
            case "Name": {
                const id = n.id as string;
                if (id === "__file__") {
                    this.usesFile = true;
                    return "__file__";
                }
                if (id === "__name__") return '"__main__"';
                if (kBuiltins[id] && !this.allAssigned.has(id)) return kBuiltins[id]!;
                return ident(id);
            }
            case "Attribute": {
                const attr = n.attr as string;
                if (attr === "__dict__") return this.expr(n.value as PyNode, scope);
                return `${this.expr(n.value as PyNode, scope)}.${attr}`;
            }
            case "Subscript": {
                const obj = this.expr(n.value as PyNode, scope);
                const sl = n.slice as PyNode;
                if (sl._type === "Slice") return `py.slice(${obj}, ${this.optExpr(sl.lower, scope)}, ${this.optExpr(sl.upper, scope)}, ${this.optExpr(sl.step, scope)})`;
                const c = constNumber(sl);
                if (c !== null && c < 0) return `${obj}.at(${c})`;
                if (c !== null) return `${obj}[${c}]`;
                // Loads go through getitem (a graph's pass by name, dicts, lists); stores stay brackets.
                if ((n.ctx as PyNode)?._type === "Load") return `py.getitem(${obj}, ${this.expr(sl, scope)})`;
                return `${obj}[${this.expr(sl, scope)}]`;
            }
            case "BinOp":
                return this.binOp(n, scope);
            case "UnaryOp": {
                const op = (n.op as PyNode)._type;
                const v = n.operand as PyNode;
                if (op === "Not") return `!${this.test(v, scope)}`;
                if (op === "USub") return constNumber(v) !== null ? `-${this.expr(v, scope)}` : `py.neg(${this.expr(v, scope)})`;
                if (op === "UAdd") return this.expr(v, scope);
                return `~${this.expr(v, scope)}`;
            }
            case "BoolOp":
                return `(${(n.values as PyNode[]).map((v) => this.expr(v, scope)).join((n.op as PyNode)._type === "And" ? " && " : " || ")})`;
            case "Compare":
                return this.compare(n, scope);
            case "IfExp":
                return `(${this.test(n.test as PyNode, scope)} ? ${this.expr(n.body as PyNode, scope)} : ${this.expr(n.orelse as PyNode, scope)})`;
            case "Call":
                return this.call(n, scope);
            case "List":
            case "Tuple":
                return `[${(n.elts as PyNode[]).map((e) => this.expr(e, scope)).join(", ")}]`;
            case "Set":
                return `py.set([${(n.elts as PyNode[]).map((e) => this.expr(e, scope)).join(", ")}])`;
            case "Dict": {
                const keys = n.keys as (PyNode | null)[];
                const values = n.values as PyNode[];
                const parts = keys.map((k, i) => {
                    const v = this.expr(values[i]!, scope);
                    if (k === null) return `...${v}`;
                    if (k._type === "Constant" && typeof k.value === "string" && /^[A-Za-z_$][\w$]*$/.test(k.value)) return `${k.value}: ${v}`;
                    if (k._type === "Constant" && (typeof k.value === "string" || typeof k.value === "number")) return `${JSON.stringify(k.value)}: ${v}`;
                    return `[${this.expr(k, scope)}]: ${v}`;
                });
                return parts.length ? `{ ${parts.join(", ")} }` : "{}";
            }
            case "ListComp":
            case "GeneratorExp":
            case "SetComp":
                return this.comprehension(n, scope, n._type === "SetComp" ? "set" : "list");
            case "DictComp":
                return this.comprehension(n, scope, "dict");
            case "JoinedStr":
                return this.fstring(n, scope);
            case "FormattedValue":
                if (!n.format_spec && (n.conversion as number) === -1 && this.isFloat(n.value as PyNode)) return `py.strFloat(${this.expr(n.value as PyNode, scope)})`;
                return `py.fmt(${this.expr(n.value as PyNode, scope)}, ${n.conversion as number}, ${n.format_spec ? this.specString(n.format_spec as PyNode, scope) : '""'})`;
            case "Lambda": {
                const a = n.args as PyNode;
                const params = (a.args as PyNode[]).map((x) => ident(x.arg as string));
                return `((${params.join(", ")}) => ${this.expr(n.body as PyNode, scope)})`;
            }
            case "Starred":
                return `...${this.expr(n.value as PyNode, scope)}`;
            case "NamedExpr":
                return `(${this.target(n.target as PyNode, scope)} = ${this.expr(n.value as PyNode, scope)})`;
            case "Await":
                return `(await ${this.expr(n.value as PyNode, scope)})`;
            case "_ellipsis":
                return "undefined";
            default:
                this.warn(n, `${n._type} is not converted`);
                return `undefined /* py2js: ${n._type} */`;
        }
    }

    private optExpr(n: unknown, scope: Scope): string {
        return n ? this.expr(n as PyNode, scope) : "null";
    }

    private binOp(n: PyNode, scope: Scope): string {
        const op = (n.op as PyNode)._type;
        const [l, r] = [n.left as PyNode, n.right as PyNode];
        const [a, b] = [this.expr(l, scope), this.expr(r, scope)];
        if (kNativeOps[op]) return `(${a} ${kNativeOps[op]} ${b})`;
        // Plain numbers stay infix; anything else may be a vector (or a string, for % and +).
        if (constNumber(l) !== null && constNumber(r) !== null && op !== "Mod" && op !== "FloorDiv") return `(${a} ${{ Add: "+", Sub: "-", Mult: "*", Div: "/" }[op]} ${b})`;
        if (op === "Mod" && l._type === "Constant" && typeof l.value === "string") return `py.mod(${a}, ${b})`;
        return `py.${kOps[op] ?? "add"}(${a}, ${b})`;
    }

    private compare(n: PyNode, scope: Scope): string {
        const ops = (n.ops as PyNode[]).map((o) => o._type);
        const operands = [n.left as PyNode, ...(n.comparators as PyNode[])];
        const parts: string[] = [];
        for (let i = 0; i < ops.length; i++) {
            const [l, r] = [operands[i]!, operands[i + 1]!];
            const [a, b] = [this.expr(l, scope), this.expr(r, scope)];
            const isNone = (x: PyNode) => x._type === "Constant" && x.value === null;
            const simple = (x: PyNode) => x._type === "Constant";
            switch (ops[i]) {
                case "Eq": parts.push(isNone(l) || isNone(r) ? `${a} == ${b}` : simple(l) || simple(r) ? `${a} === ${b}` : `py.eq(${a}, ${b})`); break;
                case "NotEq": parts.push(isNone(l) || isNone(r) ? `${a} != ${b}` : simple(l) || simple(r) ? `${a} !== ${b}` : `!py.eq(${a}, ${b})`); break;
                case "Is": parts.push(isNone(r) ? `${a} == null` : `${a} === ${b}`); break;
                case "IsNot": parts.push(isNone(r) ? `${a} != null` : `${a} !== ${b}`); break;
                case "In": parts.push(`py.in(${a}, ${b})`); break;
                case "NotIn": parts.push(`!py.in(${a}, ${b})`); break;
                default: parts.push(`${a} ${{ Lt: "<", LtE: "<=", Gt: ">", GtE: ">=" }[ops[i]!]} ${b}`);
            }
        }
        return parts.length === 1 ? parts[0]! : `(${parts.join(" && ")})`;
    }

    private comprehension(n: PyNode, scope: Scope, kind: "list" | "set" | "dict"): string {
        const gens = n.generators as PyNode[];
        const push = kind === "dict" ? `__r[${this.expr(n.key as PyNode, scope)}] = ${this.expr(n.value as PyNode, scope)};` : `__r.push(${this.expr(n.elt as PyNode, scope)});`;
        let inner = push;
        for (let i = gens.length - 1; i >= 0; i--) {
            const g = gens[i]!;
            const conds = (g.ifs as PyNode[]).map((c) => this.test(c, scope));
            const body = conds.length ? `if (${conds.join(" && ")}) { ${inner} }` : inner;
            inner = `for (const ${this.target(g.target as PyNode, scope)} of py.iter(${this.expr(g.iter as PyNode, scope)})) { ${body} }`;
        }
        const init = kind === "dict" ? "{}" : "[]";
        const ret = kind === "set" ? "py.set(__r)" : "__r";
        return `(() => { const __r = ${init}; ${inner} return ${ret}; })()`;
    }

    private fstring(n: PyNode, scope: Scope): string {
        let out = "`";
        for (const v of n.values as PyNode[]) {
            if (v._type === "Constant") out += String(v.value).replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
            else out += `\${${this.expr(v, scope)}}`;
        }
        return `${out}\``;
    }

    private specString(spec: PyNode, scope: Scope): string {
        const values = spec.values as PyNode[];
        if (values.every((v) => v._type === "Constant")) return JSON.stringify(values.map((v) => v.value).join(""));
        return this.fstring(spec, scope);
    }

    private call(n: PyNode, scope: Scope): string {
        const func = n.func as PyNode;
        const name = dotted(func);
        const args = n.args as PyNode[];
        const keywords = n.keywords as PyNode[];
        const last = func._type === "Attribute" ? (func.attr as string) : (name.split(".").pop() ?? "");
        const recv = func._type === "Attribute" ? this.expr(func.value as PyNode, scope) : "";

        // exec(open(p).read()): run the converted script in the same context.
        if (name === "exec" && args[0]?._type === "Call") {
            const inner = args[0];
            const open = inner.func as PyNode;
            if (open._type === "Attribute" && open.attr === "read" && (open.value as PyNode)._type === "Call" && dotted((open.value as PyNode).func as PyNode) === "open") {
                const pathNode = ((open.value as PyNode).args as PyNode[])[0]!;
                const path = pathNode._type === "Constant" ? JSON.stringify(this.rewrite(String(pathNode.value))) : this.expr(pathNode, scope);
                if (this.opts.kind === "module") this.warn(n, "exec() in a module: the script runs asynchronously");
                return `(await py.exec(ctx, new URL(${path}, import.meta.url).href))`;
            }
        }
        if (name === "open") this.warn(n, "open(): scripts read files with fetch() in JS");
        const argsJs = args.map((a) => this.expr(a, scope));
        if (func._type === "Attribute" && (func.attr === "script") && dotted(func.value as PyNode) === "m" && args[0]?._type === "Constant") {
            return `(await m.script(new URL(${JSON.stringify(this.rewrite(String(args[0].value)))}, import.meta.url).href))`;
        }
        if (name === "sys.path.append" || name === "sys.path.insert") return "undefined";

        // Python methods on lists, dicts and strings.
        if (func._type === "Attribute" && !keywords.length) {
            switch (last) {
                case "append": return `${recv}.push(${argsJs.join(", ")})`;
                case "extend": return `${recv}.push(...py.list(${argsJs[0]}))`;
                case "items": if (!args.length) return `py.items(${recv})`; break;
                case "keys": if (!args.length) return `py.keys(${recv})`; break;
                case "values": if (!args.length) return `py.values(${recv})`; break;
                case "get": if (args.length === 2) return `py.get(${recv}, ${argsJs.join(", ")})`; break;
                case "join": if (args.length === 1 && (func.value as PyNode)._type === "Constant") return `py.join(${recv}, ${argsJs[0]})`; break;
                case "format": if ((func.value as PyNode)._type === "Constant") return `py.format(${recv}, [${argsJs.join(", ")}])`; break;
                case "startswith": return `${recv}.startsWith(${argsJs.join(", ")})`;
                case "endswith": return `${recv}.endsWith(${argsJs.join(", ")})`;
                case "upper": if (!args.length) return `${recv}.toUpperCase()`; break;
                case "lower": if (!args.length) return `${recv}.toLowerCase()`; break;
                case "strip": if (!args.length) return `${recv}.trim()`; break;
                case "split": return `py.split(${recv}${argsJs.length ? `, ${argsJs.join(", ")}` : ""})`;
            }
        }
        if (func._type === "Attribute" && last === "format" && (func.value as PyNode)._type === "Constant")
            return `py.format(${recv}, [${argsJs.join(", ")}], { ${keywords.map((k) => `${k.arg as string}: ${this.expr(k.value as PyNode, scope)}`).join(", ")} })`;

        // str()/print() of a value known to be a float prints it as Python does ("1.0").
        if ((name === "str" || name === "print") && !this.allAssigned.has(name) && !keywords.length)
            return `${name === "str" ? "py.str" : "py.print"}(${args.map((a, i) => (this.isFloat(a) ? `py.strFloat(${argsJs[i]})` : argsJs[i])).join(", ")})`;
        const callee = this.expr(func, scope);
        if (!keywords.length) return `${callee}(${argsJs.join(", ")})`;
        const kw = keywords.filter((k) => k.arg !== null);
        const spreadKw = keywords.filter((k) => k.arg === null);
        const kwObj = `{ ${[...kw.map((k) => `${k.arg as string}: ${this.expr(k.value as PyNode, scope)}`), ...spreadKw.map((k) => `...${this.expr(k.value as PyNode, scope)}`)].join(", ")} }`;

        // Calls taking an options object: Transform(translation=...) -> Transform({ translation: ... }).
        const qualified = [name, name.split(".").slice(-2).join("."), last];
        if (qualified.some((q) => kOptionsCalls.has(q)) && !args.length) return `${callee}(${kwObj})`;
        // Known positional signatures (the script API, and this file's own functions).
        const sig = this.localDefs.get(name) ?? qualified.map((q) => kSignatures[q]).find(Boolean);
        if (sig && !spreadKw.length && !args.some((a) => a._type === "Starred")) {
            const all: string[] = [...argsJs];
            for (const k of kw) {
                const i = sig.indexOf(k.arg as string);
                if (i < 0) {
                    this.warn(n, `${name}(): unknown keyword argument '${k.arg as string}'`);
                    return `py.call(${callee}, [${argsJs.join(", ")}], ${kwObj}${recv ? `, ${recv}` : ""})`;
                }
                while (all.length < i) all.push("undefined");
                all[i] = this.expr(k.value as PyNode, scope);
            }
            return `${callee}(${all.join(", ")})`;
        }
        // Anything else: converted functions (py.def) map keywords at run time.
        return `py.call(${callee}, [${argsJs.join(", ")}], ${kwObj}${recv ? `, ${recv}` : ""})`;
    }

    private rewrite(path: string): string {
        return (this.opts.rewriteScriptPath ?? jsNameFor)(path);
    }
}

class Scope {
    constructor(
        readonly parent: Scope | null,
        readonly globals: Set<string>,
    ) {}
}

// ---- helpers ----

function pad(depth: number): string {
    return "    ".repeat(depth);
}
function indentBlock(s: string, depth: number): string {
    return s.split("\n").map((l) => (l ? pad(depth) + l : l)).join("\n");
}

function constant(v: unknown): string {
    if (v !== null && typeof v === "object" && (v as PyNode)._type === "_float") return String(Number((v as PyNode).value));
    if (v !== null && typeof v === "object" && (v as PyNode)._type === "_ellipsis") return "undefined";
    if (v === null) return "null";
    if (v === true) return "true";
    if (v === false) return "false";
    if (typeof v === "number") return String(v);
    if (typeof v === "string") return JSON.stringify(v);
    return "undefined";
}

function constNumber(n: PyNode): number | null {
    if (n._type === "Constant" && typeof n.value === "number") return n.value;
    if (n._type === "Constant" && n.value !== null && typeof n.value === "object" && (n.value as PyNode)._type === "_float") return Number((n.value as PyNode).value);
    if (n._type === "_float") return Number(n.value);
    if (n._type === "UnaryOp" && (n.op as PyNode)._type === "USub") {
        const v = constNumber(n.operand as PyNode);
        return v === null ? null : -v;
    }
    return null;
}

function isBoolish(n: PyNode): boolean {
    if (n._type === "Compare") return true;
    if (n._type === "Constant" && typeof n.value === "boolean") return true;
    if (n._type === "UnaryOp" && (n.op as PyNode)._type === "Not") return true;
    if (n._type === "BoolOp") return (n.values as PyNode[]).every(isBoolish);
    if (n._type === "Call" && ["isinstance", "hasattr", "any", "all", "bool"].includes(dotted(n.func as PyNode))) return true;
    return false;
}

function isFalcorModule(mod: string): boolean {
    return mod === "falcor" || mod.startsWith("falcor.");
}

/** a.b.c for Name/Attribute chains ("" otherwise). */
function dotted(n: PyNode): string {
    if (n._type === "Name") return n.id as string;
    if (n._type === "Attribute") {
        const base = dotted(n.value as PyNode);
        return base ? `${base}.${n.attr as string}` : "";
    }
    return "";
}

function assignTargets(n: PyNode): PyNode[] {
    switch (n._type) {
        case "Assign": return n.targets as PyNode[];
        case "AugAssign": case "AnnAssign": return [n.target as PyNode];
        case "For": return [n.target as PyNode];
        case "With": return (n.items as PyNode[]).map((i) => i.optional_vars as PyNode).filter(Boolean);
        case "NamedExpr": return [n.target as PyNode];
        default: return [];
    }
}

function targetNames(t: PyNode): string[] {
    if (t._type === "Name") return [t.id as string];
    if (t._type === "Tuple" || t._type === "List") return (t.elts as PyNode[]).flatMap(targetNames);
    if (t._type === "Starred") return targetNames(t.value as PyNode);
    return [];
}

function* walk(n: PyNode): Generator<PyNode> {
    yield n;
    for (const [k, v] of Object.entries(n)) {
        if (k === "_type") continue;
        if (Array.isArray(v)) for (const c of v) if (c && typeof c === "object" && "_type" in c) yield* walk(c as PyNode);
        if (v && typeof v === "object" && !Array.isArray(v) && "_type" in v) yield* walk(v as PyNode);
    }
}
