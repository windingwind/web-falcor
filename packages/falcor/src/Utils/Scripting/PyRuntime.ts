/**
 * Python semantics that JS lacks, for scripts converted from Python (PyToJs.ts): vector arithmetic, `%` formatting,
 * keyword arguments to converted functions, range/enumerate/zip, truthiness, and the os/math/json/sys helpers.
 * JS scripts receive it as `py` (hand-written ones can use it too).
 */

/** Thrown by exit(); the script runners treat it as a normal end (Python's SystemExit). */
export class PySystemExit extends Error {
    readonly pyExit = true;
    constructor(readonly code: unknown = 0) {
        super("SystemExit");
    }
}

type Vec = { x: number; y: number; z?: number; w?: number };
const isVec = (v: unknown): v is Vec => v !== null && typeof v === "object" && typeof (v as Vec).x === "number" && typeof (v as Vec).y === "number";
const kComps = ["x", "y", "z", "w"] as const;

/** Component-wise op for vectors (either side may be a scalar), plain op otherwise. */
function arith(op: (a: number, b: number) => number, plain: (a: never, b: never) => unknown) {
    return (a: unknown, b: unknown): unknown => {
        if (!isVec(a) && !isVec(b)) return plain(a as never, b as never);
        const proto = isVec(a) ? a : (b as Vec);
        const comps = kComps.filter((c) => typeof (proto as Record<string, unknown>)[c] === "number");
        const out = Object.create(Object.getPrototypeOf(proto)) as Record<string, number>;
        for (const c of comps) out[c] = op(isVec(a) ? (a as Record<string, number>)[c]! : (a as number), isVec(b) ? (b as Record<string, number>)[c]! : (b as number));
        return out;
    };
}

/**
 * Python's repr of a number: scientific below 1e-4 and from 1e16 ("1e-05"), else plain; `isFloat` prints
 * integral values as floats ("1.0"). JS numbers don't know they were floats: the converter says so where it can.
 */
export function pyNumberRepr(v: number, isFloat = false): string {
    if (!Number.isFinite(v)) return Number.isNaN(v) ? "nan" : v > 0 ? "inf" : "-inf";
    const a = Math.abs(v);
    if (Number.isInteger(v) && !isFloat) return String(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e16)) {
        const [mant, exp] = v.toExponential().split("e") as [string, string];
        const e = Number(exp);
        return `${mant}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;
    }
    return Number.isInteger(v) ? `${v}.0` : String(v);
}

/** Python's str() of a value (numbers not known to be floats print integral values as ints). */
function str(v: unknown): string {
    if (typeof v === "number") return pyNumberRepr(v);
    if (v === null || v === undefined) return "None";
    if (v === true) return "True";
    if (v === false) return "False";
    if (Array.isArray(v)) return `[${v.map(repr).join(", ")}]`;
    if (isVec(v)) return `[${kComps.filter((c) => typeof (v as Record<string, unknown>)[c] === "number").map((c) => (v as Record<string, number>)[c]).join(", ")}]`;
    return String(v);
}
function repr(v: unknown): string {
    return typeof v === "string" ? `'${v.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'` : str(v);
}

/** n.toFixed(digits), but exact binary ties round half to even, as Python does (2.25 -> "2.2"; JS gives "2.3"). */
function toFixedHalfEven(n: number, digits: number): string {
    const scaled = n * 10 ** digits;
    if (Number.isFinite(scaled) && Math.abs(scaled) < 2 ** 52 && !Number.isInteger(scaled) && Number.isInteger(scaled * 2) && scaled / 10 ** digits === n) {
        const lo = Math.floor(scaled);
        const even = lo % 2 === 0 ? lo : lo + 1;
        return (even / 10 ** digits).toFixed(digits);
    }
    return n.toFixed(digits);
}

/** Python format spec (subset): [fill][align][sign][0][width][,][.precision][type]. */
function formatSpec(v: unknown, spec: string): string {
    if (!spec) return str(v);
    const m = /^(?:(.)?([<>^=]))?([+\- ])?(0)?(\d+)?(,)?(?:\.(\d+))?([bcdeEfFgGnosxX%])?$/.exec(spec);
    if (!m) return str(v);
    const [, fillCh, align, sign, zero, widthS, comma, precS, type] = m;
    const prec = precS === undefined ? undefined : Number(precS);
    const n = Number(v);
    let s: string;
    switch (type) {
        case "f": case "F": s = toFixedHalfEven(n, prec ?? 6); break;
        case "e": case "E": s = n.toExponential(prec ?? 6).replace(/e([+-])(\d)$/, "e$10$2"); if (type === "E") s = s.toUpperCase(); break;
        case "g": case "G": s = prec === undefined ? String(n) : String(Number(n.toPrecision(prec || 1))); break;
        case "%": s = `${(n * 100).toFixed(prec ?? 6)}%`; break;
        case "d": case "n": s = String(Math.trunc(n)); break;
        case "x": s = Math.trunc(n).toString(16); break;
        case "X": s = Math.trunc(n).toString(16).toUpperCase(); break;
        case "o": s = Math.trunc(n).toString(8); break;
        case "b": s = Math.trunc(n).toString(2); break;
        case "c": s = String.fromCharCode(n); break;
        default: s = typeof v === "number" && prec !== undefined ? n.toFixed(prec) : str(v);
    }
    if (comma) s = s.replace(/^(-?\d+)/, (d) => d.replace(/\B(?=(\d{3})+(?!\d))/g, ","));
    if (sign === "+" && typeof v === "number" && n >= 0) s = `+${s}`;
    else if (sign === " " && typeof v === "number" && n >= 0) s = ` ${s}`;
    const width = widthS ? Number(widthS) : 0;
    if (s.length < width) {
        const fill = zero && !align ? "0" : (fillCh ?? " ");
        const pad = fill.repeat(width - s.length);
        const a = align ?? (zero ? "=" : typeof v === "number" ? ">" : "<");
        if (a === "<") s += pad;
        else if (a === "^") s = pad.slice(0, Math.floor(pad.length / 2)) + s + pad.slice(Math.floor(pad.length / 2));
        else if (a === "=" && /^[+\- ]/.test(s)) s = s[0] + pad + s.slice(1);
        else s = pad + s;
    }
    return s;
}

/** printf-style `fmt % args` (%s %d %i %f %e %g %x %r %%, with flags/width/precision). */
function percentFormat(fmt: string, args: unknown): string {
    const list = Array.isArray(args) ? args : [args];
    const named = !Array.isArray(args) && args !== null && typeof args === "object" && !isVec(args) ? (args as Record<string, unknown>) : null;
    let i = 0;
    return fmt.replace(/%(\(([^)]+)\))?([-+ 0#]*)(\d+)?(?:\.(\d+))?([sdifeEgGxXorc%])/g, (_m, _k, key: string | undefined, flags: string, width: string | undefined, prec: string | undefined, type: string) => {
        if (type === "%") return "%";
        const v = key !== undefined && named ? named[key] : list[i++];
        const align = flags.includes("-") ? "<" : "";
        const spec = `${align}${flags.includes("+") ? "+" : ""}${flags.includes("0") && !align ? "0" : ""}${width ?? ""}${prec !== undefined ? `.${prec}` : ""}${type === "s" || type === "r" ? "" : type === "i" ? "d" : type}`;
        return formatSpec(type === "r" ? repr(v) : type === "s" ? str(v) : v, spec);
    });
}

const kParams = Symbol("pyParams");

function iter(v: unknown): Iterable<unknown> {
    if (v === null || v === undefined) throw new TypeError(`'${str(v)}' object is not iterable`);
    if (typeof v === "string" || Array.isArray(v) || typeof (v as Iterable<unknown>)[Symbol.iterator] === "function") return v as Iterable<unknown>;
    if (typeof v === "object") return Object.keys(v as object); // a dict iterates its keys
    throw new TypeError(`'${typeof v}' object is not iterable`);
}
const list = (v: unknown): unknown[] => Array.from(iter(v));

function range(a: number, b?: number, step = 1): number[] {
    const [start, stop] = b === undefined ? [0, a] : [a, b];
    const out: number[] = [];
    if (step > 0) for (let i = start; i < stop; i += step) out.push(i);
    else for (let i = start; i > stop; i += step) out.push(i);
    return out;
}

function pyMod(a: unknown, b: unknown): unknown {
    if (typeof a === "string") return percentFormat(a, b);
    const [x, y] = [Number(a), Number(b)];
    return ((x % y) + y) % y; // Python's modulo takes the divisor's sign
}

/** Python truthiness: empty lists, dicts and strings are false. */
function truth(v: unknown): boolean {
    if (Array.isArray(v) || typeof v === "string") return v.length > 0;
    if (v instanceof Map || v instanceof Set) return v.size > 0;
    if (v !== null && typeof v === "object" && Object.getPrototypeOf(v) === Object.prototype) return Object.keys(v).length > 0;
    return !!v;
}

/** Exception classes for `except X:` (Python name -> matching JS errors). */
const kExceptions: Record<string, (e: unknown) => boolean> = {
    BaseException: () => true,
    Exception: (e) => !(e instanceof PySystemExit),
    SystemExit: (e) => e instanceof PySystemExit,
    NameError: (e) => e instanceof ReferenceError,
    AttributeError: (e) => e instanceof TypeError,
    TypeError: (e) => e instanceof TypeError,
    ValueError: (e) => e instanceof RangeError || e instanceof Error,
    KeyError: (e) => e instanceof Error,
    IndexError: (e) => e instanceof RangeError || e instanceof Error,
    RuntimeError: (e) => e instanceof Error && !(e instanceof PySystemExit),
    FileNotFoundError: (e) => e instanceof Error,
    ImportError: (e) => e instanceof Error,
    ModuleNotFoundError: (e) => e instanceof Error,
};

const kPathSep = "/";
/** Resolves `.` and `..` segments. */
function normalizePath(p: string): string {
    const out: string[] = [];
    for (const seg of p.split("/")) {
        if (seg === "..") out.length > 1 ? out.pop() : out.push(seg);
        else if (seg !== "." && (seg !== "" || out.length === 0)) out.push(seg);
    }
    return out.join("/") || "/";
}
const pathApi = {
    join: (...parts: string[]) => parts.reduce((acc, p) => (p.startsWith(kPathSep) || /^[a-z]+:\/\//i.test(p) ? p : acc ? `${acc.replace(/\/$/, "")}/${p}` : p), ""),
    dirname: (p: string) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : ""),
    basename: (p: string) => p.slice(p.lastIndexOf("/") + 1),
    splitext: (p: string) => {
        const base = p.slice(p.lastIndexOf("/") + 1);
        const dot = base.lastIndexOf(".");
        return dot > 0 ? [p.slice(0, p.length - base.length + dot), base.slice(dot)] : [p, ""];
    },
    exists: () => true,
    isabs: (p: string) => p.startsWith("/") || /^[a-z]+:\/\//i.test(p),
    abspath: (p: string) => p,
    normpath: (p: string) => p,
};

/**
 * A served directory's files (the dev server's /__webfalcor/ls listing), fetched synchronously: Python's
 * os.listdir/glob are synchronous. Null without a listing (static hosting), as the Python path.
 */
function listDirectorySync(dir: string): string[] | null {
    if (typeof XMLHttpRequest === "undefined") return null;
    const xhr = new XMLHttpRequest();
    xhr.open("GET", `/__webfalcor/ls?path=${encodeURIComponent(new URL(dir, globalThis.location?.href ?? "http://localhost/").pathname)}`, false);
    xhr.send();
    if (xhr.status !== 200 || !(xhr.getResponseHeader("content-type") ?? "").includes("json")) return null;
    const ls = JSON.parse(xhr.responseText) as { files: string[]; dirs: string[] };
    return [...ls.dirs, ...ls.files];
}

/** glob.glob over served files: wildcards (*, ?, [..]) in the last path component. */
function globSync(pattern: string): string[] {
    const slash = pattern.lastIndexOf("/");
    const [dir, name] = slash >= 0 ? [pattern.slice(0, slash) || "/", pattern.slice(slash + 1)] : [".", pattern];
    if (!/[*?[]/.test(name)) return (listDirectorySync(dir) ?? []).includes(name) ? [pattern] : [];
    const re = new RegExp(`^${name.replace(/[.+^${}()|\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".")}$`);
    return (listDirectorySync(dir) ?? []).filter((f) => re.test(f)).map((f) => (slash >= 0 ? `${pattern.slice(0, slash)}/${f}` : f));
}

/** The `py` helper object converted scripts use. */
export function createPyRuntime() {
    const modules = new Map<string, unknown>();
    const py = {
        add: arith((a, b) => a + b, (a: unknown, b: unknown) => (Array.isArray(a) && Array.isArray(b) ? [...a, ...b] : (a as number) + (b as number))),
        sub: arith((a, b) => a - b, (a: number, b: number) => a - b),
        mul: arith(
            (a, b) => a * b,
            (a: unknown, b: unknown) => (typeof a === "string" ? a.repeat(Number(b)) : Array.isArray(a) ? Array.from({ length: Number(b) }, () => a).flat() : (a as number) * (b as number)),
        ),
        div: arith((a, b) => a / b, (a: number, b: number) => a / b),
        floordiv: (a: unknown, b: unknown) => Math.floor(Number(a) / Number(b)),
        mod: pyMod,
        neg: (a: unknown) => (isVec(a) ? arith((x, y) => x * y, () => 0)(a, -1) : -(a as number)),
        eq: (a: unknown, b: unknown): boolean => {
            if (isVec(a) && isVec(b)) return kComps.every((c) => (a as Record<string, unknown>)[c] === (b as Record<string, unknown>)[c]);
            if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((v, i) => py.eq(v, b[i]));
            return a === b;
        },
        in: (item: unknown, container: unknown): boolean => {
            if (typeof container === "string") return container.includes(String(item));
            if (Array.isArray(container)) return container.some((v) => py.eq(v, item));
            if (container instanceof Map || container instanceof Set) return container.has(item);
            return container !== null && typeof container === "object" && String(item) in (container as object);
        },
        truth,
        str,
        /** str() of a value the converter knows is a float (1.0 prints as "1.0"). */
        strFloat: (v: unknown) => (typeof v === "number" ? pyNumberRepr(v, true) : str(v)),
        repr,
        fmt: (v: unknown, conversion = -1, spec = "") => formatSpec(conversion === 114 ? repr(v) : conversion === 115 ? str(v) : v, spec),
        format: (template: string, args: unknown[] = [], kw: Record<string, unknown> = {}) => {
            let i = 0;
            return template.replace(/\{\{|\}\}|\{([^{}:!]*)(?:!([rs]))?(?::([^{}]*))?\}/g, (m, key: string | undefined, conv: string | undefined, spec: string | undefined) => {
                if (m === "{{") return "{";
                if (m === "}}") return "}";
                const v = key === undefined || key === "" ? args[i++] : /^\d+$/.test(key) ? args[Number(key)] : kw[key];
                return formatSpec(conv === "r" ? repr(v) : conv === "s" ? str(v) : v, spec ?? "");
            });
        },
        iter,
        list,
        range,
        len: (v: unknown) => (typeof v === "string" || Array.isArray(v) ? v.length : v instanceof Map || v instanceof Set ? v.size : v !== null && typeof v === "object" && "length" in v ? Number((v as { length: number }).length) : Object.keys(v as object).length),
        enumerate: (v: unknown, start = 0) => list(v).map((x, i) => [i + start, x]),
        zip: (...lists: unknown[]) => {
            const ls = lists.map(list);
            return Array.from({ length: Math.min(...ls.map((l) => l.length)) }, (_v, i) => ls.map((l) => l[i]));
        },
        items: (d: object) => (d instanceof Map ? [...d.entries()] : Object.entries(d)),
        keys: (d: object) => (d instanceof Map ? [...d.keys()] : Object.keys(d)),
        values: (d: object) => (d instanceof Map ? [...d.values()] : Object.values(d)),
        get: (d: unknown, key: unknown, fallback: unknown = null) => {
            const g = (d as { get?: unknown } | null)?.get;
            if (typeof g === "function" && !(d !== null && typeof d === "object" && Object.getPrototypeOf(d) === Object.prototype)) return (g as (k: unknown, f: unknown) => unknown).call(d, key, fallback);
            return d !== null && typeof d === "object" && String(key) in (d as object) ? (d as Record<string, unknown>)[String(key)] : fallback;
        },
        /** obj[key]: a graph's pass by name (RenderGraph.__getitem__), a dict entry, a list/string element. */
        getitem: (obj: unknown, key: unknown): unknown => {
            if (obj === null || obj === undefined) throw new TypeError(`'${str(obj)}' object is not subscriptable`);
            if ((typeof obj === "string" || Array.isArray(obj)) && typeof key === "number") return key < 0 ? (obj as unknown[]).at(key) : (obj as unknown[])[key];
            if (obj instanceof Map) return obj.get(key);
            const o = obj as Record<string, unknown> & { getPass?: (n: string) => unknown };
            if (typeof key === "string" && !(key in o) && typeof o.getPass === "function") return o.getPass(key);
            return o[String(key)];
        },
        slice: (v: unknown, lo?: number | null, hi?: number | null, step?: number | null) => {
            const s = step ?? 1;
            const arr = typeof v === "string" ? [...v] : list(v);
            const n = arr.length;
            const norm = (i: number | null | undefined, d: number) => (i === null || i === undefined ? d : i < 0 ? Math.max(n + i, s < 0 ? -1 : 0) : Math.min(i, s < 0 ? n - 1 : n));
            const out: unknown[] = [];
            if (s > 0) for (let i = norm(lo, 0); i < norm(hi, n); i += s) out.push(arr[i]);
            else for (let i = norm(lo, n - 1); i > norm(hi, -1); i += s) out.push(arr[i]);
            return typeof v === "string" ? out.join("") : out;
        },
        join: (sep: string, items: unknown) => list(items).map(String).join(sep),
        split: (s: string, sep?: string | null, maxsplit = -1) => {
            const parts = sep === null || sep === undefined ? s.trim().split(/\s+/).filter(Boolean) : s.split(sep);
            return maxsplit < 0 || parts.length <= maxsplit + 1 ? parts : [...parts.slice(0, maxsplit), parts.slice(maxsplit).join(sep ?? " ")];
        },
        int: (v: unknown, base?: number) => (typeof v === "string" ? parseInt(v, base ?? 10) : Math.trunc(Number(v))),
        float: (v: unknown) => (v === "inf" ? Infinity : v === "-inf" ? -Infinity : Number(v)),
        bool: truth,
        abs: (v: unknown) => (isVec(v) ? arith((a) => Math.abs(a), () => 0)(v, 0) : Math.abs(Number(v))),
        min: (...a: unknown[]) => (a.length === 1 ? Math.min(...(list(a[0]) as number[])) : Math.min(...(a as number[]))),
        max: (...a: unknown[]) => (a.length === 1 ? Math.max(...(list(a[0]) as number[])) : Math.max(...(a as number[]))),
        sum: (v: unknown, start = 0) => list(v).reduce((acc: unknown, x) => py.add(acc, x), start),
        round: (v: number, digits?: number) => (digits === undefined ? Math.round(v) : Number(v.toFixed(digits))),
        sorted: (v: unknown, key?: (x: unknown) => unknown, reverse = false) => {
            const out = list(v).slice().sort((a, b) => {
                const [ka, kb] = key ? [key(a), key(b)] : [a, b];
                return (ka as number) < (kb as number) ? -1 : (ka as number) > (kb as number) ? 1 : 0;
            });
            return reverse ? out.reverse() : out;
        },
        reversed: (v: unknown) => list(v).slice().reverse(),
        any: (v: unknown) => list(v).some(truth),
        all: (v: unknown) => list(v).every(truth),
        map: (f: (...a: unknown[]) => unknown, ...its: unknown[]) => py.zip(...its).map((args) => f(...(args as unknown[]))),
        filter: (f: ((x: unknown) => unknown) | null, v: unknown) => list(v).filter((x) => truth(f ? f(x) : x)),
        dict: (v?: unknown) => (v === undefined ? {} : Array.isArray(v) ? Object.fromEntries(v as [string, unknown][]) : { ...(v as object) }),
        tuple: list,
        set: (v?: unknown) => new Set(v === undefined ? [] : list(v)),
        isinstance: (v: unknown, t: unknown): boolean => (Array.isArray(t) ? t.some((x) => py.isinstance(v, x)) : typeof t === "function" && v instanceof (t as new () => unknown)),
        hasattr: (o: unknown, k: string) => o !== null && o !== undefined && k in Object(o),
        getattr: (o: unknown, k: string, d?: unknown) => (o !== null && o !== undefined && k in Object(o) ? (o as Record<string, unknown>)[k] : d),
        setattr: (o: Record<string, unknown>, k: string, v: unknown) => void (o[k] = v),
        print: (...a: unknown[]) => console.log(a.map(str).join(" ")),
        exit: (code: unknown = 0): never => {
            throw new PySystemExit(code);
        },
        assert: (cond: unknown, msg?: unknown) => {
            if (!truth(cond)) throw new Error(`AssertionError${msg === undefined ? "" : `: ${str(msg)}`}`);
        },
        raise: (e: unknown) => (e instanceof Error ? e : new Error(str(e))),
        /** `except Name:` matcher. */
        matches: (e: unknown, name: string) => (kExceptions[name] ?? (() => e instanceof Error))(e),
        /** Records a converted function's parameter names, so calls with keyword arguments can map them. */
        def<F extends (...a: never[]) => unknown>(f: F, params: string[]): F {
            (f as unknown as { [kParams]: string[] })[kParams] = params;
            return f;
        },
        /** f(*args, **kwargs) for a callee the converter couldn't resolve: converted functions map names to positions. */
        call(f: unknown, args: unknown[], kwargs: Record<string, unknown>, thisArg?: unknown): unknown {
            if (typeof f !== "function") throw new TypeError(`'${str(f)}' is not callable`);
            const params = (f as { [kParams]?: string[] })[kParams];
            if (!params) return (f as (...a: unknown[]) => unknown).apply(thisArg, Object.keys(kwargs).length ? [...args, kwargs] : args);
            const all = [...args];
            for (const [k, v] of Object.entries(kwargs)) {
                const i = params.indexOf(k);
                if (i < 0) throw new TypeError(`${f.name}() got an unexpected keyword argument '${k}'`);
                while (all.length < i) all.push(undefined);
                all[i] = v;
            }
            return (f as (...a: unknown[]) => unknown).apply(thisArg, all);
        },
        /** `with x as y:` over a context manager (__enter__/__exit__), or a plain value. */
        with<T>(mgr: unknown, body: (v: unknown) => T): T {
            const cm = mgr as { __enter__?: () => unknown; __exit__?: (...a: unknown[]) => unknown } | null;
            const v = cm?.__enter__ ? cm.__enter__() : mgr;
            try {
                return body(v);
            } finally {
                cm?.__exit__?.(null, null, null);
            }
        },
        /** A converted local module's body, run once per script context (Python caches imported modules). */
        module<T>(key: string, body: () => T): T {
            if (!modules.has(key)) modules.set(key, body());
            return modules.get(key) as T;
        },
        math: {
            pi: Math.PI, e: Math.E, tau: 2 * Math.PI, inf: Infinity, nan: NaN,
            sqrt: Math.sqrt, sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
            exp: Math.exp, log: (x: number, b?: number) => (b === undefined ? Math.log(x) : Math.log(x) / Math.log(b)), log2: Math.log2, log10: Math.log10, pow: Math.pow,
            floor: Math.floor, ceil: Math.ceil, fabs: Math.abs, trunc: Math.trunc, hypot: Math.hypot,
            radians: (d: number) => (d * Math.PI) / 180, degrees: (r: number) => (r * 180) / Math.PI,
            isclose: (a: number, b: number, rel = 1e-9, abs = 0) => Math.abs(a - b) <= Math.max(rel * Math.max(Math.abs(a), Math.abs(b)), abs),
        },
        json: { dumps: (v: unknown) => JSON.stringify(v), loads: (s: string) => JSON.parse(s) as unknown },
        /** The working directory (the Mogwai script's own, as the Python runner chdirs there); relative paths resolve against it. */
        cwd: undefined as string | undefined,
        os: {
            path: {
                ...pathApi,
                abspath: (p: string) => (pathApi.isabs(p) || !py.cwd ? p : normalizePath(`${py.cwd}/${p}`)),
                exists: (p: string) => {
                    const abs = pathApi.isabs(p) || !py.cwd ? p : `${py.cwd}/${p}`;
                    return (listDirectorySync(pathApi.dirname(abs)) ?? [pathApi.basename(abs)]).includes(pathApi.basename(abs));
                },
            },
            sep: kPathSep,
            getcwd: () => py.cwd ?? "",
            listdir: (dir: string) => listDirectorySync(dir) ?? [],
        },
        glob: { glob: globSync },
        sys: { path: [] as string[], argv: [] as string[], exit: (code?: unknown) => py.exit(code) },
    };
    return py;
}

export type PyRuntime = ReturnType<typeof createPyRuntime>;
