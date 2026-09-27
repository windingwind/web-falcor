/**
 * Converts Python scene/graph/Mogwai scripts to JS script modules (the converter is PyToJs.ts): the driver behind
 * `web-falcor py2js` and `npm run py2js`. Files keep their path relative to `root` under `out` (default: next to
 * the input): x.pyscene -> x.scene.js, x.py -> x.js; local imports resolve to the converted modules.
 */

import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

/** Python scripts under the inputs (.pyscene, .py; __pycache__ skipped). */
function collect(p) {
    const abs = resolve(p);
    if (statSync(abs).isFile()) return [abs];
    return readdirSync(abs, { withFileTypes: true }).flatMap((e) =>
        e.name === "__pycache__" ? [] : e.isDirectory() ? collect(join(abs, e.name)) : /\.(pyscene|py)$/.test(e.name) ? [join(abs, e.name)] : [],
    );
}

/** Python's parser from a Pyodide distribution directory (pyodide.mjs + its wasm and stdlib). */
export async function pythonParser(pyodideDir, kAstDumper) {
    const { loadPyodide } = await import(pathToFileURL(join(pyodideDir, "pyodide.mjs")).href);
    const pyodide = await loadPyodide({ indexURL: pyodideDir });
    pyodide.runPython(kAstDumper);
    const dump = pyodide.globals.get("_py2js_dump");
    return (src) => JSON.parse(dump(src));
}

/**
 * Converts `inputs` (files or directories). `core` is the converter module ({ convertPython, jsNameFor, kAstDumper }).
 * Returns the number of warnings (with `check`, syntax errors count too).
 */
export async function convertFiles({ inputs, out, root = ".", check = false, core, pyodideDir, log = console }) {
    const rootDir = resolve(root);
    const files = inputs.flatMap(collect);
    const parse = await pythonParser(pyodideDir, core.kAstDumper);
    const outPath = (file) => {
        const js = core.jsNameFor(file);
        return out ? join(resolve(out), relative(rootDir, js)) : js;
    };
    const imported = new Set();
    const resolveLocal = (fromFile, name, level, searchDirs) => {
        const dir = dirname(fromFile);
        const bases = level > 0 ? [resolve(dir, ...Array(level - 1).fill(".."))] : [dir, ...searchDirs.map((d) => resolve(dir, d))];
        const rel = name.split(".").join("/");
        for (const base of bases) {
            for (const cand of [join(base, `${rel}.py`), join(base, rel, "__init__.py")]) {
                if (!existsSync(cand)) continue;
                imported.add(cand);
                const r = relative(dirname(outPath(fromFile)), outPath(cand)).replaceAll("\\", "/");
                return r.startsWith(".") ? r : `./${r}`;
            }
        }
        return null;
    };
    // Two passes: the first finds which files are imported (they convert as module factories).
    for (const f of files) core.convertPython(readFileSync(f, "utf8"), parse, { kind: "script", fileName: f, resolveModule: (n, l, s) => resolveLocal(f, n, l, s) });
    const kindOf = (f) => (f.endsWith(".pyscene") ? "scene" : imported.has(f) ? "module" : "script");
    let warnings = 0;
    for (const f of files) {
        // __file__ names the original file as a server rooted at `root` serves it.
        const sourceUrl = `/${relative(rootDir, f).replaceAll("\\", "/")}`;
        const result = core.convertPython(readFileSync(f, "utf8"), parse, { kind: kindOf(f), fileName: relative(rootDir, f), sourceUrl, resolveModule: (n, l, s) => resolveLocal(f, n, l, s) });
        const target = outPath(f);
        mkdirSync(dirname(target), { recursive: true });
        writeFileSync(target, result.code);
        for (const w of result.warnings) log.warn(`warning: ${w}`);
        warnings += result.warnings.length;
        if (check) {
            try {
                execFileSync(process.execPath, ["--check", target], { stdio: "pipe" });
            } catch (e) {
                log.error(`syntax error in ${target}:\n${String(e.stderr ?? e)}`);
                warnings++;
            }
        }
    }
    log.log(`converted ${files.length} file(s)${out ? ` into ${out}` : ""}; ${warnings} warning(s)`);
    return warnings;
}

/** Parses `web-falcor py2js` / `npm run py2js` arguments: <file-or-dir>... [--out <dir>] [--root <dir>] [--check]. */
export function parseArgs(argv) {
    const args = [...argv];
    const take = (name) => {
        const i = args.indexOf(name);
        if (i < 0) return undefined;
        const v = args[i + 1];
        args.splice(i, 2);
        return v;
    };
    const out = take("--out");
    const root = take("--root");
    const check = args.includes("--check");
    return { out, root, check, inputs: args.filter((a) => a !== "--check") };
}
