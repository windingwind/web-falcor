/**
 * The Python-to-JS converter (PyToJs.ts) and the `py` runtime (PyRuntime.ts): Python semantics in converted
 * scripts, checked by converting snippets with Python's own parser (Pyodide in Node) and running the result.
 * The whole-corpus equivalence (scenes, graphs, image-test scripts) is the Py2Js GPU suite.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { convertPython, jsNameFor, kAstDumper, type ConvertOptions, type PythonParser } from "../src/Utils/Scripting/PyToJs.js";
import { createPyRuntime, pyNumberRepr } from "../src/Utils/Scripting/PyRuntime.js";

let parse: PythonParser;

beforeAll(async () => {
    const require = createRequire(import.meta.url);
    const dir = dirname(require.resolve("pyodide/package.json"));
    const { loadPyodide } = (await import(join(dir, "pyodide.mjs"))) as { loadPyodide(o: { indexURL: string }): Promise<{ runPython(s: string): unknown; globals: { get(k: string): (s: string) => string } }> };
    const pyodide = await loadPyodide({ indexURL: dir });
    pyodide.runPython(kAstDumper);
    const dump = pyodide.globals.get("_py2js_dump");
    parse = (src) => JSON.parse(dump(src)) as ReturnType<PythonParser>;
}, 60_000);

const convert = (src: string, opts: Partial<ConvertOptions> = {}) => convertPython(src, parse, { kind: "script", fileName: "t.py", ...opts });

/** Runs a converted script module with `ctx` (plus the runtime as `py`). */
async function run(code: string, ctx: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    // A file inside the project (vitest loads nothing outside it), not a data: URL: converted modules
    // resolve relative paths against import.meta.url.
    const dir = mkdtempSync(join(dirname(fileURLToPath(import.meta.url)), ".py2js-"));
    const file = join(dir, "t.js");
    writeFileSync(file, code);
    try {
        const mod = (await import(/* @vite-ignore */ pathToFileURL(file).href)) as { default: (c: Record<string, unknown>) => Promise<void> };
        const full = { py: createPyRuntime(), ...ctx };
        await mod.default(full);
        return full;
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

describe("py runtime", () => {
    const py = createPyRuntime();
    it("prints numbers as Python does", () => {
        expect(pyNumberRepr(1)).toBe("1");
        expect(pyNumberRepr(1, true)).toBe("1.0");
        expect(pyNumberRepr(0.25)).toBe("0.25");
        expect(pyNumberRepr(1e-5)).toBe("1e-05");
        expect(pyNumberRepr(1e16, true)).toBe("1e+16");
        expect(py.str(true)).toBe("True");
        expect(py.str(null)).toBe("None");
        expect(py.str([1, "a"])).toBe("[1, 'a']");
    });
    it("formats with %, format() and format specs", () => {
        expect(py.mod("%s-%d-%.2f-%5.1f|%-4s|", ["a", 3.7, 1.005, 2.25, "b"])).toBe("a-3-1.00-  2.2|b   |");
        expect(py.mod("%(x)s", { x: 1 })).toBe("1");
        expect(py.format("{}+{:03d}={x:.1f}", [1, 2], { x: 3 })).toBe("1+002=3.0");
        expect(py.fmt(1234567.891, -1, ",.2f")).toBe("1,234,567.89");
    });
    it("has Python arithmetic", () => {
        expect(py.mod(-7, 3)).toBe(2);
        expect(py.floordiv(-7, 2)).toBe(-4);
        expect(py.add([1], [2])).toEqual([1, 2]);
        expect(py.mul("ab", 2)).toBe("abab");
        const v = py.mul({ x: 1, y: 2, z: 3 }, 2) as Record<string, number>;
        expect([v.x, v.y, v.z]).toEqual([2, 4, 6]);
    });
    it("iterates, slices and indexes as Python", () => {
        expect(py.range(5, 0, -2)).toEqual([5, 3, 1]);
        expect(py.slice([0, 1, 2, 3, 4], 1, -1)).toEqual([1, 2, 3]);
        expect(py.slice("hello", null, null, -1)).toBe("olleh");
        expect(py.getitem([1, 2, 3], -1)).toBe(3);
        expect(py.getitem({ getPass: (n: string) => `pass:${n}` }, "PT")).toBe("pass:PT");
        expect(py.truth([])).toBe(false);
        expect(py.in("b", { a: 1, b: 2 })).toBe(true);
    });
    it("maps keyword arguments onto converted functions", () => {
        const f = py.def((a: number, b = 2, c = 3) => [a, b, c], ["a", "b", "c"]);
        expect(py.call(f, [1], { c: 9 })).toEqual([1, 2, 9]);
    });
});

describe("py2js converter", () => {
    it("names outputs like the originals", () => {
        expect(jsNameFor("dir/cornell_box.pyscene")).toBe("dir/cornell_box.scene.js");
        expect(jsNameFor("PathTracer.py")).toBe("PathTracer.js");
    });

    it("turns keyword arguments into options objects and positional arguments", () => {
        const { code, warnings } = convert("t = Transform(scaling=2, translation=float3(1, 0, 0))\nm = StandardMaterial(name='x')\nq = TriangleMesh.createSphere(radius=0.5)\n", { kind: "scene" });
        expect(warnings).toEqual([]);
        expect(code).toContain('Transform({ scaling: 2, translation: float3(1, 0, 0) })');
        expect(code).toContain('StandardMaterial("x")');
        expect(code).toContain("TriangleMesh.createSphere(0.5)");
    });

    it("runs loops, functions, comprehensions and formatting like Python", async () => {
        const src = [
            "out = []",
            "def scaled(v, k=2):",
            "    return v * k",
            "for i in range(3):",
            "    out.append(scaled(i, k=10))",
            "names = [f'n{i}' for i in range(2) if i != 5]",
            "for s in [0.5, 1.0]:",
            "    names.append('s.' + str(s))",
            "label = '%s=%.1f' % ('x', 2)",
            "pairs = {k: v for k, v in zip(['a', 'b'], [1, 2])}",
            "sink(out, names, label, pairs)",
        ].join("\n");
        const { code, warnings } = convert(src);
        expect(warnings).toEqual([]);
        let got: unknown[] = [];
        await run(code, { sink: (...a: unknown[]) => (got = a) });
        expect(got).toEqual([[0, 10, 20], ["n0", "n1", "s.0.5", "s.1.0"], "x=2.0", { a: 1, b: 2 }]);
    });

    it("keeps comments", () => {
        const { code } = convert("# leading comment\nx = 1  # trailing\n");
        expect(code).toContain("// leading comment");
        expect(code).toContain("// trailing");
    });

    it("runs exec(open(...)) scripts through the runtime and ends at exit()", async () => {
        const { code } = convert("exec(open('../scripts/PathTracer.py').read())\nsink('before')\nexit()\nsink('after')\n");
        expect(code).toContain('py.exec(ctx, new URL("../scripts/PathTracer.js", import.meta.url).href)');
        const calls: string[] = [];
        const py = Object.assign(createPyRuntime(), { exec: async () => void calls.push("exec") });
        await expect(run(code, { py, sink: (s: string) => calls.push(s) })).rejects.toMatchObject({ pyExit: true });
        expect(calls).toEqual(["exec", "before"]);
    });

    it("keeps a module's `m` out of reach, as Python modules don't see the script's globals", () => {
        const { code } = convert("g = RenderGraph('G')\ntry:\n    m.addGraph(g)\nexcept NameError:\n    pass\n", { kind: "module" });
        expect(code).not.toMatch(/const \{[^}]*\bm\b[^}]*\} = ctx/);
        expect(code).toContain('py.matches(__e1, "NameError")');
    });

    it("resolves local modules through the resolver", () => {
        const { code } = convert("from helpers import render_frames\nrender_frames(m, 'x', frames=[4])\n", { resolveModule: (name) => (name === "helpers" ? "../helpers.js" : null) });
        expect(code).toContain('import __py_helpers from "../helpers.js";');
        expect(code).toContain('py.call(render_frames, [m, "x"], { frames: [4] })');
    });
});
