/**
 * Native python tests (Falcor/tests/python_tests, run natively by run_python_tests.py through
 * `python -m unittest`): the unmodified test modules run under the web Testbed bindings.
 */

import { initScripting, runTestbedScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

for (const module of ["test_dummy", "core/test_device"]) {
    gpuTest(`PythonTests.${module.replace("/", "_")}`, async ({ device }) => {
        await initScripting("/node_modules/pyodide");
        const { stderr } = await runTestbedScript(device, `/Falcor/tests/python_tests/${module}.py`, { extraFiles: module.includes("/") ? ["../helpers.py"] : ["helpers.py"], argv: ["-v"], cwd: "/Falcor/tests/python_tests" });
        console.error(`# ${module}: ${stderr.filter((l) => / \.\.\. |^Ran |^OK|^FAILED|Error/.test(l)).join(" | ")}`);
        expectEq(stderr.some((l) => /^Ran [1-9]/.test(l)), true, "unittest ran the tests");
        expectEq(stderr.some((l) => /^OK/.test(l)), true, "all tests pass");
    });
}

// Scene.get_material_params / set_material_params (SerializedMaterialParams), vs native python.
gpuTest("PythonTests.materialParamsMatchNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const { stdout } = await runTestbedScript(device, "/tests/oracle/material-params.py", { cwd: "/Falcor/media" });
    const line = stdout.find((l) => l.startsWith("MATERIALPARAMS "));
    expectEq(line !== undefined, true, "script printed its result");
    const web = JSON.parse(line!.slice("MATERIALPARAMS ".length)) as { names: string[]; before: number[][]; after: number[][]; count: number; layouts: unknown; standard: unknown };
    const native = (await (await fetch("/tests/oracle/out-native/material-params.json")).json()) as typeof web;
    expectEq(web.names.join(), native.names.join(), "materials");
    expectEq(web.count, native.count, "IMaterial.PARAM_COUNT");
    expectEq(JSON.stringify(web.layouts), JSON.stringify(native.layouts), "MATERIAL_PARAM_LAYOUTS");
    expectEq(JSON.stringify(web.standard), JSON.stringify(native.standard), "get_material_param_layout(Standard)");
    for (const k of ["before", "after"] as const) {
        const diff = web[k].flatMap((row, i) => row.map((v, j) => ({ i, j, v, n: native[k][i]![j]! }))).filter((d) => Math.fround(d.v) !== Math.fround(d.n));
        if (diff.length) console.error(`# ${k}: ${diff.slice(0, 8).map((d) => `[${d.i}][${d.j}] web ${d.v} native ${d.n}`).join("; ")}`);
        expectEq(diff.length, 0, `${k} params equal native`);
    }
});

gpuTest("PythonTests.meshVerticesMatchNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const { stdout, testbeds } = await runTestbedScript(device, "/tests/oracle/mesh-vertices.py", { cwd: "/Falcor/media", maxFrames: 1000 });
    const line = stdout.find((l) => l.startsWith("MESHVERTICES "));
    expectEq(line !== undefined, true, "script printed its result");
    type Mesh = [number[][], number[][], number[][]];
    const web = JSON.parse(line!.slice("MESHVERTICES ".length)) as { meshCount: number; box: number; before: Mesh[]; after: Mesh };
    const native = (await (await fetch("/tests/oracle/out-native/mesh-vertices.json")).json()) as typeof web;
    expectEq(web.meshCount, native.meshCount, "mesh count");
    expectEq(web.box, native.box, "small box mesh ID");
    const same = (a: Mesh, b: Mesh) => a.every((arr, k) => arr.length === b[k]!.length && arr.every((v, i) => v.every((x, j) => Math.fround(x) === Math.fround(b[k]![i]![j]!))));
    web.before.forEach((m, i) => expectEq(same(m, native.before[i]!), true, `mesh ${i} indices/positions/texcrds`));
    expectEq(same(web.after, native.after), true, "edited box read back");
    // Before the edit the path-traced image matches native's; after it the box has moved (native keeps its
    // static BLAS, so only the web render changes; 8x8-block means).
    const decode = (bytes: Uint8Array) => parseExr(bytes.slice().buffer, 1015) as { data: Float32Array; width: number; height: number };
    const before = decode(testbeds[0]!.captures.get("mesh_vertices.before.exr")!);
    const after = decode(testbeds[0]!.captures.get("mesh_vertices.after.exr")!);
    const nat = decode(new Uint8Array(await (await fetch("/tests/oracle/out-native/mesh-vertices.exr")).arrayBuffer()));
    const blockRelL1 = (a: Float32Array, b: Float32Array, w: number, h: number) => {
        let abs = 0, ref = 0;
        for (let by = 0; by < h / 8; by++)
            for (let bx = 0; bx < w / 8; bx++)
                for (let c = 0; c < 3; c++) {
                    let sa = 0, sb = 0;
                    for (let y = by * 8; y < by * 8 + 8; y++)
                        for (let x = bx * 8; x < bx * 8 + 8; x++) {
                            sa += a[(y * w + x) * 4 + c]!;
                            sb += b[(y * w + x) * 4 + c]!;
                        }
                    abs += Math.abs(sa - sb);
                    ref += Math.abs(sb);
                }
        return abs / ref;
    };
    const vsNative = blockRelL1(before.data, nat.data, nat.width, nat.height);
    const moved = blockRelL1(after.data, before.data, nat.width, nat.height);
    console.error(`# mesh-vertices: before vs native relL1 ${vsNative.toExponential(2)}, after vs before ${moved.toExponential(2)}`);
    expectEq(vsNative < 0.05, true, `before-edit block relative L1 ${vsNative}`);
    expectEq(moved > 0.05, true, `edit moved the box (${moved})`);
});

gpuTest("PythonTests.programApiMatchesNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const { stdout } = await runTestbedScript(device, "/tests/oracle/program-api.py", { cwd: "/Falcor/media" });
    const line = stdout.find((l) => l.startsWith("PROGRAMAPI "));
    expectEq(line !== undefined, true, "script printed its result");
    const web = JSON.parse(line!.slice("PROGRAMAPI ".length)) as Record<string, unknown>;
    const native = (await (await fetch("/tests/oracle/out-native/program-api.json")).json()) as Record<string, unknown>;
    for (const key of Object.keys(native)) expectEq(JSON.stringify(web[key]), JSON.stringify(native[key]), key);
});
