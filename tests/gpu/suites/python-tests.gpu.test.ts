/**
 * Native python tests (Falcor/tests/python_tests, run natively by run_python_tests.py through
 * `python -m unittest`): the unmodified test modules run under the web Testbed bindings.
 */

import { initScripting, runTestbedScript } from "@web-falcor/falcor";
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
