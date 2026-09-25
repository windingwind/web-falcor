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
