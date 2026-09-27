/**
 * FalcorTest's Python-binding CPU tests (Tests/Utils/PropertiesTests.cpp, SettingsTests.cpp), run in Pyodide:
 * Properties from and to a Python dict, and the Settings binding driven by a script.
 */

import { getPyodide, initScripting, Properties, propertiesFromPython, propertiesToPython, Settings, settingsBinding } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../../harness/registry.js";

type Py = { runPython(code: string): unknown; globals: { set(name: string, value: unknown): void; delete(name: string): void } };

// pybind11's values for the numeric_limits in the native tests.
const kDict = `
import struct
_f32max = struct.unpack("f", struct.pack("f", 3.4028234663852886e38))[0]
pnested = {"str": "string"}
p = {"b": True, "u32": 2**32 - 1, "u64": 2**64 - 1, "i32": -2**31, "i64": -2**63, "f32": _f32max, "f64": 1.7976931348623157e308,
     "uint3": [1, 2, 3], "int3": [-1, 2, -3], "float3": [0.25, 0.5, 0.75], "str": "string", "nested": pnested}
`;

async function python(): Promise<Py> {
    await initScripting("/node_modules/pyodide");
    return getPyodide() as Py;
}

gpuTest("FalcorTest.PropertiesFromPython", async () => {
    const py = await python();
    py.runPython(kDict);
    const props = propertiesFromPython(py.runPython("p"));
    expectEq(props.get("b", false), true, "b");
    expectEq(props.get("u32", 0), 4294967295, "u32");
    expectEq(props.getOpt("u64"), 2n ** 64n - 1n, "u64");
    expectEq(props.get("i32", 0), -2147483648, "i32");
    expectEq(props.getOpt("i64"), -(2n ** 63n), "i64");
    expectEq(props.get("f32", 0), 3.4028234663852886e38, "f32");
    expectEq(props.get("f64", 0), Number.MAX_VALUE, "f64");
    expectEq(JSON.stringify(props.get("uint3", [])), "[1,2,3]", "uint3");
    expectEq(JSON.stringify(props.get("int3", [])), "[-1,2,-3]", "int3");
    expectEq(JSON.stringify(props.get("float3", [])), "[0.25,0.5,0.75]", "float3");
    expectEq(props.get("str", ""), "string", "str");
    expectEq(JSON.stringify(props.getProperties("nested").toJSON()), JSON.stringify(propertiesFromPython(py.runPython("pnested")).toJSON()), "nested");
    py.runPython("del p, pnested, _f32max");
});

gpuTest("FalcorTest.PropertiesToPython", async () => {
    const py = await python();
    py.runPython(kDict);
    const props = new Properties();
    props.set("b", true);
    props.set("u32", 4294967295);
    props.set("u64", 2n ** 64n - 1n);
    props.set("i32", -2147483648);
    props.set("i64", -(2n ** 63n));
    props.set("f32", Math.fround(3.4028234663852886e38));
    props.set("f64", Number.MAX_VALUE);
    props.set("uint3", [1, 2, 3]);
    props.set("int3", [-1, 2, -3]);
    props.set("float3", [0.25, 0.5, 0.75]);
    props.set("str", "string");
    props.set("nested", new Properties({ str: "string" }));
    py.globals.set("q", propertiesToPython(props));
    expectEq(py.runPython("q == p"), true, "props.toPython() == p");
    py.runPython("del p, q, pnested, _f32max");
});

gpuTest("FalcorTest.Settings_PythonBinding", async () => {
    const py = await python();
    const settings = new Settings();
    py.globals.set("harness", { getSettings: () => settingsBinding(settings) });
    py.runPython(`
harness.getSettings().addOptions({"test1":5})
harness.getSettings().addFilteredAttributes({"testName":6, "testName.filter":["shapeName1"]})
`);
    expectEq(settings.getOption("test1", 1), 5, "option");
    expectEq(settings.getAttribute("shapeName1", "testName", 2), 6, "filtered attribute");
    expectEq(settings.getAttribute("shapeName2", "testName", 2), 2, "unfiltered shape");
    py.runPython(`
harness.getSettings().clearOptions()
harness.getSettings().clearFilteredAttributes()
`);
    expectEq(settings.getOption("test1", 1), 1, "option cleared");
    expectEq(settings.getAttribute("shapeName1", "testName", 2), 2, "attribute cleared");
    expectEq(settings.getAttribute("shapeName2", "testName", 2), 2, "unfiltered shape");
    py.globals.delete("harness");
});
