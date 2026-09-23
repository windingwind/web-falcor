/**
 * Transplanted FalcorTest GPU tests: Slang/InheritanceTests (interfaces with
 * default implementations from a base struct, dispatched on a runtime type),
 * including createDynamicObject over registered type conformances.
 */

import { Mt19937, type Device, type TypeConformance } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kNumTests = 16;
const fr = Math.fround;

function cpuResult(type: number, v: [number, number], d: [number, number, number]): [number, [number, number]] {
    const diff = (v[0] - v[1]) | 0;
    switch (type) {
        case 0:
            return [diff, [fr(d[0] - d[1]), -d[2]]];
        case 1:
            return [(diff + 1) | 0, [d[0], d[2]]];
        case 2:
            return [(diff + 2) | 0, [d[0], -d[2]]];
        default:
            return [(diff + 3) | 0, [fr(d[0] + d[1]), d[2]]];
    }
}

async function runInheritance(device: Device, entry: string, rng: Mt19937, conformances: TypeConformance[] = []): Promise<void> {
    // libstdc++ uniform_int_distribution<int>() over mt19937: reject draws >= 2^31.
    const ui = () => {
        let v: number;
        do v = rng.next();
        while (v >= 0x80000000);
        return v;
    };
    // uniform_real_distribution<double>(): generate_canonical<double, 53> (two draws).
    const uf = () => {
        const r = (rng.next() + rng.next() * 4294967296) / 18446744073709551616;
        return r >= 1 ? 1 - Number.EPSILON / 2 : r;
    };
    const types: number[] = [];
    const values: [number, number][] = [];
    const data: [number, number, number][] = [];
    for (let i = 0; i < kNumTests; i++) {
        types.push(i % 4);
        values.push([ui(), ui()]);
        data.push([fr(uf()), fr(uf()), fr(uf())]);
    }
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Slang/InheritanceTests.cs.slang", entry, { NUM_TESTS: kNumTests }, conformances);
    ctx.allocateStructuredBuffer("resultsInt", kNumTests);
    ctx.allocateStructuredBuffer("resultsFloat", kNumTests);
    const ds = ctx.getStructSize("data") / 4;
    const packed = new Float32Array(kNumTests * ds);
    data.forEach((d, i) => packed.set(d, i * ds));
    ctx.vars()["testType"] = device.createStructuredBuffer(4, kNumTests, undefined, Int32Array.from(types));
    ctx.vars()["testValue"] = device.createStructuredBuffer(8, kNumTests, undefined, Int32Array.from(values.flat()));
    ctx.vars()["data"] = device.createStructuredBuffer(ds * 4, kNumTests, undefined, packed);
    ctx.runProgram(kNumTests, 1, 1);
    const ri = await ctx.readBuffer("resultsInt", Int32Array);
    const rf = await ctx.readBuffer("resultsFloat", Float32Array);
    const e = new Expect();
    for (let i = 0; i < kNumTests; i++) {
        const [wi, wf] = cpuResult(types[i]!, values[i]!, data[i]!);
        e.check(ri[i] === wi, () => `i = ${i}: int ${ri[i]} != ${wi}`);
        e.check(rf[2 * i] === wf[0] && rf[2 * i + 1] === wf[1], () => `i = ${i}: float ${rf[2 * i]}, ${rf[2 * i + 1]} != ${wf}`);
    }
    e.done(entry);
}

// Native draws both tests from one file-scope mt19937, in test order.
const rng = new Mt19937();
gpuTest("FalcorTest.Inheritance_ManualCreate", async ({ device }) => runInheritance(device, "testInheritanceManual", rng));
gpuTest("FalcorTest.Inheritance_ConformanceCreate", async ({ device }) =>
    runInheritance(device, "testInheritanceConformance", rng, [
        { typeName: "TestV0SubNeg", interfaceName: "ITestInterface", id: 0 },
        { typeName: "TestV1DefDef", interfaceName: "ITestInterface", id: 1 },
        { typeName: "TestV2DefNeg", interfaceName: "ITestInterface", id: 2 },
        { typeName: "TestV3SumDef", interfaceName: "ITestInterface", id: 3 },
    ]),
);
