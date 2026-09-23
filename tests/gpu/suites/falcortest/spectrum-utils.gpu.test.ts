/** Transplanted FalcorTest GPU test: Utils/Color/SpectrumUtilsTests (GPU fit vs native's 1 nm CIE table). */

import { Mt19937, SpectrumUtils } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect, uniformFloat } from "../../harness/expect.js";

gpuTest("FalcorTest.WavelengthToXYZ", async ({ device }) => {
    const fr = Math.fround;
    const u = uniformFloat(new Mt19937());
    const n = 20000;
    const wavelengths = Float32Array.from({ length: n }, (_, i) => fr(300 + fr(fr(fr(i + u()) / n) * 600)));
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Utils/Color/SpectrumUtilsTests.cs.slang", "testWavelengthToXYZ");
    ctx.allocateStructuredBuffer("result", n);
    ctx.allocateStructuredBuffer("wavelengths", n, wavelengths);
    ctx.vars()["CB"]["n"] = n;
    ctx.runProgram(n, 1, 1);
    const result = await ctx.readBuffer("result", Float32Array);
    const rs = ctx.getStructSize("result") / 4;
    const e = new Expect();
    const maxSq = [0, 0, 0];
    for (let i = 0; i < n; i++) {
        const ref = SpectrumUtils.wavelengthToXYZ_CIE1931(wavelengths[i]!);
        const res = [result[i * rs]!, result[i * rs + 1]!, result[i * rs + 2]!];
        e.check(res.every((v) => v >= 0), () => `i = ${i}: negative ${res}`);
        [ref.x, ref.y, ref.z].forEach((r, c) => (maxSq[c] = Math.max(maxSq[c]!, (r - res[c]!) ** 2)));
    }
    console.error(`# WavelengthToXYZ: max squared error ${maxSq.map((v) => v.toExponential(2)).join(", ")}`);
    e.check(maxSq[0]! <= 2.0e-4 && maxSq[1]! <= 6.6e-5 && maxSq[2]! <= 5.2e-4, () => `max squared error ${maxSq}`);
    e.done("WavelengthToXYZ");
});
