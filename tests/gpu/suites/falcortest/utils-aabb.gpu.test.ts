/**
 * Transplanted FalcorTest GPU test Utils/AABBTests: the native kernel over Utils.Math.AABB and
 * its assertions in order. float3 buffers have WGSL's 16-byte stride.
 */

import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

type V3 = [number, number, number];
const kTestData: V3[] = [
    [1.0, 2.5, -0.5],
    [-3.5, -0.0, -1.25],
    [4.0, 2.75, -2.5],
    [0.5, 1.25, 4.5],
];
const FLT_MAX = 3.4028234663852886e38;
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const s = (v: number): V3 => [v, v, v];

// Native expectations: a full float3, only .x, or .x and .y.
type Check = { v: V3 } | { x: number } | { xy: [number, number] };
const v = (x: V3): Check => ({ v: x });
const x = (a: number): Check => ({ x: a });
const xy = (a: number, b: number): Check => ({ xy: [a, b] });
const d = kTestData;
const kExpected: Check[] = [
    // Test 0: set()
    v(d[0]!), v(d[0]!), v(d[1]!), v(d[2]!),
    // Test 1: valid()
    v(s(1)), v(s(1)), v(s(0)), v(d[0]!), v(add(d[0]!, [1, 1, -0.5])),
    // Test 2: invalidate()
    v(s(0)), v(s(FLT_MAX)), v(s(-FLT_MAX)),
    // Test 3: include()
    v([-3.5, 0, -1.25]), v([1, 2.5, -0.5]), v([0.5, 1.25, -2.5]), v([4, 2.75, 4.5]), v([-3.5, 0, -2.5]), v([4, 2.75, 4.5]),
    // Test 4: contains()
    ...[0, 0, 1, 0, 1, 1, 1, 0, 0].map((c) => v(s(c))),
    // Test 5: center(), extent(), area(), volume(), radius()
    v(d[0]!), v(s(0)), x(0), x(0), x(0),
    v(add(d[0]!, [0, -0.5, 0])), v([0, 1, 0]), x(0), x(0), x(0.5),
    v([0.25, 1.375, 1.0]), v([7.5, 2.75, 7.0]), x(184.75), x(144.375), x(Math.fround(0.5 * Math.sqrt(7.5 * 7.5 + 2.75 * 2.75 + 7.0 * 7.0))),
    // Test 6: intersects()
    xy(1, 1), xy(1, 1), xy(1, 1), xy(1, 1), xy(0, 0), xy(0, 0),
    // Test 7: minDistance(point)
    ...[0, 0, 0, 2, 2.5, 5, 5, 13].map(x),
    // Test 8: minDistance(box)
    ...[0, 0, 0, 0, 1, 5, 5, 13].map(x),
];

gpuTest("FalcorTest.AABB", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Utils/AABBTests.cs.slang", "testAABB");
    const stride = ctx.getStructSize("testData") / 4;
    const data = new Float32Array(kTestData.length * stride);
    kTestData.forEach((p, i) => data.set(p, i * stride));
    ctx.allocateStructuredBuffer("result", 100);
    ctx.allocateStructuredBuffer("testData", kTestData.length, data);
    ctx.vars()["CB"]["n"] = kTestData.length;
    ctx.runProgram(1);
    const r = await ctx.readBuffer("result", Float32Array);
    const e = new Expect();
    kExpected.forEach((c, i) => {
        const got = [r[i * stride]!, r[i * stride + 1]!, r[i * stride + 2]!];
        const want = "v" in c ? c.v.map(Math.fround) : "x" in c ? [Math.fround(c.x)] : c.xy;
        const cmp = got.slice(0, want.length);
        e.check(cmp.every((g, k) => g === want[k]), () => `i = ${i}: (${cmp.join(", ")}) != (${want.join(", ")})`);
    });
    e.done("AABB");
});
