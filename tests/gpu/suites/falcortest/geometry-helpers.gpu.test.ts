/**
 * Transplanted FalcorTest GPU tests: Utils/GeometryHelpersTests. The native-
 * disabled BoxSubtendedConeAngleCenterRandoms (#699, over-conservative cone)
 * stays disabled. Typed float3/float4 buffers are structured buffers here
 * (kernel override), with WGSL's 16-byte float3 stride.
 */

import { Mt19937, type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect, uniformFloat } from "../../harness/expect.js";

const kShaderFilename = "Tests/Utils/GeometryHelpersTests.cs.slang";
const fr = Math.fround;
type V3 = [number, number, number];

/** Packs vectors at a structured-buffer stride (floats per element). */
function pack(v: number[][], strideFloats: number): Float32Array {
    const out = new Float32Array(v.length * strideFloats);
    v.forEach((p, i) => out.set(p, i * strideFloats));
    return out;
}

function structured(device: Device, strideBytes: number, data: Float32Array) {
    return device.createStructuredBuffer(strideBytes, data.byteLength / strideBytes, undefined, data);
}

/** libstdc++ generate_canonical<double, 53>(mt19937): two draws. */
function canonicalDouble(rng: Mt19937): number {
    const sum = rng.next() + rng.next() * 4294967296;
    const r = sum / 18446744073709551616;
    return r >= 1 ? 1 - Number.EPSILON / 2 : r;
}

/** libstdc++ std::normal_distribution<double> (Marsaglia polar, caches the second value). */
function normalDistribution(rng: Mt19937, mean = 0, stddev = 1): () => number {
    let saved: number | null = null;
    return () => {
        let ret: number;
        if (saved !== null) {
            ret = saved;
            saved = null;
        } else {
            let x: number, y: number, r2: number;
            do {
                x = 2 * canonicalDouble(rng) - 1;
                y = 2 * canonicalDouble(rng) - 1;
                r2 = x * x + y * y;
            } while (r2 > 1 || r2 === 0);
            const mult = Math.sqrt((-2 * Math.log(r2)) / r2);
            saved = x * mult;
            ret = y * mult;
        }
        return ret * stddev + mean;
    };
}

/** Correctly rounded f32 fma(a, b, c) for a product exact in double: ties resolve by the addend's residue. */
function fmaF32(a: number, b: number, c: number): number {
    const prod = a * b; // exact: b = 3 * 2^-16
    const sum = prod + c;
    const bb = sum - prod;
    const err = prod - (sum - bb) + (c - bb); // two-sum residue
    const r = fr(sum);
    if (err === 0 || r === sum) return r;
    // At an exact midpoint between two floats the residue decides the direction.
    const other = 2 * sum - r;
    if (fr(other) !== other) return r;
    return err > 0 ? Math.max(r, other) : Math.min(r, other);
}

/** Modified reference from Ray Tracing Gems ch. 6 (the test's offset_ray). */
function offsetRay(p: V3, n: V3, fused = false): V3 {
    const f32 = new Float32Array(1);
    const i32 = new Int32Array(f32.buffer);
    const asint = (v: number) => ((f32[0] = v), i32[0]!);
    const asfloat = (v: number) => ((i32[0] = v), f32[0]!);
    return p.map((pc, c) => {
        const ofi = Math.trunc(fr(768 * n[c]!));
        const pi = asfloat(asint(pc) + (pc < 0 ? -ofi : ofi));
        // fused: fma(n, fScale, p) rounds once.
        return Math.abs(pc) < 1 / 16 ? (fused ? fmaF32(n[c]!, 3 / 65536, pc) : fr(pc + fr((3 / 65536) * n[c]!))) : pi;
    }) as V3;
}

// Native restricts this one to D3D12; it runs here.
gpuTest("FalcorTest.ComputeRayOrigin", async ({ device }) => {
    const nTests = 1 << 16;
    const next = uniformFloat(new Mt19937(), -1, 1);
    const positions: V3[] = [];
    const normals: V3[] = [];
    for (let i = 0; i < nTests; i++) {
        const scale = fr(Math.pow(10, fr(fr(fr(i / nTests) * 60) - 30)));
        const p: V3 = [next(), next(), next()];
        positions.push(p.map((v) => fr(v * scale)) as V3);
        const n: V3 = [next(), next(), next()];
        const s = fr(1 / fr(Math.sqrt(fr(fr(fr(n[0] * n[0]) + fr(n[1] * n[1])) + fr(n[2] * n[2])))));
        normals.push(n.map((v) => fr(v * s)) as V3);
    }
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kShaderFilename, "testComputeRayOrigin");
    const st = ctx.getStructSize("pos") / 4;
    ctx.allocateStructuredBuffer("result", nTests);
    ctx.allocateStructuredBuffer("pos", nTests, pack(positions, st));
    ctx.allocateStructuredBuffer("normal", nTests, pack(normals, st));
    ctx.vars()["CB"]["n"] = nTests;
    ctx.runProgram(nTests);
    const result = await ctx.readBuffer("result", Float32Array);
    // Native compiles with FloatingPointModePrecise (no contraction); WGSL can't forbid
    // FMA, so the small-|p| branch may round once instead of twice.
    const e = new Expect();
    let fusedCount = 0;
    for (let i = 0; i < nTests; i++) {
        const ref = offsetRay(positions[i]!, normals[i]!);
        const fusedRef = offsetRay(positions[i]!, normals[i]!, true);
        const got = [result[i * st]!, result[i * st + 1]!, result[i * st + 2]!];
        const exact = got.every((v, c) => v === ref[c]);
        const fused = got.every((v, c) => v === ref[c] || v === fusedRef[c]);
        if (!exact && fused) fusedCount++;
        e.check(fused, () => `i = ${i}: ${got} != ${ref} (fused ${fusedRef})`);
    }
    console.error(`# ComputeRayOrigin: ${fusedCount} of ${nTests} results match the fused (FMA) rounding`);
    e.done("ComputeRayOrigin");
});

interface BBoxTestCase {
    origin: V3;
    aabbMin: V3;
    aabbMax: V3;
    angle: number;
}

async function runBBoxTest(ctx: GPUUnitTestContext, cases: BBoxTestCase[], entry: string): Promise<void> {
    ctx.createProgram(kShaderFilename, entry);
    const sb = ctx.getStructSize("origin");
    ctx.vars()["origin"] = structured(ctx.device, sb, pack(cases.map((c) => c.origin), sb / 4));
    ctx.vars()["aabbMin"] = structured(ctx.device, sb, pack(cases.map((c) => c.aabbMin), sb / 4));
    ctx.vars()["aabbMax"] = structured(ctx.device, sb, pack(cases.map((c) => c.aabbMax), sb / 4));
    ctx.allocateStructuredBuffer("sinTheta", cases.length);
    ctx.allocateStructuredBuffer("cosTheta", cases.length);
    ctx.allocateStructuredBuffer("coneDir", cases.length);
    ctx.runProgram(cases.length);
}

async function testKnownBBoxes(ctx: GPUUnitTestContext, entry: string): Promise<void> {
    const cases: BBoxTestCase[] = [
        { origin: [0, 0, 0], aabbMin: [-0.5, -0.5, 2], aabbMax: [0.5, 0.5, 3], angle: fr(Math.atan2(fr(fr(Math.sqrt(2)) / 2), 2)) },
        { origin: [0.5, 10, -20], aabbMin: [-0.25, 5, -22], aabbMax: [3, 17, 29], angle: fr(Math.PI) },
    ];
    await runBBoxTest(ctx, cases, entry);
    const sinTheta = await ctx.readBuffer("sinTheta", Float32Array);
    const cosTheta = await ctx.readBuffer("cosTheta", Float32Array);
    const e = new Expect();
    cases.forEach((tc, i) => {
        if (tc.angle === fr(Math.PI)) e.check(sinTheta[i] === 0 && cosTheta[i] === -1, () => `case ${i}: ${sinTheta[i]} ${cosTheta[i]}`);
        else {
            const eps = 1e-4;
            const s = fr(Math.sin(tc.angle));
            const c = fr(Math.cos(tc.angle));
            e.check(s > (1 - eps) * sinTheta[i]! && s < (1 + eps) * sinTheta[i]!, () => `case ${i}: expected sin(theta) = ${s}, got ${sinTheta[i]}`);
            e.check(c > (1 - eps) * cosTheta[i]! && c < (1 + eps) * cosTheta[i]!, () => `case ${i}: expected cos(theta) = ${c}, got ${cosTheta[i]}`);
        }
    });
    e.done(entry);
}

gpuTest("FalcorTest.BoxSubtendedConeAngleCenter", async ({ device }) => testKnownBBoxes(new GPUUnitTestContext(device), "testBoundingConeAngleCenter"));
gpuTest("FalcorTest.BoxSubtendedConeAngleAverage", async ({ device }) => testKnownBBoxes(new GPUUnitTestContext(device), "testBoundingConeAngleAverage"));

gpuTest("FalcorTest.BoxSubtendedConeAngleAverageRandoms", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    const gen = new Mt19937();
    const posAndNeg = () => fr(-100 + 200 * canonicalDouble(gen));
    const pos = () => fr(fr(1e-4) + (100 - fr(1e-4)) * canonicalDouble(gen));
    const cases: BBoxTestCase[] = [];
    for (let i = 0; i < 1 << 16; i++) {
        const origin: V3 = [posAndNeg(), posAndNeg(), posAndNeg()];
        const aabbMin: V3 = [posAndNeg(), posAndNeg(), posAndNeg()];
        const d: V3 = [pos(), pos(), pos()];
        cases.push({ origin, aabbMin, aabbMax: aabbMin.map((v, c) => fr(v + d[c]!)) as V3, angle: 0 });
    }
    await runBBoxTest(ctx, cases, "testBoundingConeAngleAverage");
    const sinTheta = await ctx.readBuffer("sinTheta", Float32Array);
    const cosTheta = await ctx.readBuffer("cosTheta", Float32Array);
    const coneDir = await ctx.readBuffer("coneDir", Float32Array);
    const cs = ctx.getStructSize("coneDir") / 4;
    const norm = (v: number[]) => {
        const l = Math.hypot(v[0]!, v[1]!, v[2]!);
        return v.map((x) => x / l);
    };
    const e = new Expect();
    cases.forEach((b, i) => {
        const inside = b.origin.every((o, c) => o >= b.aabbMin[c]! && o <= b.aabbMax[c]!);
        if (inside) {
            e.check(sinTheta[i] === 0 && cosTheta[i] === -1, () => `i = ${i}: inside but ${sinTheta[i]} ${cosTheta[i]}`);
            return;
        }
        const dir = norm([coneDir[i * cs]!, coneDir[i * cs + 1]!, coneDir[i * cs + 2]!]);
        const ctheta = cosTheta[i]!;
        let minCos = 1;
        for (let j = 0; j < 8; j++) {
            const corner = [j & 1 ? b.aabbMin[0] : b.aabbMax[0], j & 2 ? b.aabbMin[1] : b.aabbMax[1], j & 4 ? b.aabbMin[2] : b.aabbMax[2]];
            const v = norm(corner.map((x, c) => x! - b.origin[c]!));
            const ct = v[0]! * dir[0]! + v[1]! * dir[1]! + v[2]! * dir[2]!;
            e.check(ct > (ctheta > 0 ? 0.99 * ctheta : 1.01 * ctheta), () => `i = ${i} corner ${j}: ${ct} vs ${ctheta}`);
            minCos = Math.min(minCos, ct);
        }
        e.check(minCos < (ctheta > 0 ? 1.01 * ctheta : 0.99 * ctheta), () => `i = ${i}: min ${minCos} vs ${ctheta}`);
    });
    e.done("BoxSubtendedConeAngleAverageRandoms");
});

gpuTest("FalcorTest.SphereSubtendedAngle", async ({ device }) => {
    const cases = [
        { origin: [0, 0, 2], radius: 1, angle: fr(Math.asin(0.5)) },
        { origin: [10, -5, 2], radius: fr(0.1), angle: fr(Math.asin(fr(fr(0.1) / fr(Math.sqrt(129))))) },
        { origin: [0.5, 0, 0], radius: fr(0.51), angle: fr(Math.PI) },
    ];
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kShaderFilename, "testBoundSphereAngle");
    ctx.vars()["spheres"] = structured(device, 16, pack(cases.map((c) => [...c.origin, c.radius]), 4));
    ctx.allocateStructuredBuffer("sinTheta", cases.length);
    ctx.allocateStructuredBuffer("cosTheta", cases.length);
    ctx.runProgram(cases.length);
    const sinTheta = await ctx.readBuffer("sinTheta", Float32Array);
    const cosTheta = await ctx.readBuffer("cosTheta", Float32Array);
    const e = new Expect();
    cases.forEach((tc, i) => {
        if (tc.angle === fr(Math.PI)) e.check(sinTheta[i] === 0 && cosTheta[i] === -1, () => `case ${i}: ${sinTheta[i]} ${cosTheta[i]}`);
        else {
            const eps = 1e-4;
            const s = fr(Math.sin(tc.angle));
            const c = fr(Math.cos(tc.angle));
            // Native's upper sin bound is (1.1 + eps).
            e.check(s > (1 - eps) * sinTheta[i]! && s < (1.1 + eps) * sinTheta[i]!, () => `case ${i}: expected sin(theta) = ${s}, got ${sinTheta[i]}`);
            e.check(c > (1 - eps) * cosTheta[i]! && c < (1 + eps) * cosTheta[i]!, () => `case ${i}: expected cos(theta) = ${c}, got ${cosTheta[i]}`);
        }
    });
    e.done("SphereSubtendedAngle");
});

// ComputeClippedTriangleArea2D

type Tri = { p: [number, number][]; area: number };
const triangleArea = (t: Tri) => {
    const [p0, p1, p2] = t.p as [[number, number], [number, number], [number, number]];
    return fr(0.5 * (-p1[1] * p2[0] + p0[1] * (p2[0] - p1[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]));
};
function isInside(t: Tri, v: [number, number]): boolean {
    const [p0, p1, p2] = t.p as [[number, number], [number, number], [number, number]];
    const s = p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * v[0] + (p0[0] - p2[0]) * v[1];
    const tt = p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * v[0] + (p1[0] - p0[0]) * v[1];
    if (s < 0 !== tt < 0) return false;
    const A = -p1[1] * p2[0] + p0[1] * (p2[0] - p1[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1];
    return A < 0 ? s <= 0 && s + tt >= A : s >= 0 && s + tt <= A;
}

// prettier-ignore
const kFixed: [number[], number][] = [
    [[1.50, 2.50, 1.50, 2.50, 1.50, 2.50], 0], [[2.00, 2.50, 2.00, 2.50, 2.00, 2.50], 0], [[3.00, 2.50, 3.00, 2.50, 3.00, 2.50], 0],
    [[1.25, 2.75, 1.50, 2.25, 1.25, 2.75], 0], [[1.50, 2.75, 1.50, 2.25, 1.50, 2.50], 0], [[1.00, 2.75, 2.00, 2.50, 1.00, 2.75], 0],
    [[1.25, 3.00, 1.50, 3.00, 1.75, 3.00], 0], [[0.50, 3.00, 2.00, 1.00, 0.50, 3.00], 0], [[0.50, 3.00, 2.00, 4.00, 0.50, 3.00], 0],
    [[0.00, 0.00, 1.00, 1.50, 2.00, 1.00], 0], [[2.25, 2.50, 2.75, 2.25, 2.50, 2.00], 0], [[1.00, 3.00, 1.50, 3.25, 2.00, 3.00], 0],
    [[1.75, 1.75, 2.25, 2.25, 2.25, 1.75], 0], [[1.25, 2.25, 1.25, 2.75, 1.75, 2.25], -0.125], [[1.25, 2.75, 1.75, 2.50, 1.50, 2.25], -0.09375],
    [[1.00, 2.00, 1.25, 3.00, 2.00, 2.50], -0.4375], [[1.25, 3.00, 1.50, 3.00, 1.75, 2.00], -0.125], [[2.00, 2.00, 1.00, 2.00, 1.00, 3.00], -0.5],
    [[1.00, 2.25, 1.00, 2.75, 1.50, 2.50], -0.125], [[1.50, 2.25, 2.50, 3.00, 2.50, 2.50], -0.0625], [[0.50, 2.75, 0.50, 3.25, 1.50, 2.75], -0.0625],
    [[1.00, 3.00, 2.00, 1.50, 1.25, 1.50], -0.25], [[1.50, 2.25, 1.50, 2.75, 2.50, 2.25], -0.1875], [[1.25, 2.25, 1.75, 2.75, 1.75, 1.75], -0.21875],
    [[1.75, 1.75, 1.75, 2.75, 2.25, 1.75], -0.125], [[1.25, 2.50, 0.50, 3.25, 2.75, 3.00], -0.375], [[1.00, 2.50, 2.50, 3.50, 1.50, 2.00], -0.5],
    [[1.00, 2.50, 2.50, 3.50, 3.00, 3.00], -0.1875], [[0.50, 2.25, 0.50, 2.75, 2.50, 2.75], -0.25], [[1.25, 1.75, 0.75, 2.75, 1.75, 1.75], -0.109375],
    [[0.00, 0.00, 1.00, 8.00, 3.00, 2.00], -1.0], [[1.25, 2.50, 0.50, 3.25, 2.75, 3.00], -0.375], [[1.50, 2.50, 2.25, 2.50, 1.50, 1.75], -0.21875],
    [[0.75, 2.25, 2.25, 2.75, 1.75, 1.75], -0.46875], [[0.75, 2.25, 1.25, 3.25, 2.25, 2.25], -0.609375], [[1.50, 1.75, 0.75, 2.50, 2.25, 3.25], -0.6875],
    [[0.00, 0.00, 2.00, 1.00, 1.00, 1.50], 0], [[2.75, 2.25, 2.25, 2.50, 2.50, 2.00], 0], [[1.00, 3.00, 2.00, 3.00, 1.50, 3.25], 0],
    [[2.25, 2.25, 1.75, 1.75, 2.25, 1.75], 0], [[1.25, 2.25, 1.75, 2.25, 1.25, 2.75], 0.125], [[1.25, 2.75, 1.50, 2.25, 1.75, 2.50], 0.09375],
    [[1.00, 2.00, 2.00, 2.50, 1.25, 3.00], 0.4375], [[1.50, 3.00, 1.25, 3.00, 1.75, 2.00], 0.125], [[1.00, 2.00, 2.00, 2.00, 1.00, 3.00], 0.5],
    [[1.00, 2.75, 1.00, 2.25, 1.50, 2.50], 0.125], [[2.50, 3.00, 1.50, 2.25, 2.50, 2.50], 0.0625], [[0.50, 3.25, 0.50, 2.75, 1.50, 2.75], 0.0625],
    [[2.00, 1.50, 1.00, 3.00, 1.25, 1.50], 0.25], [[1.50, 2.25, 2.50, 2.25, 1.50, 2.75], 0.1875], [[1.25, 2.25, 1.75, 1.75, 1.75, 2.75], 0.21875],
    [[1.75, 2.75, 1.75, 1.75, 2.25, 1.75], 0.125], [[1.25, 2.50, 2.75, 3.00, 0.50, 3.25], 0.375], [[2.50, 3.50, 1.00, 2.50, 1.50, 2.00], 0.5],
    [[2.50, 3.50, 1.00, 2.50, 3.00, 3.00], 0.1875], [[0.50, 2.25, 2.50, 2.75, 0.50, 2.75], 0.25], [[0.75, 2.75, 1.25, 1.75, 1.75, 1.75], 0.109375],
    [[0.00, 0.00, 3.00, 2.00, 1.00, 8.00], 1.0], [[0.50, 3.25, 1.25, 2.50, 2.75, 3.00], 0.375], [[1.50, 2.50, 1.50, 1.75, 2.25, 2.50], 0.21875],
    [[2.25, 2.75, 0.75, 2.25, 1.75, 1.75], 0.46875], [[1.25, 3.25, 0.75, 2.25, 2.25, 2.25], 0.609375], [[0.75, 2.50, 1.50, 1.75, 2.25, 3.25], 0.6875],
];

gpuTest("FalcorTest.ComputeClippedTriangleArea2D", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram(kShaderFilename, "testComputeClippedTriangleArea2D");
    const sp = ctx.getStructSize("pos");
    const e = new Expect();
    const run = async (tests: Tri[], aabb: { min: [number, number]; max: [number, number] }[], threshold: number, desc: string) => {
        ctx.allocateStructuredBuffer("result", tests.length);
        ctx.vars()["pos"] = structured(device, sp, pack(tests.flatMap((t) => t.p.map((p) => [...p, 0])), sp / 4));
        ctx.vars()["aabb"] = structured(device, 16, pack(aabb.map((b) => [...b.min, ...b.max]), 4));
        ctx.vars()["CB"]["n"] = tests.length;
        ctx.runProgram(tests.length);
        const result = await ctx.readBuffer("result", Float32Array);
        const rs = ctx.getStructSize("result") / 4;
        tests.forEach((t, i) => {
            const got = result[i * rs]!;
            const b = aabb[i]!;
            const maxErr = fr(fr(b.max[0] - b.min[0]) * fr(b.max[1] - b.min[1])) * threshold;
            e.check(Number.isFinite(got) && Math.abs(got - t.area) <= maxErr, () => `${desc} i=${i}: returned ${got}, expected ${t.area}`);
        });
    };
    const fixed: Tri[] = kFixed.map(([p, area]) => ({ p: [[p[0]!, p[1]!], [p[2]!, p[3]!], [p[4]!, p[5]!]], area }));
    await run(fixed, fixed.map(() => ({ min: [1, 2], max: [2, 3] })), 1e-6, "Fixed test");

    const r = new Mt19937();
    const d = normalDistribution(r);
    const f = (): [number, number] => [fr(d()), fr(d())];
    const tests: Tri[] = [];
    const aabb: { min: [number, number]; max: [number, number] }[] = [];
    for (let i = 0; i < 10000; i++) {
        const min = f();
        const max = f();
        if (min[0] > max[0]) [min[0], max[0]] = [max[0], min[0]];
        if (min[1] > max[1]) [min[1], max[1]] = [max[1], min[1]];
        const t: Tri = { p: [f(), f(), f()], area: 0 };
        const m = 100;
        let hits = 0;
        for (let j = 0; j < m; j++) {
            const y = fr(min[1] + fr(max[1] - min[1]) * fr((j + 0.5) / m));
            for (let k = 0; k < m; k++) {
                const x = fr(min[0] + fr(max[0] - min[0]) * fr((k + 0.5) / m));
                if (isInside(t, [x, y])) hits++;
            }
        }
        const sign = triangleArea(t) >= 0 ? 1 : -1;
        t.area = fr(sign * fr(fr(max[0] - min[0]) * fr(max[1] - min[1])) * fr(hits / (m * m)));
        tests.push(t);
        aabb.push({ min, max });
    }
    await run(tests, aabb, 0.01, "Random test");
    e.done("ComputeClippedTriangleArea2D");
});
