/**
 * Transplanted FalcorTest GPU tests: Core/RootBufferTests, RootBufferStructTests and
 * RootBufferParamBlockTests. WebGPU has no root descriptors, so `[root]` buffers bind
 * like any other; the tests still check that they coexist with the other resources.
 * Typed buffers run on structured buffers, and the parameter block's resource arrays
 * are unrolled into members (WGSL has no arrays of resources). Shader models 6.0 and
 * 6.3 compile to the same WGSL target.
 */

import { ComputePass, MemoryType, Mt19937, ResourceBindFlags, StandaloneParameterBlock, type Buffer, type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kNumElems = 256;
const fr = Math.fround;

function makeRandom(): () => number {
    const rng = new Mt19937();
    return () => rng.next() % 101;
}

const rawBuffer = (device: Device, data: Uint32Array, uav: boolean) =>
    device.createBuffer(data.byteLength, uav ? ResourceBindFlags.UnorderedAccess : ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, data);
const structured = (device: Device, stride: number, data: ArrayBufferView, uav = false) =>
    device.createStructuredBuffer(stride, data.byteLength / stride, uav ? ResourceBindFlags.UnorderedAccess : ResourceBindFlags.ShaderResource, data);
const fill = (next: () => number) => Uint32Array.from({ length: kNumElems }, next);

async function testRootBuffer(device: Device, useUav: boolean): Promise<void> {
    const next = makeRandom();
    const e = new Expect();
    let c0 = 31;
    const c1 = 2.5;
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/RootBufferTests.cs.slang", "main", { USE_UAV: useUav ? 1 : 0 });
    ctx.allocateStructuredBuffer("result", kNumElems);
    const v = ctx.vars();
    v["CB"]["c0"] = c0;
    v["CB"]["c1"] = c1;

    let raw = fill(next);
    v["rawBuffer"] = rawBuffer(device, raw, false);
    const structA = new Float32Array(kNumElems);
    const structB = new Uint32Array(kNumElems);
    const structData = new ArrayBuffer(kNumElems * 8);
    for (let i = 0; i < kNumElems; i++) {
        structA[i] = next() + 0.5;
        structB[i] = next();
        new Float32Array(structData, i * 8, 1)[0] = structA[i]!;
        new Uint32Array(structData, i * 8 + 4, 1)[0] = structB[i]!;
    }
    v["structBuffer"] = structured(device, 8, new Uint8Array(structData));
    const typedUint = fill(next);
    v["typedBufferUint"] = structured(device, 4, typedUint, true);
    let typedFloat4 = new Float32Array(kNumElems * 4);
    const newFloat4 = () => {
        typedFloat4 = new Float32Array(kNumElems * 4);
        for (let i = 0; i < kNumElems; i++) typedFloat4.set([next() * 0.25, next() * 0.5, next() * 0.75, next()], i * 4);
        v["typedBufferFloat4"] = structured(device, 16, typedFloat4);
    };
    newFloat4();
    let test = new Uint32Array(0);
    const newTestBuffer = () => {
        test = fill(next);
        const buffer = rawBuffer(device, test, useUav);
        v["testBuffer"] = buffer;
        e.check(v["testBuffer"].getBuffer() === buffer, () => "bound root buffer differs");
    };
    newTestBuffer();

    const verify = async (step: string) => {
        const result = await ctx.readBuffer("result", Float32Array);
        for (let i = 0; i < kNumElems; i++) {
            let r = 0;
            for (const t of [c0, c1, raw[i]!, typedUint[i]! * 2, fr(typedFloat4[i * 4 + 2]! * 3), fr(structA[i]! * 4), structB[i]! * 5, test[i]! * 6]) r = fr(r + t);
            e.check(result[i] === r, () => `i = ${i} (${step}): ${result[i]} != ${r}`);
        }
    };
    ctx.runProgram(kNumElems, 1, 1);
    await verify("step 1");

    // Change some buffers and a constant; the root buffer stays bound.
    raw = fill(next);
    v["rawBuffer"] = rawBuffer(device, raw, false);
    newFloat4();
    v["CB"]["c0"] = ++c0;
    ctx.runProgram(kNumElems, 1, 1);
    await verify("step 2");

    newTestBuffer();
    ctx.runProgram(kNumElems, 1, 1);
    await verify("step 3");
    e.done(`RootBuffer${useUav ? "UAV" : "SRV"}`);
}

async function testRootBufferInStruct(device: Device, useUav: boolean): Promise<void> {
    const next = makeRandom();
    const e = new Expect();
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/RootBufferStructTests.cs.slang", "main", { USE_UAV: useUav ? 1 : 0 });
    ctx.allocateStructuredBuffer("result", kNumElems);
    const data = ctx.vars()["CB"]["data"];
    const buf = fill(next);
    data["buf"] = structured(device, 4, buf);
    const rwBuf = fill(next);
    data["rwBuf"] = structured(device, 4, rwBuf, true);
    const rootBuf = fill(next);
    const pRootBuffer = structured(device, 4, rootBuf, useUav);
    data["rootBuf"] = pRootBuffer;
    e.check(data["rootBuf"].getBuffer() === pRootBuffer, () => "bound root buffer differs");
    ctx.runProgram(kNumElems, 1, 1);
    const result = await ctx.readBuffer("result", Uint32Array);
    for (let i = 0; i < kNumElems; i++) {
        const r = buf[i]! + rwBuf[i]! * 2 + rootBuf[i]! * 3;
        e.check(result[i] === r, () => `i = ${i}: ${result[i]} != ${r}`);
    }
    e.done(`RootBufferStruct${useUav ? "UAV" : "SRV"}`);
}

async function testRootBufferParamBlock(device: Device, useUav: boolean): Promise<void> {
    const next = makeRandom();
    const e = new Expect();
    const defines = { USE_UAV: useUav ? 1 : 0 };
    const reflectionPass = ComputePass.create(device, { path: "Tests/Core/ParamBlockReflection.cs.slang", csEntry: "main", defines });
    const blockReflection = reflectionPass.getReflector().getParameterBlock("gParamBlock");
    e.check(blockReflection !== undefined, () => "no gParamBlock reflection");
    const paramBlock = StandaloneParameterBlock.create(device, blockReflection!);
    const block = paramBlock.getRootVar();
    const c0 = next();
    block["c0"] = c0;
    const bufA = [0, 1].map(() => fill(next));
    bufA.forEach((d, j) => (block[`bufA_${j}`] = rawBuffer(device, d, false)));
    const bufB = [0, 1, 2].map(() => Float32Array.from(fill(next)));
    bufB.forEach((d, j) => (block[`bufB_${j}`] = structured(device, 4, d, true)));
    const bufC = [0, 1, 2, 3].map(() => fill(next));
    bufC.forEach((d, j) => (block[`bufC_${j}`] = structured(device, 4, d)));
    const test = fill(next);
    const pTestBuffer: Buffer = rawBuffer(device, test, useUav);
    paramBlock.setBuffer("testBuffer", pTestBuffer);
    e.check(paramBlock.getBuffer("testBuffer") === pTestBuffer, () => "bound root buffer differs");

    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/RootBufferParamBlockTests.cs.slang", "main", defines);
    ctx.allocateStructuredBuffer("result", kNumElems);
    const v = ctx.vars();
    v["gParamBlock"] = paramBlock;
    const globalBufA = fill(next);
    v["globalBufA"] = structured(device, 4, globalBufA);
    const globalTest = fill(next);
    v["globalTestBuffer"] = rawBuffer(device, globalTest, useUav);
    ctx.runProgram(kNumElems, 1, 1);
    const result = await ctx.readBuffer("result", Float32Array);
    for (let i = 0; i < kNumElems; i++) {
        const terms = [c0, bufA[0]![i]!, bufA[1]![i]! * 2, ...bufB.map((b, j) => b[i]! * (3 + j)), ...bufC.map((b, j) => b[i]! * (6 + j)), test[i]! * 10, globalBufA[i]! * 11, globalTest[i]! * 12];
        const r = terms.reduce((acc, t) => fr(acc + t), 0);
        e.check(result[i] === r, () => `i = ${i}: ${result[i]} != ${r}`);
    }
    e.done(`RootBufferParamBlock${useUav ? "UAV" : "SRV"}`);
}

for (const sm of ["6_0", "6_3"]) {
    for (const uav of [false, true]) {
        const kind = uav ? "UAV" : "SRV";
        gpuTest(`FalcorTest.RootBuffer${kind}_${sm}`, ({ device }) => testRootBuffer(device, uav));
        gpuTest(`FalcorTest.RootBufferStruct${kind}_${sm}`, ({ device }) => testRootBufferInStruct(device, uav));
        gpuTest(`FalcorTest.RootBufferParamBlock${kind}_${sm}`, ({ device }) => testRootBufferParamBlock(device, uav));
    }
}
