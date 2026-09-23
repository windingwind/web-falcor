/**
 * Transplanted FalcorTest GPU tests: Core/BufferTests. TypedBuffer runs on a
 * structured buffer (no typed buffers in WGSL); BufferStrides needs int16/
 * float64 struct members and is not portable.
 */

import { MemoryType, ResourceBindFlags, ResourceFormat, type Buffer } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext, getElements } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kIterations = 4;
enum Type {
    ByteAddressBuffer = 0,
    TypedBuffer = 1,
    StructuredBuffer = 2,
}

async function testBuffer(ctx: GPUUnitTestContext, e: Expect, type: Type, numElemsIn: number, index = 0, count = 0): Promise<void> {
    const device = ctx.device;
    const numElems = Math.ceil(numElemsIn / 256) * 256;
    const blob = Uint32Array.from({ length: count }, (_, i) => (count - i) * 3);
    const defines = { TYPE: type };
    ctx.createProgram("Tests/Core/BufferTests.cs.slang", "clearBuffer", defines);
    let buffer: Buffer;
    if (type === Type.ByteAddressBuffer) buffer = device.createBuffer(numElems * 4, ResourceBindFlags.UnorderedAccess, MemoryType.DeviceLocal);
    else if (type === Type.TypedBuffer) buffer = device.createTypedBuffer(ResourceFormat.R32Uint, numElems, ResourceBindFlags.UnorderedAccess);
    else buffer = device.createStructuredBuffer(ctx.getStructSize("buffer"), numElems, ResourceBindFlags.UnorderedAccess);
    ctx.vars()["buffer"] = buffer;
    ctx.runProgram(numElems, 1, 1);
    ctx.createProgram("Tests/Core/BufferTests.cs.slang", "updateBuffer", defines);
    ctx.vars()["buffer"] = buffer;
    for (let i = 0; i < kIterations; i++) ctx.runProgram(numElems, 1, 1);
    if (count > 0) buffer.setBlob(blob, index * 4);
    for (let i = 0; i < kIterations; i++) ctx.runProgram(numElems, 1, 1);
    ctx.createProgram("Tests/Core/BufferTests.cs.slang", "readBuffer", defines);
    ctx.allocateStructuredBuffer("result", numElems);
    ctx.vars()["buffer"] = buffer;
    ctx.runProgram(numElems, 1, 1);
    const result = await ctx.readBuffer("result", Uint32Array);
    for (let i = 0; i < numElems; i++) {
        let expected = (i + 1) * kIterations * 2;
        if (i >= index && i < index + count) expected = blob[i - index]! + (i + 1) * kIterations;
        e.check(result[i] === expected, () => `i = ${i}: ${result[i]} != ${expected} (numElems = ${numElems} index = ${index} count = ${count})`);
    }
}

for (const [name, type] of [["RawBuffer", Type.ByteAddressBuffer], ["TypedBuffer", Type.TypedBuffer], ["StructuredBuffer", Type.StructuredBuffer]] as const) {
    gpuTest(`FalcorTest.${name}`, async ({ device }) => {
        const ctx = new GPUUnitTestContext(device);
        const e = new Expect();
        for (let numElems = 1 << 8; numElems <= 1 << 16; numElems <<= 4) {
            await testBuffer(ctx, e, type, numElems, 0, 0);
            await testBuffer(ctx, e, type, numElems, 0, 1);
            await testBuffer(ctx, e, type, numElems, 0, numElems / 2);
            await testBuffer(ctx, e, type, numElems, 1, 1);
            await testBuffer(ctx, e, type, numElems, numElems / 2 + 3, numElems / 4 - 1);
        }
        e.done(name);
    });
}

gpuTest("FalcorTest.BufferUpdate", async ({ device }) => {
    const a = Uint32Array.of(1, 2, 3, 4);
    const b = Uint32Array.of(5, 6, 7, 8);
    const buffer = device.createBuffer(16);
    const e = new Expect();
    buffer.setBlob(a);
    const resA = await getElements(buffer, Uint32Array);
    e.check(resA.join() === a.join(), () => `A: ${resA}`);
    buffer.setBlob(b);
    const resB = await getElements(buffer, Uint32Array);
    e.check(resB.join() === b.join(), () => `B: ${resB}`);
    e.done("BufferUpdate");
});

gpuTest("FalcorTest.BufferWrite", async ({ device }) => {
    const e = new Expect();
    const testWrite = async (testData: Uint32Array, useInitData: boolean) => {
        const bufA = device.createBuffer(16, ResourceBindFlags.None, MemoryType.Upload, useInitData ? testData : undefined);
        const bufB = device.createBuffer(16, ResourceBindFlags.None);
        if (!useInitData) {
            new Uint32Array(bufA.map().buffer, 0, 4).set(testData);
            bufA.unmap();
        }
        device.renderContext.copyBuffer(bufB, bufA);
        const r1 = await getElements(bufB, Uint32Array);
        e.check(r1.join() === testData.join(), () => `copy ${r1} != ${testData}`);
        const testData2 = testData.map((v) => v * 10);
        new Uint32Array(bufA.map().buffer, 0, 4).set(testData2);
        bufA.unmap();
        device.renderContext.copyBuffer(bufB, bufA);
        const r2 = await getElements(bufB, Uint32Array);
        e.check(r2.join() === testData2.join(), () => `remap ${r2} != ${testData2}`);
    };
    await testWrite(Uint32Array.of(1, 2, 3, 4), false);
    await testWrite(Uint32Array.of(3, 4, 5, 6), true);
    e.done("BufferWrite");
});
