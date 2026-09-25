/**
 * Transplanted FalcorTest GPU tests: Core/LargeBuffer. Each test runs when the adapter's
 * maxBufferSize (copies) or maxStorageBufferBindingSize (shader reads) admits its buffer, and is
 * skipped otherwise, like native's "Disabled due to ... limit" cases. `[root]` buffers bind like others.
 */

import { MemoryType, Mt19937, ResourceBindFlags, type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const GB = 2 ** 30;
const r = new Mt19937();

/** 256 elements of `words` uint32s: a 0xcdcdcdcd default and random test data. */
function testData(device: Device, words: number) {
    const def = new Uint32Array(256 * words).fill(0xcdcdcdcd);
    const data = new Uint32Array(256 * words);
    for (let i = 0; i < 256; i++) data.fill(r.next(), i * words, (i + 1) * words);
    const make = (d: Uint32Array) => device.createBuffer(d.byteLength, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, d);
    return { data, pDefault: make(def), pTest: make(data), testSize: data.byteLength };
}

async function testCopyRegion(device: Device, bufferSize: number): Promise<void> {
    const e = new Expect();
    const ctx = device.renderContext;
    const { data, pDefault, pTest, testSize } = testData(device, 1);
    const readback = device.createBuffer(testSize, ResourceBindFlags.None, MemoryType.ReadBack);
    const buffer = device.createBuffer(bufferSize, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal);
    e.check(buffer.size === bufferSize, () => `size ${buffer.size}`);
    const dstOffset = buffer.size - testSize;
    ctx.copyBufferRegion(buffer, dstOffset, pDefault, 0, testSize);
    ctx.submit(true);
    ctx.copyBufferRegion(buffer, dstOffset, pTest, 0, testSize);
    ctx.submit(true);
    // >4GB: the offset truncated to 32 bits must be a different place.
    if (dstOffset + testSize > 2 ** 32) {
        ctx.copyBufferRegion(buffer, dstOffset % 2 ** 32, pDefault, 0, testSize);
        ctx.submit(true);
    }
    ctx.copyBufferRegion(readback, 0, buffer, dstOffset, testSize);
    ctx.submit(true);
    const result = new Uint32Array((await readback.mapAsync()).slice().buffer);
    for (let i = 0; i < data.length; i++) e.check(result[i] === data[i], () => `i = ${i}: ${result[i]} != ${data[i]}`);
    buffer.destroy();
    e.done("LargeBufferCopyRegion");
}

type ReadKind = "Raw" | "Structured" | "StructuredUint";

async function testRead(device: Device, kind: ReadKind, useRootDesc: boolean, bufferSize: number): Promise<void> {
    const e = new Expect();
    const words = kind === "Structured" ? 4 : 1;
    const elemCount = bufferSize / (words * 4);
    const { data, pDefault, pTest, testSize } = testData(device, words);
    const buffer =
        kind === "Raw"
            ? device.createBuffer(bufferSize, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal)
            : device.createStructuredBuffer(words * 4, elemCount, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, undefined, false);
    const rc = device.renderContext;
    const dstOffset = buffer.size - testSize;
    rc.copyBufferRegion(buffer, dstOffset, pTest, 0, testSize);
    rc.submit(true);
    if (dstOffset + testSize > 2 ** 32) {
        rc.copyBufferRegion(buffer, dstOffset % 2 ** 32, pDefault, 0, testSize);
        rc.submit(true);
    }
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/LargeBuffer.cs.slang", `test${kind === "Raw" ? "ReadRaw" : kind === "Structured" ? "ReadStructured" : "ReadStructuredUint"}`, { USE_ROOT_DESC: useRootDesc ? "1" : "0" });
    ctx.allocateStructuredBuffer("result", 256);
    ctx.vars()[kind === "Raw" ? "buffer" : kind === "Structured" ? "structuredBuffer" : "structuredBufferUint"] = buffer;
    ctx.vars()["CB"]["elemCount"] = elemCount;
    ctx.runProgram(256, 1, 1);
    const result = await ctx.readBuffer("result", Uint32Array);
    for (let i = 0; i < 256; i++) e.check(result[i] === data[i * words], () => `i = ${i}: ${result[i]} != ${data[i * words]}`);
    buffer.destroy();
    e.done(`LargeBufferRead${kind}`);
}

const tooLarge = (limit: "maxBufferSize" | "maxStorageBufferBindingSize", size: number) => (c: { device: Device }) =>
    size > c.device.gpuDevice.limits[limit] ? `needs ${limit} >= ${size} (adapter: ${c.device.gpuDevice.limits[limit]})` : null;

// Native disables the >4GB cases (4GB buffer limit) and the raw/uint SRV cases over 2GB.
const copies: [string, number][] = [["LargeBufferCopyRegion1", 3 * GB], ["LargeBufferCopyRegion2", 4 * GB]];
for (const [name, size] of copies) gpuTest.skipIf(tooLarge("maxBufferSize", size), `FalcorTest.${name}`, ({ device }) => testCopyRegion(device, size));

const reads: [string, ReadKind, boolean, number][] = [
    ["LargeBufferReadRawRoot1", "Raw", true, 3 * GB],
    ["LargeBufferReadRawRoot2", "Raw", true, 4 * GB],
    ["LargeBufferReadStructuredRoot1", "Structured", true, 3 * GB],
    ["LargeBufferReadStructuredRoot2", "Structured", true, 4 * GB],
    ["LargeBufferReadRawSRV1", "Raw", false, 2 * GB],
    ["LargeBufferReadStructuredSRV1", "Structured", false, 2 * GB],
    ["LargeBufferReadStructuredSRV2", "Structured", false, 3 * GB],
    ["LargeBufferReadStructuredSRV3", "Structured", false, 4 * GB - 1024],
    ["LargeBufferReadStructuredUintSRV1", "StructuredUint", false, 2 * GB],
];
for (const [name, kind, root, size] of reads) gpuTest.skipIf(tooLarge("maxStorageBufferBindingSize", size), `FalcorTest.${name}`, ({ device }) => testRead(device, kind, root, size));
