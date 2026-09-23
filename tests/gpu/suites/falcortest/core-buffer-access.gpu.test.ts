/**
 * Transplanted FalcorTest GPU tests: Core/BufferAccessTests. Readback maps are
 * asynchronous on the web (Buffer.mapAsync()); everything else is as native.
 */

import { MemoryType, ResourceBindFlags, type Buffer, type Device } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { getElements } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kElementCount = 256;
const kTestData = Uint32Array.from({ length: kElementCount }, (_, i) => i);

const createTestBuffer = (device: Device, memoryType: MemoryType, initialize: boolean) =>
    device.createBuffer(kElementCount * 4, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType, initialize ? kTestData : undefined);

function checkData(e: Expect, data: Uint32Array, what: string): void {
    for (let i = 0; i < kElementCount; i++) e.check(data[i] === i, () => `${what}: i = ${i}: ${data[i]}`);
}

function initBufferIndirect(device: Device, buffer: Buffer): void {
    device.renderContext.copyBuffer(buffer, createTestBuffer(device, MemoryType.DeviceLocal, true));
}

async function checkBufferIndirect(device: Device, e: Expect, buffer: Buffer, what: string): Promise<void> {
    const result = createTestBuffer(device, MemoryType.DeviceLocal, false);
    device.renderContext.copyBuffer(result, buffer);
    checkData(e, await getElements(result, Uint32Array), what);
}

gpuTest("FalcorTest.BufferDeviceLocalWrite", async ({ device }) => {
    const e = new Expect();
    const a = createTestBuffer(device, MemoryType.DeviceLocal, false);
    a.setBlob(kTestData);
    await checkBufferIndirect(device, e, a, "setBlob");
    await checkBufferIndirect(device, e, createTestBuffer(device, MemoryType.DeviceLocal, true), "init data");
    e.done("BufferDeviceLocalWrite");
});

gpuTest("FalcorTest.BufferDeviceLocalRead", async ({ device }) => {
    const e = new Expect();
    const b = createTestBuffer(device, MemoryType.DeviceLocal, false);
    initBufferIndirect(device, b);
    checkData(e, await getElements(b, Uint32Array), "getBlob");
    e.done("BufferDeviceLocalRead");
});

gpuTest("FalcorTest.BufferUploadWrite", async ({ device }) => {
    const e = new Expect();
    const a = createTestBuffer(device, MemoryType.Upload, false);
    a.setBlob(kTestData);
    await checkBufferIndirect(device, e, a, "setBlob");
    await checkBufferIndirect(device, e, createTestBuffer(device, MemoryType.Upload, true), "init data");
    e.done("BufferUploadWrite");
});

gpuTest("FalcorTest.BufferUploadMap", async ({ device }) => {
    const e = new Expect();
    const b = createTestBuffer(device, MemoryType.Upload, false);
    new Uint32Array(b.map().buffer).set(kTestData);
    b.unmap();
    await checkBufferIndirect(device, e, b, "map/unmap");
    e.done("BufferUploadMap");
});

gpuTest("FalcorTest.BufferReadbackRead", async ({ device }) => {
    const e = new Expect();
    const b = createTestBuffer(device, MemoryType.ReadBack, false);
    initBufferIndirect(device, b);
    checkData(e, await getElements(b, Uint32Array), "getBlob");
    e.done("BufferReadbackRead");
});

gpuTest("FalcorTest.BufferReadbackMap", async ({ device }) => {
    const e = new Expect();
    const b = createTestBuffer(device, MemoryType.ReadBack, false);
    initBufferIndirect(device, b);
    const data = await b.mapAsync();
    checkData(e, new Uint32Array(data.slice().buffer), "mapAsync");
    b.unmap();
    e.done("BufferReadbackMap");
});
