/** Transplanted FalcorTest GPU tests: Utils/BufferAllocatorTests. */

import { BufferAllocator, ResourceFormat } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { getElements } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const f32 = (...v: number[]) => Float32Array.from(v);
const u32 = (...v: number[]) => Uint32Array.from(v);
/** struct S { float a; float b; uint32_t c; } */
const S = (a: number, b: number, c: number) => {
    const buf = new ArrayBuffer(12);
    new Float32Array(buf, 0, 2).set([a, b]);
    new Uint32Array(buf, 8, 1)[0] = c;
    return new Uint8Array(buf);
};

gpuTest("FalcorTest.BufferAllocatorNoAlign", async ({ device }) => {
    const e = new Expect();
    const buf = new BufferAllocator(0, 0, 0);
    e.check(buf.getSize() === 0, () => "initial size");
    let offset = buf.allocate(4);
    buf.set(offset, u32(11));
    offset = buf.allocate(4);
    buf.set(offset, u32(99));
    offset = buf.allocate(16);
    buf.set(offset, f32(1.1, 2.5, 13.3, -1.2));
    e.check(offset === 8, () => `float4 offset ${offset}`);
    buf.emplaceBack(f32(18.4));
    buf.emplaceBack(Uint16Array.of(33391));
    e.check(buf.getSize() === 30, () => `size ${buf.getSize()}`);
    offset = buf.allocate(25 * 4);
    e.check(offset === 30, () => `array offset ${offset}`);
    buf.setBlob(Float32Array.from({ length: 25 }, (_, i) => i + 0.11), offset);
    buf.emplaceBack(S(5.9, 3.3, 19));
    buf.pushBack(S(9.1, 10, 333));
    e.check(buf.getSize() === 154, () => `size ${buf.getSize()}`);

    const view = (p: Uint8Array) => new DataView(p.buffer, p.byteOffset, p.byteLength);
    const v = view(buf.getStartPointer());
    const fr = Math.fround;
    const checks: [number, "u32" | "f32" | "u16", number][] = [
        [0, "u32", 11], [4, "u32", 99], [8, "f32", 1.1], [12, "f32", 2.5], [16, "f32", 13.3], [20, "f32", -1.2], [24, "f32", 18.4], [28, "u16", 33391],
        [130, "f32", 5.9], [134, "f32", 3.3], [138, "u32", 19], [142, "f32", 9.1], [146, "f32", 10], [150, "u32", 333],
    ];
    const get = (o: number, t: "u32" | "f32" | "u16") => (t === "u32" ? v.getUint32(o, true) : t === "u16" ? v.getUint16(o, true) : v.getFloat32(o, true));
    for (const [o, t, want] of checks) e.check(get(o, t) === (t === "f32" ? fr(want) : want), () => `offset ${o}: ${get(o, t)}`);
    for (let i = 0; i < 25; i++) e.check(v.getFloat32(30 + 4 * i, true) === fr(i + 0.11), () => `array ${i}`);

    const validate = async (what: string) => {
        const gpu = buf.getGPUBuffer(device)!;
        e.check(gpu.structSize === 0 && gpu.format === ResourceFormat.Unknown && gpu.size === 156, () => `${what}: buffer size ${gpu.size}`);
        const data = await getElements(gpu, Uint8Array);
        const ref = buf.getStartPointer();
        for (let i = 0; i < buf.getSize(); i++) e.check(data[i] === ref[i], () => `${what}: byte ${i}: ${data[i]} != ${ref[i]}`);
    };
    await validate("initial");
    const p = view(buf.getStartPointer());
    p.setFloat32(24, 55.4, true);
    buf.modified(24, 4);
    p.setFloat32(130, 0.004, true);
    buf.modified(130, 4);
    await validate("modified");
    e.done("BufferAllocatorNoAlign");
});

gpuTest("FalcorTest.BufferAllocatorAlign", async ({ device }) => {
    const e = new Expect();
    const buf = new BufferAllocator(16, 0, 128);
    const steps: [number, number, number][] = [
        [20, 0, 20], [4, 32, 36], [100, 128, 228], [4, 240, 244], [128, 256, 384], [590, 384, 974], [20, 976, 996], [24, 1024, 1048], [130, 1056, 1186],
    ];
    for (const [size, wantOffset, wantSize] of steps) {
        const offset = buf.allocate(size);
        e.check(offset === wantOffset && buf.getSize() === wantSize, () => `allocate(${size}): offset ${offset}, size ${buf.getSize()}`);
    }
    const gpu = buf.getGPUBuffer(device)!;
    e.check(gpu.structSize === 0 && gpu.size === 1188, () => `buffer size ${gpu.size}`);
    e.done("BufferAllocatorAlign");
});

gpuTest("FalcorTest.BufferAllocatorStructNoAlign", async ({ device }) => {
    const e = new Expect();
    const buf = new BufferAllocator(0, 16, 0);
    buf.emplaceBack(f32(1, 2, 3, 4));
    buf.emplaceBack(f32(5, 6, 7, 8));
    buf.emplaceBack(f32(9, 10, 11, 12));
    buf.emplaceBack(f32(13));
    buf.pushBack(f32(14, 15, 16, 17));
    e.check(buf.getSize() === 68, () => `size ${buf.getSize()}`);
    const cpu = new Float32Array(buf.getStartPointer().slice().buffer);
    for (let i = 0; i < 17; i++) e.check(cpu[i] === i + 1, () => `cpu ${i}`);
    const gpu = buf.getGPUBuffer(device)!;
    e.check(gpu.structSize === 16 && gpu.size === 80, () => `buffer ${gpu.structSize}/${gpu.size}`);
    const data = await getElements(gpu, Float32Array);
    for (let i = 0; i < 17; i++) e.check(data[i] === i + 1, () => `gpu ${i}: ${data[i]}`);
    e.done("BufferAllocatorStructNoAlign");
});

gpuTest("FalcorTest.BufferAllocatorStructAlign", async ({ device }) => {
    const e = new Expect();
    const buf = new BufferAllocator(16, 32, 128);
    let offset = buf.emplaceBack(f32(1.2));
    e.check(offset === 0 && buf.getSize() === 4, () => `emplace ${offset}/${buf.getSize()}`);
    offset = buf.pushBack(f32(3.4));
    e.check(offset === 16 && buf.getSize() === 20, () => `push ${offset}/${buf.getSize()}`);
    offset = buf.allocate(12);
    buf.set(offset, f32(4.7));
    buf.set(offset + 4, f32(5.7));
    buf.set(offset + 8, f32(6.7));
    e.check(offset === 32 && buf.getSize() === 44, () => `allocate(12) ${offset}/${buf.getSize()}`);
    offset = buf.allocate(84);
    buf.setBlob(Float32Array.from({ length: 21 }, (_, i) => i + 0.5), offset);
    e.check(offset === 128 && buf.getSize() === 212, () => `allocate(84) ${offset}/${buf.getSize()}`);
    const cpu = new Float32Array(buf.getStartPointer().slice().buffer);
    const fr = Math.fround;
    e.check(cpu[0] === fr(1.2) && cpu[4] === fr(3.4) && cpu[8] === fr(4.7) && cpu[9] === fr(5.7) && cpu[10] === fr(6.7), () => `cpu values`);
    for (let i = 0; i < 21; i++) e.check(cpu[32 + i] === i + 0.5, () => `cpu array ${i}`);
    const gpu = buf.getGPUBuffer(device)!;
    e.check(gpu.structSize === 32 && gpu.size === 224, () => `buffer ${gpu.structSize}/${gpu.size}`);
    const data = await getElements(gpu, Float32Array);
    for (let i = 0; i < buf.getSize() / 4; i++) e.check(data[i] === cpu[i], () => `gpu ${i}: ${data[i]} != ${cpu[i]}`);
    e.done("BufferAllocatorStructAlign");
});
