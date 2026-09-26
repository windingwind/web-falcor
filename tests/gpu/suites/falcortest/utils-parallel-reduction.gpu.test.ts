/**
 * Transplanted FalcorTest Utils/ParallelReductionTests: Sum and MinMax over native's twelve
 * formats and sizes, against CPU references on the same quantized data, plus the result-buffer copy.
 */

import { FormatType, MemoryType, Mt19937, ParallelReduction, ParallelReductionType, ResourceBindFlags, ResourceFormat, float16ToFloat32, float32ToFloat16, getFormatBytesPerBlock, getFormatChannelCount, getFormatType } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { Expect, uniformFloat } from "../../harness/expect.js";

const f32 = Math.fround;

// unorm_t / snorm_t quantization (float arithmetic, then the integer cast truncates).
const unorm = (v: number, bits: number) => {
    const scale = 2 ** bits - 1;
    return Math.fround(Math.trunc(Math.fround(Math.fround(Math.min(Math.max(v, 0), 1) * scale) + 0.5)) / scale);
};
const snorm = (v: number, bits: number) => {
    const scale = 2 ** (bits - 1) - 1;
    return Math.fround(Math.trunc(Math.fround(Math.fround(Math.min(Math.max(v, -1), 1) * scale) + (v >= 0 ? 0.5 : -0.5))) / scale);
};

async function testReduction(device: any, e: Expect, reduction: ParallelReduction, format: ResourceFormat, width: number, height: number): Promise<void> {
    const channels = getFormatChannelCount(format);
    const elems = width * height * channels;
    const sz = getFormatBytesPerBlock(format) / channels;
    const type = getFormatType(format);
    const u = uniformFloat(new Mt19937());
    const f = Math.fround;

    const data = new ArrayBuffer(elems * sz);
    const values = new Float64Array(elems);
    for (let i = 0; i < elems; i++) {
        let value = 0;
        if (type === FormatType.Float) {
            value = f(f(u() * 200) - 100);
            if (sz === 2) new Uint16Array(data)[i] = float32ToFloat16(value), (value = float16ToFloat32(float32ToFloat16(value)));
            else new Float32Array(data)[i] = value;
        } else if (type === FormatType.Sint) {
            value = Math.trunc(f(f(u() * 200) - 100));
            (sz === 1 ? new Int8Array(data) : sz === 2 ? new Int16Array(data) : new Int32Array(data))[i] = value;
        } else if (type === FormatType.Uint) {
            value = Math.trunc(f(u() * 200));
            (sz === 1 ? new Uint8Array(data) : sz === 2 ? new Uint16Array(data) : new Uint32Array(data))[i] = value;
        } else if (type === FormatType.Unorm) {
            value = unorm(u(), sz * 8);
            (sz === 1 ? new Uint8Array(data) : new Uint16Array(data))[i] = Math.round(value * (2 ** (sz * 8) - 1));
        } else {
            value = snorm(f(f(u() * 2) - 1), sz * 8);
            (sz === 1 ? new Int8Array(data) : new Int16Array(data))[i] = Math.round(value * (2 ** (sz * 8 - 1) - 1));
        }
        values[i] = value;
    }
    const refSum = [0, 0, 0, 0], absSum = [0, 0, 0, 0];
    const refMin = [Infinity, Infinity, Infinity, Infinity], refMax = [-Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < elems; i++) {
        const c = i % channels, v = values[i]!;
        refSum[c]! += v;
        absSum[c]! += Math.abs(v);
        refMin[c] = Math.min(refMin[c]!, v);
        refMax[c] = Math.max(refMax[c]!, v);
    }
    const isFloat = type !== FormatType.Uint && type !== FormatType.Sint;
    const label = `${ResourceFormat[format]} ${width}x${height}`;
    const texture = device.createTexture2D(width, height, format, 1, 1, new Uint8Array(data));
    const read = async (buffer: any, n: number) => {
        const bytes = await device.renderContext.readBuffer(buffer, 0, n * 4);
        return type === FormatType.Uint ? new Uint32Array(bytes.buffer, 0, n) : type === FormatType.Sint ? new Int32Array(bytes.buffer, 0, n) : new Float32Array(bytes.buffer, 0, n);
    };

    // Sum: the returned result equals the copy in the result buffer, and the CPU reference.
    {
        const resultBuffer = device.createBuffer(16, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, new Uint8Array(16));
        const result = await reduction.execute(device.renderContext, texture, ParallelReductionType.Sum, resultBuffer, 0);
        const copy = await read(resultBuffer, 4);
        for (let i = 0; i < 4; i++) e.check(copy[i] === result[i], () => `${label} sum buffer[${i}] = ${copy[i]} vs ${result[i]}`);
        for (let i = 0; i < 4; i++) {
            if (i >= channels) e.check(result[i] === 0, () => `${label} sum[${i}] = ${result[i]} (expected 0)`);
            else if (isFloat) {
                const relError = Math.abs(result[i]! - refSum[i]!) / absSum[i]!;
                e.check(relError <= 1e-6, () => `${label} sum[${i}] relError ${relError}`);
            } else e.check(result[i] === refSum[i], () => `${label} sum[${i}] = ${result[i]} vs ${refSum[i]}`);
        }
    }
    // MinMax: exact.
    {
        const resultBuffer = device.createBuffer(32, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, new Uint8Array(32));
        const result = await reduction.execute(device.renderContext, texture, ParallelReductionType.MinMax, resultBuffer, 0);
        const copy = await read(resultBuffer, 8);
        for (let i = 0; i < 8; i++) e.check(copy[i] === result[i], () => `${label} minmax buffer[${i}] = ${copy[i]} vs ${result[i]}`);
        for (let i = 0; i < channels; i++) {
            e.check(result[i] === refMin[i], () => `${label} min[${i}] = ${result[i]} vs ${refMin[i]}`);
            e.check(result[4 + i] === refMax[i], () => `${label} max[${i}] = ${result[4 + i]} vs ${refMax[i]}`);
        }
    }
    texture.destroy();
}

gpuTest("FalcorTest.ParallelReduction", async ({ device }) => {
    // Quick test of the snorm/unorm quantization.
    const e = new Expect();
    e.check(unorm(163.499 / 255, 8) === f32(163 / 255) && unorm(163.501 / 65535, 16) === f32(164 / 65535), () => "unorm quantization");
    e.check(snorm(10.499 / 127, 8) === f32(10 / 127) && snorm(10.501 / 127, 8) === f32(11 / 127) && snorm(-10.499 / 127, 8) === f32(-10 / 127) && snorm(-10.501 / 127, 8) === f32(-11 / 127), () => "snorm8 quantization");
    e.check(snorm(-10.499 / 32767, 16) === f32(-10 / 32767) && snorm(-10.501 / 32767, 16) === f32(-11 / 32767), () => "snorm16 quantization");

    const reduction = new ParallelReduction(device);
    const cases: [ResourceFormat, number, number][] = [
        [ResourceFormat.RGBA32Float, 1, 1],
        [ResourceFormat.RGBA32Float, 32, 64],
        [ResourceFormat.RGBA32Float, 127, 71],
        [ResourceFormat.RGBA8Unorm, 256, 192],
        [ResourceFormat.RGBA8Snorm, 91, 130],
        [ResourceFormat.RG16Float, 220, 121],
        [ResourceFormat.RG16Unorm, 256, 192],
        [ResourceFormat.RG16Snorm, 333, 101],
        [ResourceFormat.RGBA32Uint, 33, 99],
        [ResourceFormat.R32Uint, 22, 291],
        [ResourceFormat.R16Int, 64, 33],
        [ResourceFormat.RG8Int, 403, 57],
    ];
    for (const [format, w, h] of cases) {
        if ((format === ResourceFormat.RG16Unorm || format === ResourceFormat.RG16Snorm) && !device.gpuDevice.features.has("texture-formats-tier1")) {
            console.log(`# ParallelReduction: ${ResourceFormat[format]} skipped (no texture-formats-tier1 on this adapter)`);
            continue;
        }
        await testReduction(device, e, reduction, format, w, h);
    }
    e.done("ParallelReduction");
});

