/**
 * M1 Core/API GPU unit tests (mirrors FalcorTest's Tests/Core coverage).
 */

import {
    MemoryType,
    ResourceBindFlags,
    ResourceFormat,
    TextureFilteringMode,
    UnsupportedFeatureError,
    TextureReductionMode,
} from "@web-falcor/falcor";
import { gpuTest, expectEq, expectArrayEq, expectArrayClose } from "../harness/registry.js";

gpuTest("Buffer.uploadReadbackRoundtrip", async ({ device }) => {
    const data = new Uint32Array(256).map((_, i) => i * 3 + 1);
    const buffer = device.createBuffer(data.byteLength, undefined, MemoryType.DeviceLocal, data);
    const readback = await buffer.getBlob();
    expectArrayEq(new Uint32Array(readback.buffer, readback.byteOffset, 256), data, "buffer contents");
    buffer.destroy();
});

gpuTest("Buffer.setBlobOffset", async ({ device }) => {
    const buffer = device.createBuffer(64);
    device.renderContext.clearBuffer(buffer);
    const patch = new Uint32Array([0xdeadbeef, 0xcafebabe]);
    buffer.setBlob(patch, 16);
    const readback = new Uint32Array((await buffer.getBlob()).buffer);
    expectEq(readback[3], 0, "untouched word");
    expectEq(readback[4], 0xdeadbeef, "patched word 0");
    expectEq(readback[5], 0xcafebabe, "patched word 1");
    buffer.destroy();
});

gpuTest("Buffer.structuredWithCounter", async ({ device }) => {
    const buffer = device.createStructuredBuffer(16, 8, undefined, undefined, true);
    expectEq(buffer.elementCount, 8, "elementCount");
    expectEq(buffer.structSize, 16, "structSize");
    expectEq(buffer.counterBuffer !== undefined, true, "counter exists");
    buffer.destroy();
});

gpuTest("Buffer.copyRegion", async ({ device }) => {
    const src = device.createBuffer(64, undefined, MemoryType.DeviceLocal, new Uint32Array(16).fill(7));
    const dst = device.createBuffer(64);
    const ctx = device.renderContext;
    ctx.clearBuffer(dst);
    ctx.copyBufferRegion(dst, 32, src, 0, 32);
    const words = new Uint32Array((await dst.getBlob()).buffer);
    expectEq(words[0], 0, "before region");
    expectEq(words[8], 7, "inside region");
    src.destroy();
    dst.destroy();
});

gpuTest("Texture.uploadReadbackRoundtrip", async ({ device }) => {
    const w = 16, h = 16;
    const pixels = new Uint8Array(w * h * 4).map((_, i) => i % 251);
    const tex = device.createTexture2D(w, h, ResourceFormat.RGBA8Unorm, 1, 1, pixels);
    const readback = await device.renderContext.readTextureSubresource(tex);
    expectArrayEq(readback, pixels, "texture contents");
    tex.destroy();
});

gpuTest("Texture.mipChainAutoCount", ({ device }) => {
    const tex = device.createTexture2D(256, 64, ResourceFormat.RGBA8Unorm);
    expectEq(tex.mipCount, 9, "full mip chain of 256x64");
    tex.destroy();
});

gpuTest("Texture.clearAndReadback", async ({ device }) => {
    const tex = device.createTexture2D(8, 8, ResourceFormat.RGBA8Unorm, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
    device.renderContext.clearTexture(tex, [1, 0.5, 0, 1]);
    const px = await device.renderContext.readTextureSubresource(tex);
    expectEq(px[0], 255, "r");
    // 0.5 * 255 = 127.5: implementations may round either way.
    expectEq(px[1] === 127 || px[1] === 128, true, `g (0.5 unorm, got ${px[1]})`);
    expectEq(px[2], 0, "b");
    expectEq(px[3], 255, "a");
    tex.destroy();
});

gpuTest("RenderContext.blitFormatConversion", async ({ device }) => {
    const src = device.createTexture2D(4, 4, ResourceFormat.RGBA8Unorm, 1, 1, new Uint8Array(4 * 4 * 4).fill(200));
    const dst = device.createTexture2D(4, 4, ResourceFormat.RGBA16Float, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
    device.renderContext.blit(src, dst);
    const raw = await device.renderContext.readTextureSubresource(dst);
    // f16 halfs: 200/255 ≈ 0.7843
    const half = new Uint16Array(raw.buffer, 0, 4);
    const toF32 = (h: number) => {
        const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x3ff;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    };
    expectArrayClose([toF32(half[0]!), toF32(half[1]!), toF32(half[2]!)], [0.7843, 0.7843, 0.7843], 1e-2, "blit color");
    src.destroy();
    dst.destroy();
});

gpuTest("ComputeContext.rawDispatch", async ({ device }) => {
    const wgsl = `
        @group(0) @binding(0) var<storage, read_write> data: array<u32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
            if (gid.x < arrayLength(&data)) { data[gid.x] = data[gid.x] * 2u + 1u; }
        }`;
    const module = device.gpuDevice.createShaderModule({ code: wgsl });
    const pipeline = device.gpuDevice.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const input = new Uint32Array(128).map((_, i) => i);
    const buffer = device.createBuffer(input.byteLength, undefined, MemoryType.DeviceLocal, input);
    const bindGroup = device.gpuDevice.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: buffer.gpuBuffer } }],
    });
    device.renderContext.dispatchRaw(pipeline, [bindGroup], 2);
    const result = new Uint32Array((await buffer.getBlob()).buffer);
    expectArrayEq(result, input.map((v) => v * 2 + 1), "compute output");
    buffer.destroy();
});

gpuTest("Fence.signalWait", async ({ device }) => {
    const fence = device.createFence();
    const v1 = fence.signal();
    expectEq(v1, 1n, "first auto signal value");
    const v2 = fence.signal();
    await fence.wait(v2);
    expectEq(fence.getSignaledValue() >= v2, true, "signaled value reached");
});

gpuTest("Sampler.createAndUnsupportedModes", ({ device }) => {
    const sampler = device.createSampler({ magFilter: TextureFilteringMode.Linear, maxAnisotropy: 8 });
    expectEq(sampler.gpuSampler !== undefined, true, "sampler created");
    let threw = false;
    try {
        device.createSampler({ reductionMode: TextureReductionMode.Min });
    } catch (err) {
        threw = err instanceof UnsupportedFeatureError;
    }
    expectEq(threw, true, "Min reduction throws UnsupportedFeatureError");
});

gpuTest("GpuTimer.measureDispatch", async ({ device }) => {
    const timer = device.createGpuTimer();
    const ctx = device.renderContext;
    const buffer = device.createBuffer(1024 * 1024);
    timer.begin(ctx);
    ctx.clearBuffer(buffer);
    timer.end(ctx);
    const ms = await timer.resolve(ctx);
    // Either the feature is missing (0) or we get a sane small duration.
    expectEq(ms >= 0 && ms < 1000, true, `elapsed in sane range (got ${ms})`);
    buffer.destroy();
    timer.destroy();
});
