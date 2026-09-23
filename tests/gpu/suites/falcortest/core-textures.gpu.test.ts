/**
 * Transplanted FalcorTest GPU tests: Core/TextureTests and Core/BlitTests.
 * Texture_Load8Bit is not portable: it views a BGRX8 texture as uint, and
 * WebGPU has neither BGRX8 nor unorm-as-uint views.
 */

import { Mt19937, ResourceBindFlags, ResourceFormat, createMippedTextureFromFiles, kMaxPossible } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect, uniformFloat } from "../../harness/expect.js";

gpuTest("FalcorTest.RWTexture3D", async ({ device }) => {
    const tex = device.createTexture3D(16, 16, 16, ResourceFormat.R32Uint, 1, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/TextureTests.cs.slang", "testTexture3DWrite");
    ctx.vars()["tex3D_uav"] = tex;
    ctx.runProgram(16, 16, 16);
    ctx.createProgram("Tests/Core/TextureTests.cs.slang", "testTexture3DRead");
    ctx.allocateStructuredBuffer("result", 4096);
    ctx.vars()["tex3D_srv"] = tex;
    ctx.runProgram(16, 16, 16);
    const result = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    let i = 0;
    for (let z = 0; z < 16; z++) for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++, i++) e.check(result[i] === x * y * z + 577, () => `i = ${i}: ${result[i]}`);
    e.done("RWTexture3D");
});

gpuTest("FalcorTest.TextureMinMaxMip", async ({ device }) => {
    const w = 16;
    const h = 16;
    const colorInc = 128 / w;
    const base = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) base.fill(x * colorInc + y * colorInc, (x + y * w) * 4, (x + y * w) * 4 + 4);
    const tex = device.createTexture2D(w, h, ResourceFormat.RGBA8Unorm, 1, kMaxPossible, base, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget);
    tex.generateMips(device.renderContext, true);
    const e = new Expect();
    const fr = Math.fround;
    let prev = base;
    for (let m = 1; m < tex.mipCount; m++) {
        const level = await device.renderContext.readTextureSubresource(tex, m);
        const lw = w >> m;
        const lh = h >> m;
        const calc = new Uint8Array(lw * lh * 4);
        for (let y = 0; y < lh; y++)
            for (let x = 0; x < lw; x++) {
                let avg = 0, mn = Infinity, mx = -Infinity, alpha = 0;
                for (let dy = 0; dy < 2; dy++)
                    for (let dx = 0; dx < 2; dx++) {
                        const o = (x * 2 + dx + (y * 2 + dy) * lw * 2) * 4;
                        avg = fr(avg + fr(fr(prev[o]! / 255) / 4));
                        mn = Math.min(mn, fr(prev[o + 1]! / 255));
                        mx = Math.max(mx, fr(prev[o + 2]! / 255));
                        alpha = fr(alpha + fr(fr(prev[o + 3]! / 255) / 4));
                    }
                const want = [avg, mn, mx, alpha].map((v) => Math.trunc(fr(v * 255)));
                const o = (x + y * lw) * 4;
                for (let c = 0; c < 4; c++) e.check(level[o + c] === want[c], () => `mip ${m} [${x}, ${y}] channel ${c}: ${level[o + c]} != ${want[c]}`);
                calc.set(want, o);
            }
        prev = calc;
    }
    e.done("TextureMinMaxMip");
});

gpuTest("FalcorTest.Texture2D_LoadMips", async ({ device }) => {
    const tex = await createMippedTextureFromFiles(device, [0, 1, 2].map((i) => `/Falcor/data/tests/tiny_mip${i}.png`), false);
    const e = new Expect();
    e.check(tex !== null && tex.mipCount === 3, () => `mip count ${tex?.mipCount}`);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgram("Tests/Core/TextureLoadTests.cs.slang", "testLoadMips");
    ctx.allocateStructuredBuffer("result", 3);
    ctx.vars()["texUnorm"] = tex!;
    ctx.runProgram(1, 1, 1);
    const r = await ctx.readBuffer("result", Uint32Array);
    const want = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255];
    e.check(Array.from(r.subarray(0, 12)).join() === want.join(), () => `result ${Array.from(r.subarray(0, 12))}`);
    e.done("Texture2D_LoadMips");
});

async function testBlit(ctx: GPUUnitTestContext, isFloat: boolean, srcW: number, srcH: number, scale: 1 | 2): Promise<void> {
    const device = ctx.device;
    const dstW = srcW / scale;
    const dstH = srcH / scale;
    const rng = new Mt19937();
    const u = uniformFloat(rng);
    const n = srcW * srcH * 4;
    const src = isFloat ? Float32Array.from({ length: n }, u) : Uint32Array.from({ length: n }, () => rng.next());
    const expected: number[] = [];
    const load = (x: number, y: number, i: number) => src[(y * srcW + x) * 4 + i]!;
    for (let y = 0; y < dstH; y++)
        for (let x = 0; x < dstW; x++)
            for (let i = 0; i < 4; i++) {
                if (scale === 1) expected.push(src[expected.length]!);
                else {
                    const sum = load(2 * x, 2 * y, i) + load(2 * x + 1, 2 * y, i) + load(2 * x, 2 * y + 1, i) + load(2 * x + 1, 2 * y + 1, i);
                    expected.push(isFloat ? Math.fround(sum / 4) : Math.floor(sum / 4));
                }
            }
    const format = isFloat ? ResourceFormat.RGBA32Float : ResourceFormat.RGBA32Uint;
    const pSrc = device.createTexture2D(srcW, srcH, format, 1, 1, src, ResourceBindFlags.ShaderResource);
    const pDst = device.createTexture2D(dstW, dstH, format, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
    device.renderContext.blit(pSrc, pDst);
    ctx.createProgram("Tests/Core/BlitTests.cs.slang", "readback", { FLOAT_FORMAT: isFloat ? 1 : 0 });
    ctx.allocateStructuredBuffer("result", dstW * dstH * 4);
    ctx.vars()["tex"] = pDst;
    ctx.vars()["CB"]["sz"] = [dstW, dstH];
    ctx.runProgram(dstW, dstH, 1);
    const result = await ctx.readBuffer("result", isFloat ? Float32Array : Uint32Array);
    const e = new Expect();
    for (let i = 0; i < expected.length; i++) {
        const ok = isFloat ? Math.abs(result[i]! - expected[i]!) <= 1e-6 : result[i] === expected[i];
        e.check(ok, () => `i = ${i}: ${result[i]} vs ${expected[i]}`);
    }
    e.done(`blit ${isFloat ? "float" : "uint"} x${scale}`);
}

gpuTest("FalcorTest.BlitFloatNoFilter", async ({ device }) => testBlit(new GPUUnitTestContext(device), true, 33, 63, 1));
gpuTest("FalcorTest.BlitFloatFilter", async ({ device }) => testBlit(new GPUUnitTestContext(device), true, 32, 64, 2));
gpuTest("FalcorTest.BlitUintNoFilter", async ({ device }) => testBlit(new GPUUnitTestContext(device), false, 33, 63, 1));
gpuTest("FalcorTest.BlitUintFilter", async ({ device }) => testBlit(new GPUUnitTestContext(device), false, 32, 64, 2));
