/**
 * Transplanted FalcorTest GPU test Utils/TextureAnalyzerTests: the eight native test textures
 * (Falcor/data/tests), analyzed one by one and through the array interface, against native's
 * expected masks, constant values and min/max values.
 */

import { Bitmap, MemoryType, ResourceBindFlags, ResourceType, Texture, TextureAnalyzer, TextureAnalyzerRangeFlags, TextureAnalyzerResult, Buffer } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { Expect } from "../../harness/expect.js";

type V4 = [number, number, number, number];
const u = (r: number, g: number, b: number, a: number): V4 => [r / 255, g / 255, b / 255, a];
const kExpected: { mask: number; value: V4; minValue: V4; maxValue: V4 }[] = [
    { mask: 0x00011110, value: u(128, 255, 64, 1), minValue: u(128, 255, 64, 1), maxValue: u(128, 255, 64, 1) },
    { mask: 0x00011112, value: u(128, 255, 64, 1), minValue: u(128, 0, 64, 1), maxValue: u(128, 255, 64, 1) },
    { mask: 0x00011116, value: u(128, 255, 64, 1), minValue: u(128, 0, 64, 1), maxValue: u(128, 255, 255, 1) },
    { mask: 0x00011117, value: u(128, 255, 64, 1), minValue: u(128, 0, 64, 1), maxValue: u(192, 255, 255, 1) },
    { mask: 0x00011110, value: u(81, 98, 201, 1), minValue: u(81, 98, 201, 1), maxValue: u(81, 98, 201, 1) },
    { mask: 0x00011118, value: u(163, 169, 218, 1), minValue: u(163, 169, 218, 0), maxValue: u(163, 169, 218, 1) },
    { mask: 0x0001951e, value: [1 / 8, 2 / 8, 3 / 8, 4 / 8], minValue: [1 / 8, 2 / 8, 3 / 8, 1 / 16], maxValue: [1 / 8, Infinity, 3 / 8, 4 / 8] },
    { mask: 0x0003222d, value: [-19, -17, -15, -13], minValue: [0, 0, 0, 0], maxValue: [0, 0, 0, 1 / 256] },
];

gpuTest("FalcorTest.TextureAnalyzer", async ({ device }) => {
    const textures: Texture[] = [];
    for (let i = 0; i < kExpected.length; i++) {
        const bmp = (await Bitmap.createFromFile(`/Falcor/data/tests/texture${i + 1}.${i < 6 ? "png" : "exr"}`, true))!;
        const tex = new Texture(device, { type: ResourceType.Texture2D, width: bmp.width, height: bmp.height, format: bmp.format, bindFlags: ResourceBindFlags.ShaderResource });
        tex.setSubresourceBlob(0, 0, bmp.data);
        textures.push(tex);
    }
    const ctx = device.renderContext;
    const analyzer = new TextureAnalyzer(device);
    const result = new Buffer(device, { size: kExpected.length * 64, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal });
    const e = new Expect();
    const f = Math.fround;
    const verify = async (label: string) => {
        const bytes = new Uint8Array((await ctx.readBuffer(result)).buffer);
        kExpected.forEach((x, i) => {
            const r = TextureAnalyzerResult.fromBytes(bytes, i * 64);
            e.check(r.mask === x.mask, () => `${label} i = ${i}: mask 0x${r.mask.toString(16)} != 0x${x.mask.toString(16)}`);
            let range = 0;
            for (let c = 0; c < 4; c++) {
                const constant = (x.mask & (1 << c)) === 0;
                range |= x.mask >>> (4 + 4 * c);
                e.check(r.isConstant(1 << c) === constant, () => `${label} i = ${i} c = ${c}: isConstant`);
                e.check(r.minValue[c] === f(x.minValue[c]!), () => `${label} i = ${i} c = ${c}: min ${r.minValue[c]} != ${x.minValue[c]}`);
                e.check(r.maxValue[c] === f(x.maxValue[c]!), () => `${label} i = ${i} c = ${c}: max ${r.maxValue[c]} != ${x.maxValue[c]}`);
                if (constant) e.check(r.value[c] === f(x.value[c]!), () => `${label} i = ${i} c = ${c}: value ${r.value[c]} != ${x.value[c]}`);
            }
            for (const [flag, test] of [[TextureAnalyzerRangeFlags.Pos, r.isPos(15)], [TextureAnalyzerRangeFlags.Neg, r.isNeg(15)], [TextureAnalyzerRangeFlags.Inf, r.isInf(15)], [TextureAnalyzerRangeFlags.NaN, r.isNaN(15)]] as const)
                e.check(test === ((range & flag) !== 0), () => `${label} i = ${i}: range flag ${flag}`);
        });
    };
    ctx.clearBuffer(result);
    textures.forEach((t, i) => analyzer.analyze(ctx, t, 0, 0, result, i * 64));
    await verify("single");
    // The array interface, over a buffer holding garbage (native fills it with 0xbabababa).
    result.setBlob(new Uint8Array(kExpected.length * 64).fill(0xba));
    analyzer.analyzeAll(ctx, textures, result);
    await verify("array");
    e.done("TextureAnalyzer");
});
