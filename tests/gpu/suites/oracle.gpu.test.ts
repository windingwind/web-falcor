/**
 * M5 EXIT TEST — native-oracle image comparison.
 * The same quad.gltf rendered by native Falcor (GBufferRT on hardware DXR,
 * pre-rendered EXRs in tests/oracle/out-native/) and by web-falcor
 * (GBufferRaster via WGSL); texC and posW compared per-pixel.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native.py --headless
 */

import { GltfImporter, RenderGraph, createPass, float3 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

async function fetchExr(url: string): Promise<{ data: Float32Array; width: number; height: number }> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`missing oracle image ${url} (${res.status})`);
    const buffer = await res.arrayBuffer();
    const { data, width, height } = parseExr(buffer, 1015 /* FloatType */);
    // Falcor EXR captures are bottom-up relative to our top-down readback: flip rows.
    const src = data as Float32Array;
    const flipped = new Float32Array(src.length);
    const rowFloats = width * 4;
    for (let y = 0; y < height; y++) flipped.set(src.subarray(y * rowFloats, (y + 1) * rowFloats), (height - 1 - y) * rowFloats);
    return { data: flipped, width, height };
}

function compare(
    web: Float32Array,
    native: Float32Array,
    width: number,
    height: number,
    webChannels: number,
    nativeChannels: number,
    compareChannels: number,
): { mean: number; max: number; badPixels: number } {
    let sum = 0;
    let max = 0;
    let badPixels = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            let pixelMax = 0;
            for (let c = 0; c < compareChannels; c++) {
                const d = Math.abs(web[i * webChannels + c]! - native[i * nativeChannels + c]!);
                sum += d;
                max = Math.max(max, d);
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 0.02) badPixels++;
        }
    }
    return { mean: sum / (width * height * compareChannels), max, badPixels };
}

gpuTest("Oracle.gbufferMatchesNativeFalcor", async ({ device }) => {
    const size = 256;
    const scene = await GltfImporter.importFromUrl(device, "/tests/oracle/assets/quad.gltf");
    // Mirror render-native.py exactly.
    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, "OracleGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "GBufferRaster"), "GBufferRaster");
    graph.markOutput("GBufferRaster.texC");
    graph.markOutput("GBufferRaster.posW");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);

    const webTexC = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRaster.texC")!)).buffer);
    const webPosW = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRaster.posW")!)).buffer);

    const nativeTexC = await fetchExr("/tests/oracle/out-native/oracle.GBufferRT.texC.0.exr");
    const nativePosW = await fetchExr("/tests/oracle/out-native/oracle.GBufferRT.posW.0.exr");
    expectEq(nativeTexC.width, size, "oracle resolution");

    // parse-exr returns RGBA float data.
    const texC = compare(webTexC, nativeTexC.data, size, size, 2, 4, 2);
    const posW = compare(webPosW, nativePosW.data, size, size, 4, 4, 3);

    // Tolerances: raster-vs-RT edge coverage differs on triangle boundaries;
    // interiors must match tightly. Allow a thin band of edge pixels.
    console.log(`# oracle texC: mean=${texC.mean.toExponential(2)} max=${texC.max.toFixed(3)} bad=${texC.badPixels}`);
    console.log(`# oracle posW: mean=${posW.mean.toExponential(2)} max=${posW.max.toFixed(3)} bad=${posW.badPixels}`);
    expectEq(texC.mean < 1e-3, true, `texC mean diff ${texC.mean}`);
    expectEq(posW.mean < 2e-3, true, `posW mean diff ${posW.mean}`);
    expectEq(texC.badPixels < size * 6, true, `texC bad pixels ${texC.badPixels} (edge band)`);
    expectEq(posW.badPixels < size * 6, true, `posW bad pixels ${posW.badPixels} (edge band)`);
});
