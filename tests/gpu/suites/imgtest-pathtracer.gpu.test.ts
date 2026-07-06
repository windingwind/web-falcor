/**
 * OVERALL VERIFY — upstream image test: the UNMODIFIED PathTracer.py graph
 * (VBufferRT -> PathTracer + guide outputs -> Accumulate -> ToneMap) over
 * cornell_box.pyscene, 4 accumulated frames, diffed against native captures.
 * (Upstream's test scene is Arcade.pyscene — FBX importer pending.)
 *
 * rayCount/pathLength are marked by the graph and allocate, but need the
 * PixelStats port (pending) — not compared here.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-pathtracer.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

function halfToFloat(h: number): number {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

gpuTest("ImageTestPathTracer.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/PathTracer.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 4; f++) graph!.execute(ctx);

    const readFloats = async (ref: string) => new Float32Array((await ctx.readTextureSubresource(graph!.getOutput(ref)!)).buffer);
    const readBytes = async (ref: string) => new Uint8Array((await ctx.readTextureSubresource(graph!.getOutput(ref)!)).buffer);
    const readHalfAsFloat = async (ref: string) => {
        const raw = new Uint16Array((await ctx.readTextureSubresource(graph!.getOutput(ref)!)).buffer);
        return Float32Array.from(raw, halfToFloat);
    };

    const compareExr = async (web: Float32Array, oracle: string, meanTol: number, badTol: number, badThreshold = 1e-3) => {
        const res = await fetch(`/tests/oracle/out-native/${oracle}`);
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        expectEq(width, size, `${oracle} resolution`);
        const webComponents = web.length / (size * size);
        let sum = 0;
        let bad = 0;
        let bad2 = 0;
        let bad1 = 0;
        let maxD = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * webComponents;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pixelMax = 0;
                for (let c = 0; c < 3; c++) {
                    const d = Math.abs(web[wi + c]! - data[ni + c]!);
                    sum += d;
                    pixelMax = Math.max(pixelMax, d);
                }
                if (pixelMax > badThreshold) bad++;
                if (pixelMax > 1e-2) bad2++;
                if (pixelMax > 1e-1) bad1++;
                maxD = Math.max(maxD, pixelMax);
            }
        }
        const mean = sum / (size * size * 3);
        console.error(`# ${oracle}: mean=${mean.toExponential(2)} bad=${bad} bad2=${bad2} bad1=${bad1} max=${maxD.toExponential(2)}`);
        expectEq(mean < meanTol, true, `${oracle} mean ${mean}`);
        expectEq(bad <= badTol, true, `${oracle} bad ${bad}`);
    };

    const comparePng = async (web: Uint8Array, oracle: string, mseTol: number) => {
        const blob = await (await fetch(`/tests/oracle/out-native/${oracle}`)).blob();
        const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
        const canvas = new OffscreenCanvas(size, size);
        const c2d = canvas.getContext("2d", { willReadFrequently: true })!;
        c2d.drawImage(bitmap, 0, 0);
        const nat = c2d.getImageData(0, 0, size, size).data;
        let mse = 0;
        for (let i = 0; i < size * size; i++) {
            for (let ch = 0; ch < 3; ch++) {
                const d = (web[i * 4 + ch]! - nat[i * 4 + ch]!) / 255;
                mse += d * d;
            }
        }
        mse /= size * size * 3;
        console.error(`# ${oracle}: mse=${mse.toExponential(2)}`);
        expectEq(mse < mseTol, true, `${oracle} MSE ${mse}`);
    };

    // Radiance: same policy as the verified cornell PathTracer oracle
    // (stochastic 1-spp content: bad-pixel gate at 0.05; observed ~60).
    await compareExr(await readFloats("PathTracer.color"), "oracle-imgtest-pathtracer.PathTracer.color.0.exr", 1e-3, 300, 0.05);
    await compareExr(await readHalfAsFloat("PathTracer.guideNormal"), "oracle-imgtest-pathtracer.PathTracer.guideNormal.0.exr", 1e-3, 300, 0.05);
    await compareExr(await readFloats("PathTracer.reflectionPosW"), "oracle-imgtest-pathtracer.PathTracer.reflectionPosW.0.exr", 5e-3, 300, 0.05);
    // 8-bit unorm guide outputs: direct byte compare (linear values).
    await comparePng(await readBytes("PathTracer.albedo"), "oracle-imgtest-pathtracer.PathTracer.albedo.0.png", 5e-4);
    await comparePng(await readBytes("PathTracer.specularAlbedo"), "oracle-imgtest-pathtracer.PathTracer.specularAlbedo.0.png", 5e-4);
    await comparePng(await readBytes("PathTracer.indirectAlbedo"), "oracle-imgtest-pathtracer.PathTracer.indirectAlbedo.0.png", 5e-4);
    // Final frame (sRGB bytes, direct compare).
    await comparePng(await readBytes("ToneMapper.dst"), "oracle-imgtest-pathtracer.ToneMapper.dst.0.png", 5e-4);
});
