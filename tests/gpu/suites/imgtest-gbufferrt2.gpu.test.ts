/**
 * OVERALL VERIFY — upstream image test: the UNMODIFIED GBufferRT.py
 * graph (inline ray queries, the web-default path) over cornell_box.pyscene;
 * geometric + material G-buffer channels compared against native EXRs.
 *
 * Skipped channels: vbuffer/mtlData (packed uints round-tripped through float
 * EXR captures are not faithfully comparable; hit equality is covered by every
 * PT oracle), mvec/mvecW/disocclusion (native frame-0 prev-camera state
 * differs, as documented for VBufferRT), normWRoughnessMaterialID (native
 * RGB10A2Unorm; WGSL has no rgb10a2 storage — web uses RGBA16Float, format-
 * divergent by design), time/mask (not marked by this graph).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-gbufferrt.py --headless
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

gpuTest("ImageTestGBufferRT2.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/GBufferRT.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    // [channel, comparedComponents, meanTol, badTol, fp16?]
    const kCompare: [string, number, number, number, boolean?][] = [
        ["posW", 3, 1e-3, 200],
        ["normW", 3, 1e-3, 200],
        ["tangentW", 3, 1e-3, 250],
        ["faceNormalW", 3, 1e-3, 200],
        ["texC", 2, 1e-3, 200],
        ["texGrads", 3, 1e-3, 200, true], // 4th component lives in the .A.exr companion
        ["depth", 1, 1e-3, 200],
        ["linearZ", 2, 5e-3, 400],
        ["guideNormalW", 3, 1e-3, 200],
        ["diffuseOpacity", 3, 1e-3, 200],
        ["specRough", 3, 1e-3, 200],
        ["emissive", 3, 1e-3, 200],
        ["viewW", 3, 1e-3, 200],
    ];

    for (const [channel, components, tol, badTol, isFp16] of kCompare) {
        const tex = graph!.getOutput(`GBufferRT.${channel}`)!;
        const raw = await ctx.readTextureSubresource(tex);
        const web = isFp16 ? Float32Array.from(new Uint16Array(raw.buffer), halfToFloat) : new Float32Array(raw.buffer);
        const webComponents = web.length / (size * size);

        const res = await fetch(`/tests/oracle/out-native/oracle-imgtest-gbufferrt2.GBufferRT.${channel}.0.exr`);
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        expectEq(width, size, `${channel} oracle resolution`);

        let sum = 0;
        let badPixels = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * webComponents;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pixelMax = 0;
                for (let c = 0; c < components; c++) {
                    // Native captures are fp16 EXRs: huge values (linearZ miss
                    // = 1e8) saturate to +Inf there; clamp both to half range.
                    const clampHalf = (v: number) => Math.max(-65504, Math.min(v, 65504));
                    // linearZ slope on exactly view-parallel surfaces is 0/0
                    // upstream (normalize(0), rcp(0)): UB resolved differently
                    // by the native driver vs Tint — skip where degenerate.
                    if (channel === "linearZ" && c === 1 && (web[wi + c]! > 1e6 || data[ni + c]! > 1e6)) continue;
                    const d = Math.abs(clampHalf(web[wi + c]!) - clampHalf(data[ni + c]!));
                    sum += d;
                    pixelMax = Math.max(pixelMax, d);
                }
                if (pixelMax > 1e-3) badPixels++;
            }
        }
        const mean = sum / (size * size * components);
        console.error(`# imgtestGBufferRT2.${channel}: mean=${mean.toExponential(2)} bad=${badPixels}`);

        expectEq(mean < tol, true, `${channel} mean ${mean}`);
        expectEq(badPixels <= badTol, true, `${channel} bad pixels ${badPixels}`);
    }
});
