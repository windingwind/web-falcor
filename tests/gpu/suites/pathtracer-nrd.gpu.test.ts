/**
 * PathTracer's NRD outputs (the guide buffers NRDPass consumes, PathTracerNRD.py's settings) over
 * Arcade vs native (tests/oracle/render-native-pathtracer-nrd.py): the five radiance outputs
 * accumulated over 64 frames (8x8-block relative L1 as the other path-tracer oracles; the image
 * sum for near-empty ones), the
 * other thirteen per pixel. §9 formats: RGB10A2Unorm -> RGBA16Float and R16Float -> R32Float
 * (not storage formats in WebGPU), so native's quantization is applied before comparing.
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const [w, h] = [320, 180];
const frames = 64;
const radiance = ["nrdDiffuseRadianceHitDist", "nrdSpecularRadianceHitDist", "nrdDeltaReflectionRadianceHitDist", "nrdDeltaTransmissionRadianceHitDist", "nrdResidualRadianceHitDist"];
const guides = [
    "nrdEmission",
    "nrdDiffuseReflectance",
    "nrdSpecularReflectance",
    "nrdDeltaReflectionReflectance",
    "nrdDeltaReflectionEmission",
    "nrdDeltaReflectionNormWRoughMaterialID",
    "nrdDeltaReflectionPathLength",
    "nrdDeltaReflectionHitDist",
    "nrdDeltaTransmissionReflectance",
    "nrdDeltaTransmissionEmission",
    "nrdDeltaTransmissionNormWRoughMaterialID",
    "nrdDeltaTransmissionPathLength",
    "nrdDeltaTransmissionPosW",
];

gpuTest("PathTracerNRD.outputsMatchNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const scene = await runSceneScript(device, await (await fetch("/Falcor/media/Arcade/Arcade.pyscene")).text(), "/Falcor/media/Arcade");
    scene.camera.setAspectRatio(w / h);
    const graph = new RenderGraph(device, "PathTracerNRDOutputs");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center", useAlphaTest: true }), "GBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, maxSurfaceBounces: 10, useRussianRoulette: true }), "PathTracer");
    graph.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.addEdge("GBufferRT.viewW", "PathTracer.viewW");
    for (const name of radiance) {
        graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), `Acc_${name}`);
        graph.addEdge(`PathTracer.${name}`, `Acc_${name}.input`);
        graph.markOutput(`Acc_${name}.output`);
    }
    for (const name of guides) graph.markOutput(`PathTracer.${name}`);
    graph.markOutput("PathTracer.color");
    graph.onResize(w, h);
    graph.setScene(scene);
    await graph.init();
    for (let f = 0; f < frames; f++) graph.execute(ctx);

    const readWeb = async (output: string) => {
        const tex = graph.getOutput(output)!;
        const bytes = await ctx.readTextureSubresource(tex);
        // RGBA16Float outputs read back as halves.
        if (bytes.byteLength === w * h * 8) {
            const half = new Uint16Array(bytes.buffer, bytes.byteOffset, w * h * 4);
            const f = (v: number) => {
                const s = v & 0x8000 ? -1 : 1, e = (v >> 10) & 0x1f, m = v & 0x3ff;
                return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
            };
            return { data: Float32Array.from(half, f), channels: 4 };
        }
        const data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        return { data, channels: data.length / (w * h) };
    };
    const readNative = async (file: string) => {
        const buf = await (await fetch(`/tests/oracle/out-native/${file}`)).arrayBuffer();
        const exr = parseExr(buf, 1015) as { data: Float32Array; width: number; height: number };
        return { data: exr.data, channels: exr.data.length / (exr.width * exr.height), flipped: true };
    };
    const at = (img: { data: Float32Array; channels: number; flipped?: boolean }, x: number, y: number, c: number) => {
        const row = img.flipped ? h - 1 - y : y;
        return c < img.channels ? img.data[(row * w + x) * img.channels + c]! : 0;
    };

    const lines: string[] = [];
    let failures = 0;
    // Radiance: 8x8-block relative L1 over rgb; hit distance (.a) only where both are finite
    // (invalid paths carry NRD's infinite hit distance).
    for (const name of radiance) {
        const web = await readWeb(`Acc_${name}.output`);
        const nat = await readNative(`ptnrd.Acc_${name}.output.${frames}.RGBA.exr`);
        const block = 8;
        const absC = [0, 0, 0, 0], refC = [0, 0, 0, 0];
        for (let by = 0; by < h / block; by++)
            for (let bx = 0; bx < w / block; bx++)
                for (let c = 0; c < 4; c++) {
                    let a = 0, b = 0;
                    for (let y = by * block; y < (by + 1) * block; y++) for (let x = bx * block; x < (bx + 1) * block; x++) {
                        const [va, vb] = [at(web, x, y, c), at(nat, x, y, c)];
                        if (!Number.isFinite(va) || !Number.isFinite(vb)) continue;
                        a += va;
                        b += vb;
                    }
                    absC[c] += Math.abs(a - b);
                    refC[c] += Math.abs(b);
                }
        const relOf = (cs: number[]) => {
            const a = cs.reduce((s, c) => s + absC[c]!, 0), r = cs.reduce((s, c) => s + refC[c]!, 0);
            return r > 0 ? a / r : a;
        };
        const rel = relOf([0, 1, 2]);
        const sum = (img: typeof web) => [0, 1, 2].reduce((t, c) => {
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (Number.isFinite(at(img, x, y, c))) t += at(img, x, y, c);
            return t;
        }, 0);
        const [sumWeb, sumNat] = [sum(web), sum(nat)];
        const px = (x: number, y: number) => `web ${[0, 1, 2, 3].map((c) => at(web, x, y, c).toPrecision(3))} nat ${[0, 1, 2, 3].map((c) => at(nat, x, y, c).toPrecision(3))}`;
        lines.push(`${name} block relL1 rgb ${rel.toExponential(2)} hitDist ${relOf([3]).toExponential(2)}, sum ${sumWeb.toPrecision(4)} / ${sumNat.toPrecision(4)}`);
        // Near-empty outputs (Arcade's few mirror texels) are noise-dominated per block: gate on the sum.
        const sparse = sumNat < 0.01 * w * h;
        if (sparse ? !(Math.abs(sumWeb - sumNat) <= 0.1 * sumNat) : !(rel < 0.05)) failures++;
    }
    // Guides: fraction of pixels off by more than 2% of the native value (and 1e-3 absolute).
    for (const name of guides) {
        const web = await readWeb(`PathTracer.${name}`);
        // Native's RGB10A2Unorm normals are blitted to RGBA32Float before capture.
        const packed = name.endsWith("NormWRoughMaterialID");
        const nat = await readNative(packed ? `ptnrd.Blit_${name}.dst.${frames}.RGBA.exr` : `ptnrd.PathTracer.${name}.${frames}.RGBA.exr`);
        const channels = web.channels;
        let bad = 0, nonzero = 0;
        for (let y = 0; y < h; y++)
            for (let x = 0; x < w; x++) {
                let off = false;
                for (let c = 0; c < channels; c++) {
                    let a = at(web, x, y, c);
                    const b = at(nat, x, y, c);
                    // Native RGB10A2Unorm: clamp to [0,1] and quantize (2 bits of alpha).
                    const levels = c === 3 ? 3 : 1023;
                    if (packed) a = Math.round(Math.min(Math.max(a, 0), 1) * levels) / levels;
                    const tol = packed ? 1.5 / levels : Math.max(1e-3, 0.02 * Math.abs(b));
                    if (Math.abs(a - b) > tol) off = true;
                    if (b !== 0) nonzero++;
                }
                if (off) bad++;
            }
        const px = (x: number, y: number) => `web ${[0, 1, 2, 3].map((c) => at(web, x, y, c).toPrecision(3))} nat ${[0, 1, 2, 3].map((c) => at(nat, x, y, c).toPrecision(3))}`;
        lines.push(`${name} bad ${bad}/${w * h} (native nonzero ${nonzero})${packed ? `; (160,90) ${px(160, 90)}` : ""}`);
        if (bad > w * h * 0.02) failures++;
    }
    console.error(`# pathtracer-nrd:\n#   ${lines.join("\n#   ")}`);
    expectEq(failures, 0, "NRD outputs off vs native");
});
