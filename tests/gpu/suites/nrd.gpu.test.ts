/**
 * NRDPass ("NRD", NRD 3.1.0) in PathTracerNRD.py's configuration without the DLSS stage, over
 * Arcade with a static camera. Native NRDPass is D3D12-only (no oracle on Linux), so the checks
 * are behavioural:
 *  - ReLAX diffuse/specular: after 32 frames, ModulateIllumination over NRD's filtered radiance
 *    is far closer to a 256-frame accumulated reference than the raw 1-spp path tracer frame,
 *    and remodulating the unfiltered inputs gives back that raw frame (the demodulation chain is
 *    consistent with the path tracer's).
 *  - ReLAX diffuse on its own and the two motion-vector methods run; with a static camera the
 *    motion vectors are 0 wherever the path tracer found the delta path (reflection here; Arcade
 *    has no delta transmission).
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

const requireNRD = async () => {
    if (!(await fetch("/tools/nrd-3.1.0/shader-files.json", { method: "HEAD" })).ok) throw new SkipError("tools/nrd-3.1.0 missing (node scripts/setup-web.mjs)");
};

const [w, h] = [320, 180];

gpuTest("NRD.relaxDenoisesTowardsTheReference", async ({ device }) => {
    await requireNRD();
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const scene = await runSceneScript(device, await (await fetch("/Falcor/media/Arcade/Arcade.pyscene")).text(), "/Falcor/media/Arcade");
    scene.camera.setAspectRatio(w / h);
    const g = new RenderGraph(device, "PathTracerNRD");
    g.addPass(createPass(device, "GBufferRT", { samplePattern: "Center", useAlphaTest: true }), "GBufferRT");
    g.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, maxSurfaceBounces: 10, useRussianRoulette: true }), "PathTracer");
    g.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Reference");
    g.addPass(createPass(device, "NRD", { maxIntensity: 250 }), "NRDDiffuseSpecular");
    g.addPass(createPass(device, "NRD", { method: "SpecularReflectionMv", worldSpaceMotion: false }), "NRDReflectionMotionVectors");
    g.addPass(createPass(device, "NRD", { method: "SpecularDeltaMv", worldSpaceMotion: false }), "NRDTransmissionMotionVectors");
    g.addPass(createPass(device, "NRD", { method: "RelaxDiffuse", maxIntensity: 250, worldSpaceMotion: false, enableReprojectionTestSkippingWithoutMotion: true, spatialVarianceEstimationHistoryThreshold: 1 }), "NRDDeltaReflection");
    g.addPass(createPass(device, "ModulateIllumination", { useResidualRadiance: false }), "ModulateNRD");
    g.addPass(createPass(device, "ModulateIllumination", { useResidualRadiance: false }), "ModulateRaw");
    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer");
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW");
    g.addEdge("PathTracer.color", "Reference.input");
    g.addEdge("PathTracer.nrdDiffuseRadianceHitDist", "NRDDiffuseSpecular.diffuseRadianceHitDist");
    g.addEdge("PathTracer.nrdSpecularRadianceHitDist", "NRDDiffuseSpecular.specularRadianceHitDist");
    g.addEdge("GBufferRT.mvecW", "NRDDiffuseSpecular.mvec");
    g.addEdge("GBufferRT.normWRoughnessMaterialID", "NRDDiffuseSpecular.normWRoughnessMaterialID");
    g.addEdge("GBufferRT.linearZ", "NRDDiffuseSpecular.viewZ");
    g.addEdge("PathTracer.nrdDeltaReflectionHitDist", "NRDReflectionMotionVectors.specularHitDist");
    g.addEdge("GBufferRT.linearZ", "NRDReflectionMotionVectors.viewZ");
    g.addEdge("GBufferRT.normWRoughnessMaterialID", "NRDReflectionMotionVectors.normWRoughnessMaterialID");
    g.addEdge("GBufferRT.mvec", "NRDReflectionMotionVectors.mvec");
    g.addEdge("PathTracer.nrdDeltaReflectionRadianceHitDist", "NRDDeltaReflection.diffuseRadianceHitDist");
    g.addEdge("NRDReflectionMotionVectors.reflectionMvec", "NRDDeltaReflection.mvec");
    g.addEdge("PathTracer.nrdDeltaReflectionNormWRoughMaterialID", "NRDDeltaReflection.normWRoughnessMaterialID");
    g.addEdge("PathTracer.nrdDeltaReflectionPathLength", "NRDDeltaReflection.viewZ");
    g.addEdge("GBufferRT.posW", "NRDTransmissionMotionVectors.deltaPrimaryPosW");
    g.addEdge("PathTracer.nrdDeltaTransmissionPosW", "NRDTransmissionMotionVectors.deltaSecondaryPosW");
    g.addEdge("GBufferRT.mvec", "NRDTransmissionMotionVectors.mvec");
    for (const [m, diffuse, specular] of [
        ["ModulateNRD", "NRDDiffuseSpecular.filteredDiffuseRadianceHitDist", "NRDDiffuseSpecular.filteredSpecularRadianceHitDist"],
        ["ModulateRaw", "PathTracer.nrdDiffuseRadianceHitDist", "PathTracer.nrdSpecularRadianceHitDist"],
    ] as const) {
        g.addEdge("PathTracer.nrdEmission", `${m}.emission`);
        g.addEdge("PathTracer.nrdDiffuseReflectance", `${m}.diffuseReflectance`);
        g.addEdge(diffuse, `${m}.diffuseRadiance`);
        g.addEdge("PathTracer.nrdSpecularReflectance", `${m}.specularReflectance`);
        g.addEdge(specular, `${m}.specularRadiance`);
        g.addEdge("PathTracer.nrdResidualRadianceHitDist", `${m}.residualRadiance`);
    }
    for (const o of ["ModulateNRD.output", "ModulateRaw.output", "Reference.output", "PathTracer.color", "NRDDeltaReflection.filteredDiffuseRadianceHitDist", "NRDReflectionMotionVectors.reflectionMvec", "NRDTransmissionMotionVectors.deltaMvec", "PathTracer.nrdDeltaReflectionHitDist", "PathTracer.nrdDeltaTransmissionPosW"]) g.markOutput(o);
    g.onResize(w, h);
    g.setScene(scene);
    await g.init();

    const read = async (o: string) => {
        const bytes = await ctx.readTextureSubresource(g.getOutput(o)!);
        if (bytes.byteLength === w * h * 8) {
            const half = new Uint16Array(bytes.buffer, bytes.byteOffset, w * h * 4);
            return Float32Array.from(half, (v) => {
                const s = v & 0x8000 ? -1 : 1, e = (v >> 10) & 0x1f, m = v & 0x3ff;
                return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
            });
        }
        return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    };
    const t0 = performance.now();
    for (let f = 0; f < 32; f++) g.execute(ctx);
    const [nrd, raw, color] = [await read("ModulateNRD.output"), await read("ModulateRaw.output"), await read("PathTracer.color")];
    const frameMs = (performance.now() - t0) / 32;
    for (let f = 32; f < 256; f++) g.execute(ctx);
    const ref = await read("Reference.output");
    const channels = (a: Float32Array) => a.length / (w * h);
    const mse = (a: Float32Array, b: Float32Array) => {
        let s = 0, n = 0;
        for (let i = 0; i < w * h; i++)
            for (let c = 0; c < 3; c++) {
                // Tonemap-like compression so fireflies don't dominate.
                const [x, y] = [a[i * channels(a) + c]!, b[i * channels(b) + c]!];
                if (!Number.isFinite(x) || !Number.isFinite(y)) return NaN;
                const d = x / (1 + x) - y / (1 + y);
                s += d * d;
                n++;
            }
        return s / n;
    };
    const [eNRD, eRaw, eRemod] = [mse(nrd, ref), mse(color, ref), mse(raw, color)];
    // Fraction of pixels with ~0 motion among those with a delta hit (the others carry NRD's
    // 65504 invalid-path sentinel, which reprojects imprecisely).
    const hitDist = await read("PathTracer.nrdDeltaReflectionHitDist");
    // Delta motion needs a delta-transmission position (Arcade has no transmissive surfaces).
    const secondary = await read("PathTracer.nrdDeltaTransmissionPosW");
    const mv = [await read("NRDReflectionMotionVectors.reflectionMvec"), await read("NRDTransmissionMotionVectors.deltaMvec")].map((a, k) => {
        let still = 0, n = 0;
        for (let i = 0; i < w * h; i++) {
            if (k === 0 && !(hitDist[i * channels(hitDist)]! < 1e3)) continue;
            if (k === 1 && [0, 1, 2].every((c) => secondary[i * channels(secondary) + c] === 0)) continue;
            n++;
            if (Math.abs(a[i * channels(a)]!) < 1e-3 && Math.abs(a[i * channels(a) + 1]!) < 1e-3) still++;
        }
        return n > 0 ? still / n : 1;
    });
    {
        const a = await read("NRDReflectionMotionVectors.reflectionMvec");
        const c = channels(a);
        console.error(`#MV samples ${[[160, 90], [40, 150], [280, 30], [100, 100]].map(([x, y]) => `(${x},${y}) ${a[(y! * w + x!) * c]!.toPrecision(3)},${a[(y! * w + x!) * c + 1]!.toPrecision(3)}`).join(" ")}`);
    }
    const delta = await read("NRDDeltaReflection.filteredDiffuseRadianceHitDist");
    const deltaFinite = delta.every((v) => Number.isFinite(v));
    console.error(`# nrd: ${frameMs.toFixed(1)} ms/frame; MSE vs 256-frame reference: NRD ${eNRD.toExponential(2)}, raw 1spp ${eRaw.toExponential(2)} (x${(eRaw / eNRD).toFixed(1)}); remodulated inputs vs color ${eRemod.toExponential(2)}; still-pixel fraction of the motion vectors ${mv.map((v) => v.toFixed(3))}; delta reflection finite ${deltaFinite}`);
    expectEq(eNRD < eRaw / 4, true, `NRD output closer to the reference than the raw frame (${eNRD} vs ${eRaw})`);
    expectEq(eRemod < 1e-4, true, `remodulated NRD inputs reproduce the path tracer color (${eRemod})`);
    expectEq(mv.every((v) => v > 0.9), true, `static camera: motion vectors ~0 (${mv})`);
    expectEq(deltaFinite, true, "RelaxDiffuse output finite");
});

gpuTest("NRD.reblurDenoisesTowardsTheReference", async ({ device }) => {
    // ReBLUR diffuse/specular (the other togglable method) in the same setup: mip chains, the
    // hit-distance normalization in PackRadiance, temporal stabilization.
    await requireNRD();
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const scene = await runSceneScript(device, await (await fetch("/Falcor/media/Arcade/Arcade.pyscene")).text(), "/Falcor/media/Arcade");
    scene.camera.setAspectRatio(w / h);
    const g = new RenderGraph(device, "ReBLUR");
    g.addPass(createPass(device, "GBufferRT", { samplePattern: "Center", useAlphaTest: true }), "GBufferRT");
    g.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, maxSurfaceBounces: 10, useRussianRoulette: true }), "PathTracer");
    g.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Reference");
    g.addPass(createPass(device, "NRD", { method: "ReblurDiffuseSpecular", maxIntensity: 250 }), "NRD");
    g.addPass(createPass(device, "ModulateIllumination", { useResidualRadiance: false }), "Modulate");
    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer");
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW");
    g.addEdge("PathTracer.color", "Reference.input");
    g.addEdge("PathTracer.nrdDiffuseRadianceHitDist", "NRD.diffuseRadianceHitDist");
    g.addEdge("PathTracer.nrdSpecularRadianceHitDist", "NRD.specularRadianceHitDist");
    g.addEdge("GBufferRT.mvecW", "NRD.mvec");
    g.addEdge("GBufferRT.normWRoughnessMaterialID", "NRD.normWRoughnessMaterialID");
    g.addEdge("GBufferRT.linearZ", "NRD.viewZ");
    g.addEdge("PathTracer.nrdEmission", "Modulate.emission");
    g.addEdge("PathTracer.nrdDiffuseReflectance", "Modulate.diffuseReflectance");
    g.addEdge("NRD.filteredDiffuseRadianceHitDist", "Modulate.diffuseRadiance");
    g.addEdge("PathTracer.nrdSpecularReflectance", "Modulate.specularReflectance");
    g.addEdge("NRD.filteredSpecularRadianceHitDist", "Modulate.specularRadiance");
    g.addEdge("PathTracer.nrdResidualRadianceHitDist", "Modulate.residualRadiance");
    for (const o of ["Modulate.output", "Reference.output", "PathTracer.color"]) g.markOutput(o);
    g.onResize(w, h);
    g.setScene(scene);
    await g.init();
    const read = async (o: string) => {
        const bytes = await ctx.readTextureSubresource(g.getOutput(o)!);
        return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    };
    for (let f = 0; f < 32; f++) g.execute(ctx);
    const [nrd, color] = [await read("Modulate.output"), await read("PathTracer.color")];
    for (let f = 32; f < 256; f++) g.execute(ctx);
    const ref = await read("Reference.output");
    const mse = (a: Float32Array, b: Float32Array) => {
        let s = 0;
        for (let i = 0; i < w * h; i++)
            for (let c = 0; c < 3; c++) {
                const [x, y] = [a[i * 4 + c]!, b[i * 4 + c]!];
                if (!Number.isFinite(x) || !Number.isFinite(y)) return NaN;
                s += (x / (1 + x) - y / (1 + y)) ** 2;
            }
        return s / (w * h * 3);
    };
    const [eNRD, eRaw] = [mse(nrd, ref), mse(color, ref)];
    console.error(`# nrd reblur: MSE vs 256-frame reference: NRD ${eNRD.toExponential(2)}, raw 1spp ${eRaw.toExponential(2)} (x${(eRaw / eNRD).toFixed(1)})`);
    expectEq(eNRD < eRaw / 4, true, `ReBLUR output closer to the reference than the raw frame (${eNRD} vs ${eRaw})`);
});
