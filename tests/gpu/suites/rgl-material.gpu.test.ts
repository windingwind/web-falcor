/**
 * RGL measured materials (Scene/Material/RGLFile + RGLMaterial) over the real
 * CC0 measurements from rgl.epfl.ch/materials: the browser parses the
 * `tensor_file` container, builds the marginal/conditional CDFs that make the
 * VNDF and luminance tables samplable, places all eleven tables plus the albedo
 * LUT in the shared material buffer and renders through the upstream
 * RGLCommon/RGLMaterialInstance code.
 *
 * Checks are physical rather than golden-image (native Falcor cannot be used as
 * an oracle for these on this host): the BSDF must be non-zero and reciprocal,
 * its sampling must agree with its own pdf and evaluation, and the integrated
 * albedo must be a plausible energy-conserving reflectance.
 *
 * Fetch the measurements with: npm run download:assets -- rgl
 */

import { BSDFIntegrator, Buffer, ComputePass, MaterialType, MemoryType, ResourceBindFlags, Scene, float2, float3, float4, loadRGLFile } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose, SkipError } from "../harness/registry.js";

const kRglDir = "/Falcor/media/rgl/";

function makeScene(device: ConstructorParameters<typeof Scene>[0], rgl: Awaited<ReturnType<typeof loadRGLFile>>) {
    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    return new Scene(
        device,
        [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }],
        [{ name: rgl.name, basic: {}, rgl, header: { materialType: MaterialType.RGL } }],
    );
}

/** Evaluates the material's BSDF at explicit local direction pairs. */
async function evalBSDF(device: ConstructorParameters<typeof Scene>[0], scene: Scene, pairs: [number[], number[]][]) {
    const dirs = new Float32Array(pairs.length * 8);
    pairs.forEach(([wi, wo], i) => {
        dirs.set([wi[0]!, wi[1]!, wi[2]!, 0], i * 8);
        dirs.set([wo[0]!, wo[1]!, wo[2]!, 0], i * 8 + 4);
    });
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const dirBuf = new Buffer(device, { size: dirs.byteLength, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "rgl::dirs" });
    dirBuf.setBlob(new Uint8Array(dirs.buffer));
    const resBuf = new Buffer(device, { size: pairs.length * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "rgl::results" });

    const pass = ComputePass.create(device, { path: "WebFalcor/BSDFEvalProbe.cs.slang", defines: scene.getSceneDefines() });
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    const cb = root["CB"] as Record<string, unknown>;
    cb["gCount"] = pairs.length;
    cb["gMaterialID"] = 0;
    root["gDirections"] = dirBuf;
    root["gResults"] = resBuf;
    const ctx = device.renderContext;
    pass.execute(ctx, pairs.length, 1);
    return new Float32Array((await ctx.readBuffer(resBuf)).buffer);
}

/** Averages the material's own importance-sampled weights per incident direction. */
async function sampleBSDF(device: ConstructorParameters<typeof Scene>[0], scene: Scene, incident: number[][], sampleCount: number) {
    const dirs = new Float32Array(incident.length * 4);
    incident.forEach((wi, i) => dirs.set([wi[0]!, wi[1]!, wi[2]!, 0], i * 4));
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const dirBuf = new Buffer(device, { size: dirs.byteLength, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "rgl::incident" });
    dirBuf.setBlob(new Uint8Array(dirs.buffer));
    const sumBuf = new Buffer(device, { size: incident.length * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "rgl::sums" });

    const pass = ComputePass.create(device, { path: "WebFalcor/BSDFEvalProbe.cs.slang", csEntry: "mainSample", defines: scene.getSceneDefines() });
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    const cb = root["SampleCB"] as Record<string, unknown>;
    cb["gSampleCount"] = sampleCount;
    cb["gSampleMtlID"] = 0;
    cb["gDirCount"] = incident.length;
    root["gIncident"] = dirBuf;
    root["gSampleSums"] = sumBuf;
    const ctx = device.renderContext;
    pass.execute(ctx, incident.length, 1);
    return new Float32Array((await ctx.readBuffer(sumBuf)).buffer);
}

/** Directions on the hemisphere, deterministic and away from grazing angles. */
function hemisphereDirections(count: number): number[][] {
    const dirs: number[][] = [];
    for (let i = 0; i < count; i++) {
        const cosTheta = 0.15 + (0.8 * (i + 0.5)) / count;
        const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
        const phi = i * Math.PI * (3 - Math.sqrt(5));
        dirs.push([sinTheta * Math.cos(phi), sinTheta * Math.sin(phi), cosTheta]);
    }
    return dirs;
}

for (const [file, grid] of [
    ["acrylic_felt_green_rgb.bsdf", "isotropic"],
    ["aniso_morpho_melenaus_rgb.bsdf", "anisotropic"],
] as const) {
    gpuTest(`RGLMaterial.${grid}MeasurementEvaluates`, async ({ device }) => {
        if (!(await fetch(kRglDir + file, { method: "HEAD" })).ok) {
            throw new SkipError(`Falcor/media/rgl/${file} missing (npm run download:assets -- rgl)`);
        }
        const rgl = await loadRGLFile(kRglDir + file);
        console.error(`# RGL ${file}: "${rgl.description}" isotropic=${rgl.isotropic} theta=${rgl.thetaI.length} phi=${rgl.phiI.length} vndf=${rgl.vndfSize.join("x")} lumi=${rgl.lumiSize.join("x")}`);
        expectEq(rgl.isotropic, grid === "isotropic", "isotropy matches the measurement");

        // The CDFs the host builds must be normalized per slice: the marginal of
        // every 2D slice ends at exactly 1 (SamplableDistribution4D step 3).
        const sliceCount = rgl.phiI.length * rgl.thetaI.length;
        for (let s = 0; s < sliceCount; s++) {
            const last = rgl.vndfMarginal[(s + 1) * rgl.vndfSize[1] - 1]!;
            expectClose(last, 1, 1e-5, `vndf marginal CDF normalized (slice ${s})`);
        }

        const scene = makeScene(device, rgl);
        expectEq(scene.getSceneDefines().get("WEBFALCOR_MTL_RGL"), "1", "RGL material type define");

        // Evaluate over many direction pairs: a correctly wired measurement is
        // non-negative, finite, and non-zero for a good fraction of the pairs.
        const dirs = hemisphereDirections(16);
        const pairs: [number[], number[]][] = [];
        for (const wi of dirs) for (const wo of dirs) pairs.push([wi, wo]);
        const results = await evalBSDF(device, scene, pairs);

        let nonZero = 0;
        let bad = 0;
        let maxValue = 0;
        for (let i = 0; i < pairs.length; i++) {
            for (let c = 0; c < 3; c++) {
                const v = results[i * 4 + c]!;
                if (!Number.isFinite(v) || v < 0) bad++;
                maxValue = Math.max(maxValue, v);
            }
            if (results[i * 4]! + results[i * 4 + 1]! + results[i * 4 + 2]! > 1e-6) nonZero++;
        }
        console.error(`# RGL eval: ${nonZero}/${pairs.length} pairs non-zero, max ${maxValue.toFixed(4)}, ${bad} invalid channels`);
        expectEq(bad, 0, "all evaluations finite and non-negative");
        expectEq(nonZero > pairs.length * 0.5, true, `measurement responds over the hemisphere (${nonZero})`);

        // Helmholtz reciprocity: f(wi, wo) == f(wo, wi) (eval returns f * cos(theta_o),
        // so rescale first). The tables are interpolated from only a handful of
        // measured incident angles, so this holds in the median rather than exactly —
        // a transposed lookup or a swapped axis would still blow it apart.
        const swapped = await evalBSDF(
            device,
            scene,
            pairs.map(([wi, wo]) => [wo, wi] as [number[], number[]]),
        );
        const relative: number[] = [];
        for (let i = 0; i < pairs.length; i++) {
            const [wi, wo] = pairs[i]!;
            for (let c = 0; c < 3; c++) {
                const f = results[i * 4 + c]! / wo[2]!;
                const fSwapped = swapped[i * 4 + c]! / wi[2]!;
                const scale = Math.max(f, fSwapped);
                if (scale < 1e-3) continue; // reciprocity is meaningless in the noise floor
                relative.push(Math.abs(f - fSwapped) / scale);
            }
        }
        relative.sort((a, b) => a - b);
        const median = relative[relative.length >> 1]!;
        console.error(`# RGL reciprocity: ${relative.length} comparable channels, median relative difference ${median.toExponential(2)}`);
        expectEq(relative.length > 100, true, `enough comparable samples (${relative.length})`);
        expectEq(median < 0.25, true, `reciprocal in the median (${median})`);

        // Directional albedo: energy conserving and non-trivial.
        const integrator = new BSDFIntegrator(device, scene);
        const albedos = await integrator.integrateIsotropic(device.renderContext, 0, [0.3, 0.6, 0.9]);
        console.error(`# RGL albedo: ${albedos.map((a) => `(${a.x.toFixed(3)}, ${a.y.toFixed(3)}, ${a.z.toFixed(3)})`).join(" ")}`);
        for (const a of albedos) {
            for (const c of [a.x, a.y, a.z]) {
                expectEq(Number.isFinite(c) && c >= 0 && c <= 1.2, true, `albedo channel in range (${c})`);
            }
        }
        expectEq(albedos.some((a) => a.x + a.y + a.z > 0.01), true, "material reflects light");

        // Sampling consistency: the mean weight of the material's own importance
        // samples estimates the same directional albedo as integrating eval over a
        // uniform hemisphere. This is the check that exercises the VNDF/luminance
        // CDFs the host builds — a broken CDF samples the wrong distribution and
        // the two estimates part ways.
        const incident = [0.3, 0.6, 0.9].map((cosTheta) => [Math.sqrt(1 - cosTheta * cosTheta), 0, cosTheta]);
        const sampled = await sampleBSDF(device, scene, incident, 4096);
        for (let i = 0; i < incident.length; i++) {
            const reference = albedos[i]!;
            const estimate = [sampled[i * 4]!, sampled[i * 4 + 1]!, sampled[i * 4 + 2]!];
            const validFraction = sampled[i * 4 + 3]! / 4096;
            console.error(`# RGL sampling (cos=${incident[i]![2]!.toFixed(1)}): sampled (${estimate.map((v) => v.toFixed(4)).join(", ")}) vs integrated (${reference.x.toFixed(4)}, ${reference.y.toFixed(4)}, ${reference.z.toFixed(4)}), ${(validFraction * 100).toFixed(0)}% valid`);
            expectEq(validFraction > 0.5, true, `most samples are valid (${validFraction})`);
            const refChannels = [reference.x, reference.y, reference.z];
            for (let c = 0; c < 3; c++) {
                const scale = Math.max(refChannels[c]!, estimate[c]!, 0.02);
                expectEq(Math.abs(estimate[c]! - refChannels[c]!) / scale < 0.25, true, `sampled albedo agrees with the integral (channel ${c}: ${estimate[c]} vs ${refChannels[c]})`);
            }
        }

        // The LUT feeds BSDFProperties.diffuseReflectionAlbedo once computed.
        await scene.computeMeasuredAlbedoLUTs(device.renderContext);
        const withLut = await evalBSDF(device, scene, [[[0, 0, 1], [0, 0, 1]]]);
        const reported = withLut[3]!;
        console.error(`# RGL reported albedo (G, normal incidence): ${reported.toFixed(4)}`);
        expectEq(Number.isFinite(reported) && reported >= 0 && reported <= 1.2, true, `LUT albedo in range (${reported})`);
    });
}
