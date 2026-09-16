/**
 * MERL measured materials (Scene/Material/MERLFile + MERLMaterial): the browser
 * loads a `.binary` from the MERL database layout, packs the samples into the
 * shared material buffer and evaluates them through the upstream
 * `MERLCommon::eval` bin lookup.
 *
 * The real database is licence-gated, so the fixtures are analytic BRDFs written
 * in exactly that format by `node tools/gen-assets.mjs` — which makes the ground
 * truth closed-form on both sides:
 *   • merl-lambert: constant f = albedo / pi, so every evaluation equals
 *     albedo/pi * cos(theta) and the integrated albedo LUT equals the albedo.
 *   • merl-index-probe: each bin stores its own (thetaH, thetaD, phiD) index, so
 *     a wrong half/difference-vector mapping or buffer layout is visible directly.
 */

import { BSDFIntegrator, Buffer, ComputePass, MemoryType, MaterialType, ResourceBindFlags, Scene, float2, float3, float4, loadMERLBinary } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose, SkipError } from "../harness/registry.js";

const kMerlDir = "/Falcor/media/merl/";

/** Single triangle: enough geometry for a valid scene; the probes shade directly. */
function makeScene(device: ConstructorParameters<typeof Scene>[0], merl: Awaited<ReturnType<typeof loadMERLBinary>>) {
    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    return new Scene(
        device,
        [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }],
        [{ name: merl.name, basic: {}, merl, header: { materialType: MaterialType.MERL } }],
    );
}

/**
 * Mirrors MERLCommon's half/difference-vector parameterisation and bin mapping,
 * returning the continuous bin coordinates. The integer part is the bin index;
 * the fractional part says how close the direction sits to a bin boundary,
 * where float (GPU) and double (here) arithmetic may truncate differently.
 */
function merlBinCoords(wi: [number, number, number], wo: [number, number, number]): [number, number, number] {
    const norm = (v: number[]) => {
        const l = Math.hypot(v[0]!, v[1]!, v[2]!);
        return [v[0]! / l, v[1]! / l, v[2]! / l];
    };
    const h = norm([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
    const thetaH = Math.acos(Math.min(1, Math.max(-1, h[2]!)));
    const phiH = Math.atan2(h[1]!, h[0]!);
    const rotate = (v: number[], axis: number[], angle: number) => {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const dot = v[0]! * axis[0]! + v[1]! * axis[1]! + v[2]! * axis[2]!;
        const w = [axis[1]! * v[2]! - axis[2]! * v[1]!, axis[2]! * v[0]! - axis[0]! * v[2]!, axis[0]! * v[1]! - axis[1]! * v[0]!];
        return [0, 1, 2].map((k) => v[k]! * c + axis[k]! * dot * (1 - c) + w[k]! * s);
    };
    const temp = rotate([...wi], [0, 0, 1], -phiH);
    const diff = rotate(temp, [0, 1, 0], -thetaH);
    const thetaD = Math.acos(Math.min(1, Math.max(-1, diff[2]!)));
    let phiD = Math.atan2(diff[1]!, diff[0]!);

    if (phiD < 0) phiD += Math.PI;
    return [thetaH <= 0 ? 0 : Math.sqrt((thetaH * 2) / Math.PI) * 90, ((thetaD * 2) / Math.PI) * 90, (phiD / Math.PI) * 180];
}

/** Runs the BSDF eval probe for a list of (wi, wo) pairs in the local frame. */
async function evalBSDF(device: ConstructorParameters<typeof Scene>[0], scene: Scene, pairs: [number[], number[]][]) {
    const dirs = new Float32Array(pairs.length * 8);
    pairs.forEach(([wi, wo], i) => {
        dirs.set([wi[0]!, wi[1]!, wi[2]!, 0], i * 8);
        dirs.set([wo[0]!, wo[1]!, wo[2]!, 0], i * 8 + 4);
    });
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const dirBuf = new Buffer(device, { size: dirs.byteLength, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "merl::dirs" });
    dirBuf.setBlob(new Uint8Array(dirs.buffer));
    const resBuf = new Buffer(device, { size: pairs.length * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "merl::results" });

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

gpuTest("MERLMaterial.lambertEvaluatesAndIntegratesToItsAlbedo", async ({ device }) => {
    if (!(await fetch(`${kMerlDir}merl-lambert.binary`, { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/merl/merl-lambert.binary missing (node tools/gen-assets.mjs)");
    }
    const merl = await loadMERLBinary(`${kMerlDir}merl-lambert.binary`);
    expectEq(merl.data.length, 1458000 * 3, "sample count");
    const scene = makeScene(device, merl);
    expectEq(scene.getSceneDefines().get("WEBFALCOR_MTL_MERL"), "1", "MERL material type define");

    // eval() returns f * cos(theta_o); the fixture's f is albedo / pi everywhere.
    const albedo = [0.6, 0.5, 0.4];
    const pairs: [number[], number[]][] = [];
    for (let i = 1; i <= 8; i++) {
        const ti = (i / 9) * (Math.PI / 2);
        for (let j = 1; j <= 8; j++) {
            const to = (j / 9) * (Math.PI / 2);
            const phi = (j / 8) * Math.PI * 2;
            pairs.push([
                [Math.sin(ti), 0, Math.cos(ti)],
                [Math.sin(to) * Math.cos(phi), Math.sin(to) * Math.sin(phi), Math.cos(to)],
            ]);
        }
    }
    const results = await evalBSDF(device, scene, pairs);
    let worst = 0;
    for (let i = 0; i < pairs.length; i++) {
        const cosThetaO = pairs[i]![1][2]!;
        for (let c = 0; c < 3; c++) {
            const expected = (albedo[c]! / Math.PI) * cosThetaO;
            worst = Math.max(worst, Math.abs(results[i * 4 + c]! - expected));
        }
    }
    console.error(`# MERL lambert eval: ${pairs.length} pairs, worst |f*cos - albedo/pi*cos| = ${worst.toExponential(2)}`);
    expectClose(worst, 0, 1e-6, "eval matches the analytic Lambertian");

    // The albedo LUT is the hemispherical integral of that BRDF — the albedo itself.
    const integrator = new BSDFIntegrator(device, scene);
    const integrated = await integrator.integrateIsotropic(device.renderContext, 0, [0.25, 0.5, 0.75, 1.0]);
    for (const value of integrated) {
        expectClose(value.x, albedo[0]!, 2e-3, "integrated albedo R");
        expectClose(value.y, albedo[1]!, 2e-3, "integrated albedo G");
        expectClose(value.z, albedo[2]!, 2e-3, "integrated albedo B");
    }
    console.error(`# MERL lambert integrated albedo: ${integrated.map((v) => v.x.toFixed(4)).join(", ")} (expect 0.6)`);

    // Once computed, the LUT drives BSDFProperties.diffuseReflectionAlbedo.
    await scene.computeMERLAlbedoLUTs(device.renderContext);
    const withLut = await evalBSDF(device, scene, [[[0, 0, 1], [0, 0, 1]]]);
    expectClose(withLut[3]!, albedo[1]!, 5e-3, "albedo LUT reported through BSDFProperties");
});

gpuTest("MERLMaterial.indexProbeMatchesTheBinMapping", async ({ device }) => {
    if (!(await fetch(`${kMerlDir}merl-index-probe.binary`, { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/merl/merl-index-probe.binary missing (node tools/gen-assets.mjs)");
    }
    const merl = await loadMERLBinary(`${kMerlDir}merl-index-probe.binary`);
    const scene = makeScene(device, merl);

    // Spread directions over the hemisphere so many different bins are hit, and
    // keep the ones that sit well inside their bins: on a boundary the GPU's
    // float arithmetic and this double-precision mirror may truncate to
    // neighbouring bins, which says nothing about the mapping being right.
    const pairs: [number[], number[]][] = [];
    const indices: [number, number, number][] = [];
    for (let i = 0; i < 512 && pairs.length < 64; i++) {
        const ti = (((i % 8) + 1) / 10) * (Math.PI / 2);
        const phiI = (i / 512) * Math.PI * 2;
        const to = ((((i / 8) | 0) % 8) + 1) / 10 * (Math.PI / 2);
        const phiO = (i / 97) * Math.PI * 2 + 0.3;
        const wi: [number, number, number] = [Math.sin(ti) * Math.cos(phiI), Math.sin(ti) * Math.sin(phiI), Math.cos(ti)];
        const wo: [number, number, number] = [Math.sin(to) * Math.cos(phiO), Math.sin(to) * Math.sin(phiO), Math.cos(to)];
        const coords = merlBinCoords(wi, wo);
        if (coords.some((c) => c - Math.floor(c) < 0.02 || c - Math.floor(c) > 0.98)) continue;
        pairs.push([wi, wo]);
        indices.push([Math.min(Math.floor(coords[0]), 89), Math.min(Math.max(Math.floor(coords[1]), 0), 89), Math.min(Math.max(Math.floor(coords[2]), 0), 179)]);
    }
    expectEq(pairs.length, 64, "probe pairs away from bin boundaries");

    const results = await evalBSDF(device, scene, pairs);

    let mismatches = 0;
    let worst = 0;
    for (let i = 0; i < pairs.length; i++) {
        const wo = pairs[i]![1];
        const [h, d, p] = indices[i]!;
        // The fixture stores the normalized indices; eval multiplies by cos(theta_o).
        const expected = [h / 89, d / 89, p / 179].map((v) => v * wo[2]!);
        for (let c = 0; c < 3; c++) {
            const diff = Math.abs(results[i * 4 + c]! - expected[c]!);
            worst = Math.max(worst, diff);
            if (diff > 1e-5) mismatches++;
        }
    }
    console.error(`# MERL index probe: ${mismatches} channel mismatches over ${pairs.length} pairs, worst ${worst.toExponential(2)}`);
    expectEq(mismatches, 0, "every bin lookup lands on the CPU-computed index");
});
