/**
 * MERLMix measured materials (Scene/Material/MERLMixMaterial): several MERL
 * BRDFs in one material, selected per texel by an index map.
 *
 * The fixtures are analytic, so the ground truth is closed-form: three constant
 * (Lambertian) BRDFs with distinct albedos plus an 8x8 selector whose texel
 * (x, y) holds the index x + 8y. Every texel therefore names exactly which
 * albedo the shader must return, `% 3` covers the wrap-around upstream applies
 * to out-of-range indices, and the albedo LUT must hold one row per BRDF.
 *
 * Web divergences under test (docs §9): the BRDF tables, the index map and the
 * stacked albedo LUT all live in the single shared material buffer, addressed
 * by byte offsets, and the index fetch reproduces upstream's point/wrap sampler
 * in shader code because the packed texture array has only a linear sampler.
 */

import { Buffer, ComputePass, MemoryType, MaterialType, ResourceBindFlags, Scene, decodeTGA, float2, float3, float4, initScripting, loadMERLBinary, runSceneScript, type MERLIndexMap } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose, SkipError } from "../harness/registry.js";

const kMerlDir = "/Falcor/media/merl/";
/** The fixtures' albedos, in the order the mix stores them. */
const kAlbedos = [
    [0.6, 0.5, 0.4],
    [0.2, 0.7, 0.3],
    [0.1, 0.2, 0.9],
];
const kFiles = ["merl-lambert.binary", "merl-lambert-green.binary", "merl-lambert-blue.binary"];

/** Runs the BSDF eval probe for (uv, wi, wo) triples in the local frame. */
async function evalBSDF(device: ConstructorParameters<typeof Scene>[0], scene: Scene, probes: { uv: [number, number]; wi: number[]; wo: number[] }[]) {
    const dirs = new Float32Array(probes.length * 8);
    probes.forEach((p, i) => {
        // The probe carries the shading point's uv in the two unused .w slots.
        dirs.set([p.wi[0]!, p.wi[1]!, p.wi[2]!, p.uv[0]], i * 8);
        dirs.set([p.wo[0]!, p.wo[1]!, p.wo[2]!, p.uv[1]], i * 8 + 4);
    });
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const dirBuf = new Buffer(device, { size: dirs.byteLength, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "merlmix::dirs" });
    dirBuf.setBlob(new Uint8Array(dirs.buffer));
    const resBuf = new Buffer(device, { size: probes.length * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "merlmix::results" });

    const pass = ComputePass.create(device, { path: "WebFalcor/BSDFEvalProbe.cs.slang", defines: scene.getSceneDefines() });
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    const cb = root["CB"] as Record<string, unknown>;
    cb["gCount"] = probes.length;
    cb["gMaterialID"] = 0;
    root["gDirections"] = dirBuf;
    root["gResults"] = resBuf;
    const ctx = device.renderContext;
    pass.execute(ctx, probes.length, 1);
    return new Float32Array((await ctx.readBuffer(resBuf)).buffer);
}

gpuTest("MERLMixMaterial.selectsTheBrdfPerTexelAndStacksAlbedoLUTs", async ({ device }) => {
    for (const file of [...kFiles, "merl-index-map.tga"]) {
        if (!(await fetch(`${kMerlDir}${file}`, { method: "HEAD" })).ok) {
            throw new SkipError(`Falcor/media/merl/${file} missing (node scripts/gen-assets.mjs)`);
        }
    }
    const brdfs = [];
    for (const file of kFiles) brdfs.push(await loadMERLBinary(`${kMerlDir}${file}`));

    // The selector comes from the file, so the loader is under test too.
    const tga = decodeTGA(await (await fetch(`${kMerlDir}merl-index-map.tga`)).arrayBuffer());
    const indices = new Uint8Array(tga.width * tga.height);
    for (let i = 0; i < indices.length; i++) indices[i] = tga.rgba[i * 4]!;
    const indexMap: MERLIndexMap = { width: tga.width, height: tga.height, indices };
    expectEq(indexMap.width, 8, "index map width");
    expectEq([...indices.slice(0, 4)].join(","), "0,1,2,3", "index map first row");

    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    const scene = new Scene(
        device,
        [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }],
        [{ name: "mix", basic: {}, merlMix: { brdfs, indexMap }, header: { materialType: MaterialType.MERLMix } }],
    );
    expectEq(scene.getSceneDefines().get("WEBFALCOR_MTL_MERLMIX"), "1", "MERLMix material type define");

    // One probe per texel, sampled at the texel centre, plus out-of-range uvs
    // that must wrap exactly as upstream's wrap sampler does.
    const theta = 0.4;
    const wi = [Math.sin(theta), 0, Math.cos(theta)];
    const wo = [Math.sin(0.7) * Math.cos(1.1), Math.sin(0.7) * Math.sin(1.1), Math.cos(0.7)];
    const probes: { uv: [number, number]; wi: number[]; wo: number[] }[] = [];
    const expectedIndex: number[] = [];
    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            probes.push({ uv: [(x + 0.5) / 8, (y + 0.5) / 8], wi, wo });
            expectedIndex.push((x + 8 * y) % brdfs.length);
        }
    }
    const wrapped: [number, number][] = [
        [1.5 + 0.5 / 8, 0.5 / 8], // u wraps forward: texel (4, 0)
        [-0.5 + 0.5 / 8, 0.5 / 8], // u wraps backward: texel (4, 0)
        [0.5 / 8, 2 + 3.5 / 8], // v wraps forward: texel (0, 3)
    ];
    const wrappedTexel = [[4, 0], [4, 0], [0, 3]];
    wrapped.forEach((uv) => probes.push({ uv, wi, wo }));
    wrappedTexel.forEach(([x, y]) => expectedIndex.push((x! + 8 * y!) % brdfs.length));

    const results = await evalBSDF(device, scene, probes);
    let worst = 0;
    let mismatches = 0;
    for (let i = 0; i < probes.length; i++) {
        const albedo = kAlbedos[expectedIndex[i]!]!;
        for (let c = 0; c < 3; c++) {
            // eval returns f * cos(theta_o), and each fixture's f is albedo / pi.
            const expected = (albedo[c]! / Math.PI) * wo[2]!;
            const diff = Math.abs(results[i * 4 + c]! - expected);
            worst = Math.max(worst, diff);
            if (diff > 1e-6) mismatches++;
        }
    }
    console.error(`# MERLMix per-texel selection: ${mismatches} channel mismatches over ${probes.length} probes, worst ${worst.toExponential(2)}`);
    expectEq(mismatches, 0, "every texel evaluates its own BRDF");

    // The LUT stacks one 256-entry table per BRDF; the probe reports
    // BSDFProperties.diffuseReflectionAlbedo.g, which must be that BRDF's albedo.
    await scene.computeMeasuredAlbedoLUTs(device.renderContext);
    const withLut = await evalBSDF(device, scene, probes);
    let worstAlbedo = 0;
    for (let i = 0; i < probes.length; i++) {
        worstAlbedo = Math.max(worstAlbedo, Math.abs(withLut[i * 4 + 3]! - kAlbedos[expectedIndex[i]!]![1]!));
    }
    console.error(`# MERLMix albedo LUT rows: worst |reported - albedo.g| = ${worstAlbedo.toExponential(2)}`);
    expectClose(worstAlbedo, 0, 5e-3, "each BRDF's LUT row reports its own albedo");
});

gpuTest("MERLMixMaterial.pysceneLoadsTheBrdfListAndIndexMap", async ({ device }) => {
    for (const file of [...kFiles, "merl-index-map.tga"]) {
        if (!(await fetch(`${kMerlDir}${file}`, { method: "HEAD" })).ok) {
            throw new SkipError(`Falcor/media/merl/${file} missing (node scripts/gen-assets.mjs)`);
        }
    }
    // The scene script drives the whole host path: the MERLMixMaterial binding
    // with its list of paths, the Index texture slot, and the TGA decode.
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/merlmix.pyscene")).text();
    const scene = await runSceneScript(device, source, "/Falcor/media");
    expectEq(scene.stats.materials, 1, "one material");
    expectEq(scene.getSceneDefines().get("WEBFALCOR_MTL_MERLMIX"), "1", "MERLMix material type define");

    const wi = [0, 0, 1];
    const wo = [Math.sin(0.5), 0, Math.cos(0.5)];
    const probes: { uv: [number, number]; wi: number[]; wo: number[] }[] = [];
    const expectedIndex: number[] = [];
    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            probes.push({ uv: [(x + 0.5) / 8, (y + 0.5) / 8], wi, wo });
            expectedIndex.push((x + 8 * y) % kFiles.length);
        }
    }
    const results = await evalBSDF(device, scene, probes);
    let worst = 0;
    for (let i = 0; i < probes.length; i++) {
        for (let c = 0; c < 3; c++) {
            const expected = (kAlbedos[expectedIndex[i]!]![c]! / Math.PI) * wo[2]!;
            worst = Math.max(worst, Math.abs(results[i * 4 + c]! - expected));
        }
    }
    console.error(`# MERLMix from pyscene: worst |eval - analytic| = ${worst.toExponential(2)} over ${probes.length} texels`);
    expectClose(worst, 0, 1e-6, "the scene-script path selects the same BRDFs");
});
