/**
 * IES light profiles (Scene/Lights/LightProfile): an IESNA LM-63 file is parsed
 * and baked into the 256x256 table the shader samples, with the flux factor
 * reduced on the GPU — the same two steps native performs.
 *
 * The profiles are written with analytic candela distributions by
 * `node tools/gen-assets.mjs` (manufacturer photometry is free to download but
 * carries no reusable licence), so the whole bake has a closed form: this test
 * recomputes the kernel's output on the CPU and compares texel by texel.
 */

import { LightProfile, ParallelReduction, ParallelReductionType, Scene, initScripting, parseIesProfile, runSceneScript, float2, float3, float4 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose, SkipError } from "../harness/registry.js";

const kIesDir = "/Falcor/media/ies/";
const kResolution = 256;

/** Mirrors BakeIesProfile.cs.slang's findAngleIndex (a continuous index into the angle table). */
function findAngleIndex(angle: number, data: Float32Array, offset: number, count: number): number {
    if (count === 1) return 0;
    let right = data[offset]!;
    if (angle <= right) return 0;
    for (let i = 1; i < count; i++) {
        const left = right;
        right = data[offset + i]!;
        if (angle >= left && angle <= right) return i - 1 + (right > left ? (angle - left) / (right - left) : 0);
    }
    return count - 1;
}

/** Mirrors the bake kernel for one texel: returns [candela * normalization, fluxContribution]. */
function bakeTexel(data: Float32Array, x: number, y: number): [number, number] {
    const verticalCount = Math.trunc(data[3]!);
    const horizontalCount = Math.trunc(data[4]!);
    const headerSize = 13;
    const verticalAngle = x * (180 / kResolution);
    let horizontalAngle = y * (360 / kResolution) - 180;
    const lastVertical = data[headerSize + verticalCount - 1]!;
    const lastHorizontal = data[headerSize + verticalCount + horizontalCount - 1]!;
    if (verticalAngle > lastVertical) return [0, 0];

    if (lastHorizontal <= 180) {
        horizontalAngle = Math.abs(horizontalAngle);
        if (lastHorizontal === 90 && horizontalAngle > 90) horizontalAngle = 180 - horizontalAngle;
    } else if (horizontalAngle < 0) {
        horizontalAngle += 360;
    }

    const vi = findAngleIndex(verticalAngle, data, headerSize, verticalCount);
    const hi = findAngleIndex(horizontalAngle, data, headerSize + verticalCount, horizontalCount);
    const dataOffset = headerSize + horizontalCount + verticalCount;
    const at = (h: number, v: number) => data[dataOffset + h * verticalCount + v]!;
    const a = at(Math.floor(hi), Math.floor(vi));
    const b = at(Math.floor(hi), Math.ceil(vi));
    const c = at(Math.ceil(hi), Math.floor(vi));
    const d = at(Math.ceil(hi), Math.ceil(vi));
    const fv = vi - Math.floor(vi);
    const fh = hi - Math.floor(hi);
    const candelas = (a + (b - a) * fv) * (1 - fh) + (c + (d - c) * fv) * fh;
    const result = candelas * data[0]!;

    const theta = (verticalAngle / 180) * Math.PI;
    return [result, (result * Math.sin(theta) * 2 * Math.PI * Math.PI) / (kResolution * kResolution)];
}

for (const [file, description] of [
    ["ies-cosine.ies", "rotationally symmetric cosine lobe"],
    ["ies-spot.ies", "narrow spot (clipped beyond 45 degrees)"],
    ["ies-asymmetric.ies", "azimuth-dependent profile"],
] as const) {
    gpuTest(`LightProfile.bakes_${file.replace(/[^a-z]/g, "_")}`, async ({ device }) => {
        if (!(await fetch(kIesDir + file, { method: "HEAD" })).ok) {
            throw new SkipError(`Falcor/media/ies/${file} missing (node tools/gen-assets.mjs)`);
        }
        const profile = await LightProfile.createFromIesProfile(device, kIesDir + file);
        const data = profile.rawData;
        console.error(`# IES ${file} (${description}): ${data.length} values, ${Math.trunc(data[3]!)} vertical x ${Math.trunc(data[4]!)} horizontal angles, normalization ${data[0]!.toExponential(3)}`);

        const ctx = device.renderContext;
        await profile.bake(ctx);
        const texture = profile.getTexture()!;
        expectEq([texture.width, texture.height].join("x"), `${kResolution}x${kResolution}`, "baked resolution");

        // Texel-by-texel against the CPU mirror of the bake kernel.
        const baked = new Float32Array((await ctx.readTextureSubresource(texture)).buffer);
        let worst = 0;
        let expectedFlux = 0;
        let nonZero = 0;
        for (let y = 0; y < kResolution; y++) {
            for (let x = 0; x < kResolution; x++) {
                const [expected, flux] = bakeTexel(data, x, y);
                expectedFlux += flux;
                if (expected > 0) nonZero++;
                worst = Math.max(worst, Math.abs(baked[y * kResolution + x]! - expected));
            }
        }
        console.error(`# IES bake: worst texel error ${worst.toExponential(2)}, ${nonZero}/${kResolution * kResolution} texels lit, flux factor ${profile.fluxFactor.toFixed(6)} (expected ${expectedFlux.toFixed(6)})`);
        expectClose(worst, 0, 1e-5, "baked profile matches the kernel's analytic result");
        expectEq(nonZero > 1000, true, `profile is not empty (${nonZero} lit texels)`);
        // The flux factor is the sum over the flux texture (ParallelReduction).
        expectClose(profile.fluxFactor, expectedFlux, Math.max(1e-4, expectedFlux * 1e-4), "flux factor");
    });
}

gpuTest("LightProfile.boundToTheMaterialSystem", async ({ device }) => {
    if (!(await fetch(`${kIesDir}ies-cosine.ies`, { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/ies/ies-cosine.ies missing (node tools/gen-assets.mjs)");
    }
    // A scene without a profile keeps the feature compiled out, as native does.
    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    const scene = new Scene(device, [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }], [{ basic: { baseColor: new float4(1, 1, 1, 1) } }]);
    expectEq(scene.getSceneDefines().get("MATERIAL_SYSTEM_USE_LIGHT_PROFILE"), "0", "feature off without a profile");

    const profile = await LightProfile.createFromIesProfile(device, `${kIesDir}ies-cosine.ies`);
    await profile.bake(device.renderContext);
    scene.lightProfile = profile;
    expectEq(scene.getSceneDefines().get("MATERIAL_SYSTEM_USE_LIGHT_PROFILE"), "1", "feature on once a profile is set");

    // Loading through a .pyscene mirrors native's sceneBuilder.loadLightProfile.
    await initScripting("/node_modules/pyodide");
    const sceneSource = [
        "sceneBuilder.loadLightProfile('ies/ies-cosine.ies')",
        "mat = StandardMaterial('emitter')",
        "mat.emissiveColor = float3(1.0, 1.0, 1.0)",
        "mat.lightProfileEnabled = True",
        "quad = TriangleMesh.createQuad(float2(1.0, 1.0))",
        "meshID = sceneBuilder.addTriangleMesh(quad, mat)",
        "nodeID = sceneBuilder.addNode('quad', Transform(translation=float3(0.0, 0.0, 0.0)))",
        "sceneBuilder.addMeshInstance(nodeID, meshID)",
    ].join("\n");
    const loaded = await runSceneScript(device, sceneSource, "/Falcor/media");
    expectEq(loaded.lightProfile !== null, true, "pyscene loadLightProfile produced a profile");
    expectEq(loaded.getSceneDefines().get("MATERIAL_SYSTEM_USE_LIGHT_PROFILE"), "1", "profile reaches the scene defines");
    expectClose(loaded.lightProfile!.fluxFactor, profile.fluxFactor, 1e-5, "same profile, same flux factor");
    console.error(`# IES via pyscene: flux factor ${loaded.lightProfile!.fluxFactor.toFixed(6)}`);
});

gpuTest("LightProfile.rejectsUnsupportedFiles", async () => {
    // Mirrors parseIesFile's validation (unsupported header, TILT, short data).
    let threw = 0;
    for (const text of ["not an ies file\nTILT=NONE\n", "IESNA:LM-63-2002\nTILT=INCLUDE\n1 2 3\n", "IESNA:LM-63-2002\nTILT=NONE\n1 2 3\n"]) {
        try {
            parseIesProfile(text, true);
        } catch {
            threw++;
        }
    }
    expectEq(threw, 3, "each malformed profile is rejected");
    // A well-formed one parses, and the normalization lands in slot 0.
    const data = parseIesProfile(["IESNA:LM-63-2002", "TILT=NONE", "1 1000 1 2 1 1 1 0 0 0", "1 1 100", "0 90", "0", "500 250"].join("\n"), true);
    expectEq(data.length, 13 + 1 + 2 + 2, "parsed value count");
    expectClose(data[0]!, 1 / 500, 1e-9, "normalization is 1 / maxCandelas");
    expectEq(ParallelReductionType.Sum > 0 && typeof ParallelReduction === "function", true, "reduction type available");
});
