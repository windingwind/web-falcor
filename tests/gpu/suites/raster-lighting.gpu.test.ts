/**
 * Raster deferred lighting + cascaded shadow map (web-only passes, packages/render-passes/src/RasterLighting).
 * Scene: a ground quad (y = 0) under a floating occluder quad, one directional light, optional constant sky.
 *  1. DeferredLighting (scene-material BRDF, ray-traced visibility, no sky) equals MinimalPathTracer
 *     maxBounces = 0 (one directional light => its direct term is deterministic) to fp32 precision.
 *  2. CascadedShadowMap visibility agrees with ray-traced visibility except along the penumbra.
 *  3. Constant sky: the SH9 diffuse irradiance gives albedo * L exactly on an unoccluded plane.
 *  4. Env specular occlusion: Auto is on in every shadow mode when the scene has an env map; the occluded ground
 *     pixels match a CPU mirror-ray / occluder-quad intersection.
 *  5. Without an env map, envSpecularOcclusion changes nothing (bitwise, every shadow mode).
 */

import {
    EnvMap,
    LightType,
    RenderGraph,
    Scene,
    createPass,
    float2,
    float3,
    float4,
    generateTangents,
    type Device,
    type StaticVertex,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 128;

function quad(cx: number, y: number, cz: number, half: number): { vertices: StaticVertex[]; indices: Uint32Array } {
    const n = new float3(0, 1, 0);
    const t0 = new float4(0, 0, 0, 0);
    const vertices: StaticVertex[] = [
        { position: new float3(cx - half, y, cz - half), normal: n, tangent: t0, texCrd: new float2(0, 0) },
        { position: new float3(cx + half, y, cz - half), normal: n, tangent: t0, texCrd: new float2(1, 0) },
        { position: new float3(cx - half, y, cz + half), normal: n, tangent: t0, texCrd: new float2(0, 1) },
        { position: new float3(cx + half, y, cz + half), normal: n, tangent: t0, texCrd: new float2(1, 1) },
    ];
    const indices = new Uint32Array([2, 1, 0, 1, 2, 3]);
    generateTangents(vertices, indices);
    return { vertices, indices };
}

function makeScene(device: Device, sky: number[] | null): Scene {
    const scene = new Scene(
        device,
        [
            { ...quad(0, 0, 0, 2), materialID: 0 },
            { ...quad(0.2, 0.6, 0.1, 0.4), materialID: 1 },
        ],
        [
            { basic: { baseColor: new float4(0.8, 0.8, 0.8, 1), specular: new float4(0, 0.5, 0, 0) } },
            { basic: { baseColor: new float4(0.7, 0.3, 0.2, 1), specular: new float4(0, 0.3, 0, 0) } },
        ],
        [{ type: LightType.Directional, dirW: new float3(0.3, -1, 0.2), intensity: new float3(3, 3, 3) }],
    );
    if (sky) scene.setEnvMap(new EnvMap(device, { width: 1, height: 1, data: new Float32Array([sky[0]!, sky[1]!, sky[2]!, 1]) } as never));
    scene.camera.setPosition(new float3(0.8, 2.2, 2.4));
    scene.camera.setTarget(new float3(0, 0, 0));
    scene.camera.setAspectRatio(1.0);
    return scene;
}

async function renderLighting(device: Device, scene: Scene, lightingProps: Record<string, unknown>, outputs: string[], withRef = false, withPosW = false) {
    const graph = new RenderGraph(device, "RasterLightingTest");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "GBuffer");
    graph.addPass(createPass(device, "CascadedShadowMap", { cascadeCount: 2, mapSize: 1024 }), "CSM");
    graph.addPass(createPass(device, "DeferredLighting", lightingProps), "Lighting");
    graph.addEdge("GBuffer.vbuffer", "Lighting.vbuffer");
    graph.addEdge("GBuffer.viewW", "Lighting.viewW");
    graph.addEdge("CSM.shadowMap", "Lighting.shadowMap");
    if (withRef) {
        graph.addPass(createPass(device, "MinimalPathTracer", { maxBounces: 0 }), "Ref");
        graph.addEdge("GBuffer.vbuffer", "Ref.vbuffer");
        graph.addEdge("GBuffer.viewW", "Ref.viewW");
        graph.markOutput("Ref.color");
    }
    for (const o of outputs) graph.markOutput(`Lighting.${o}`);
    graph.markOutput("GBuffer.mask");
    if (withPosW) graph.markOutput("GBuffer.posW");
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);
    const read = async (name: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(name)!)).buffer);
    const out: Record<string, Float32Array> = {};
    for (const o of outputs) out[o] = await read(`Lighting.${o}`);
    out["mask"] = await read("GBuffer.mask");
    if (withRef) out["ref"] = await read("Ref.color");
    if (withPosW) out["posW"] = await read("GBuffer.posW");
    return out;
}

gpuTest("DeferredLighting.rayTracedMatchesMinimalPathTracerDirect", async ({ device }) => {
    const scene = makeScene(device, null);
    const r = await renderLighting(device, scene, { brdf: "Material", shadowMode: "RayTraced", envLighting: false }, ["color", "visibility"], true);
    let maxDiff = 0, lit = 0, shadowed = 0;
    for (let i = 0; i < size * size; i++) {
        for (let c = 0; c < 3; c++) maxDiff = Math.max(maxDiff, Math.abs(r["color"]![i * 4 + c]! - r["ref"]![i * 4 + c]!));
        if (r["mask"]![i] === 1) (r["visibility"]![i]! > 0.5 ? lit++ : shadowed++);
    }
    console.error(`# rt-vs-mpt: maxAbs=${maxDiff.toExponential(2)} lit=${lit} shadowed=${shadowed}`);
    expectEq(lit > 1000 && shadowed > 200, true, `scene has lit and shadowed pixels (${lit}/${shadowed})`);
    expectEq(maxDiff < 1e-4, true, `DeferredLighting(RT) vs MinimalPathTracer max abs ${maxDiff}`);
});

gpuTest("CascadedShadowMap.visibilityMatchesRayTraced", async ({ device }) => {
    const scene = makeScene(device, null);
    const sm = await renderLighting(device, scene, { brdf: "Material", shadowMode: "ShadowMap", envLighting: false, pcfRadius: 0 }, ["visibility", "color"]);
    const rt = await renderLighting(device, scene, { brdf: "Material", shadowMode: "RayTraced", envLighting: false }, ["visibility", "color"]);
    let hits = 0, mismatch = 0, shadowed = 0;
    for (let i = 0; i < size * size; i++) {
        if (sm["mask"]![i] !== 1) continue;
        hits++;
        if (rt["visibility"]![i]! < 0.5) shadowed++;
        if (Math.abs(sm["visibility"]![i]! - rt["visibility"]![i]!) > 0.5) mismatch++;
    }
    console.error(`# csm-vs-rt: hits=${hits} shadowed=${shadowed} mismatch=${mismatch}`);
    expectEq(shadowed > 200, true, `occluder casts a shadow (${shadowed})`);
    expectEq(mismatch < hits * 0.01, true, `shadow-map vs ray-traced visibility mismatches ${mismatch}/${hits}`);
});

gpuTest("DeferredLighting.constantSkyIrradiance", async ({ device }) => {
    const sky = [0.4, 0.5, 0.6];
    const scene = makeScene(device, sky);
    const r = await renderLighting(device, scene, { brdf: "LambertGGX", analyticLights: false, envBackground: true }, ["diffuse", "color"]);
    // Ground pixels: diffuse = albedo (0.8) * L for a constant sky (SH9 is exact for constants).
    let n = 0, maxRel = 0;
    for (let i = 0; i < size * size; i++) {
        if (r["mask"]![i] !== 1) continue;
        const d = r["diffuse"]!;
        if (Math.abs(d[i * 4 + 1]! - 0.8 * sky[1]!) > 0.1) continue; // skip the occluder quad (green albedo 0.3)
        n++;
        for (let c = 0; c < 3; c++) maxRel = Math.max(maxRel, Math.abs(d[i * 4 + c]! / (0.8 * sky[c]!) - 1));
    }
    // Background pixel (top-left corner looks at the sky) = sky radiance.
    const bg = r["color"]!;
    console.error(`# sky: ground px=${n} maxRel=${maxRel.toExponential(2)} bg=${bg[0]!.toFixed(4)},${bg[1]!.toFixed(4)},${bg[2]!.toFixed(4)}`);
    expectEq(n > 1000, true, `ground pixels ${n}`);
    expectEq(maxRel < 2e-3, true, `SH irradiance rel error ${maxRel}`);
    expectEq(r["mask"]![0] === 0 && Math.abs(bg[0]! - sky[0]!) < 1e-5, true, "background = sky");
});

function bitwiseDiff(a: Float32Array, b: Float32Array): number {
    const ua = new Uint32Array(a.buffer, a.byteOffset, a.length), ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
    let n = 0;
    for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) n++;
    return n;
}

gpuTest("DeferredLighting.envSpecOcclusionAutoInEveryShadowMode", async ({ device }) => {
    const scene = makeScene(device, [0.4, 0.5, 0.6]);
    const base = { brdf: "LambertGGX", analyticLights: false };
    const spec: Record<string, Float32Array> = {};
    let posW: Float32Array | null = null, mask: Float32Array | null = null;
    for (const mode of ["ShadowMap", "None", "RayTraced"]) {
        for (const eo of ["Auto", "On", "Off"]) {
            const r = await renderLighting(device, scene, { ...base, shadowMode: mode, envSpecularOcclusion: eo }, ["specular"], false, true);
            spec[`${mode}/${eo}`] = r["specular"]!;
            posW = r["posW"]!; mask = r["mask"]!;
        }
    }
    // Auto == On in every mode, and (no analytic lights) the env specular does not depend on the shadow mode.
    for (const mode of ["ShadowMap", "None", "RayTraced"]) {
        expectEq(bitwiseDiff(spec[`${mode}/Auto`]!, spec[`${mode}/On`]!), 0, `${mode}: Auto == On (bitwise)`);
        expectEq(bitwiseDiff(spec[`${mode}/Off`]!, spec["RayTraced/Off"]!), 0, `${mode}: Off independent of shadow mode`);
        expectEq(bitwiseDiff(spec[`${mode}/Auto`]!, spec["RayTraced/Auto"]!), 0, `${mode}: Auto independent of shadow mode`);
    }
    // CPU reference on the ground plane (y = 0, N = +y): the mirror ray is blocked iff it crosses the occluder quad
    // (y = 0.6, x in [-0.2, 0.6], z in [-0.3, 0.5]) first.
    const on = spec["ShadowMap/Auto"]!, off = spec["ShadowMap/Off"]!;
    const eye = [0.8, 2.2, 2.4];
    let ground = 0, blocked = 0, mismatch = 0, raised = 0;
    for (let i = 0; i < size * size; i++) {
        if (mask![i] !== 1) continue;
        const p = [posW![i * 4]!, posW![i * 4 + 1]!, posW![i * 4 + 2]!];
        if (Math.abs(p[1]!) > 1e-4) continue;
        ground++;
        const v = [eye[0]! - p[0]!, eye[1]! - p[1]!, eye[2]! - p[2]!];
        const l = Math.hypot(v[0]!, v[1]!, v[2]!);
        const rd = [-v[0]! / l, v[1]! / l, -v[2]! / l];
        const t = 0.6 / rd[1]!;
        const hx = p[0]! + t * rd[0]!, hz = p[2]! + t * rd[2]!;
        const edge = Math.min(Math.abs(hx + 0.2), Math.abs(hx - 0.6), Math.abs(hz + 0.3), Math.abs(hz - 0.5));
        const cpuBlocked = hx > -0.2 && hx < 0.6 && hz > -0.3 && hz < 0.5;
        const gpuBlocked = on[i * 4 + 1]! === 0 && off[i * 4 + 1]! > 0;
        if (on[i * 4 + 1]! > off[i * 4 + 1]!) raised++;
        if (cpuBlocked) blocked++;
        if (cpuBlocked !== gpuBlocked && edge > 0.01) mismatch++;
    }
    console.error(`# env-spec-occlusion: ground=${ground} blocked(cpu)=${blocked} mismatch=${mismatch} raised=${raised}`);
    expectEq(ground > 1000 && blocked > 200, true, `ground ${ground}, mirror-blocked ${blocked}`);
    expectEq(mismatch, 0, "GPU occlusion == CPU mirror-ray test (away from the quad edges)");
    expectEq(raised, 0, "occlusion never raises the specular term");
});

gpuTest("DeferredLighting.envSpecOcclusionWithoutEnvMapIsNoOp", async ({ device }) => {
    const scene = makeScene(device, null);
    for (const mode of ["ShadowMap", "None", "RayTraced"]) {
        const ref = await renderLighting(device, scene, { brdf: "LambertGGX", shadowMode: mode, envSpecularOcclusion: "Off" }, ["color", "specular"]);
        for (const eo of ["Auto", "On"]) {
            const r = await renderLighting(device, scene, { brdf: "LambertGGX", shadowMode: mode, envSpecularOcclusion: eo }, ["color", "specular"]);
            expectEq(bitwiseDiff(r["color"]!, ref["color"]!) + bitwiseDiff(r["specular"]!, ref["specular"]!), 0, `no env map, ${mode}: ${eo} == Off (bitwise)`);
        }
    }
});
