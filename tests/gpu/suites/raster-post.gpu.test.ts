/**
 * GPU tests for the real-time raster/post passes added without a native oracle (Falcor 8 has none of them):
 * SSAO, PhysicalSky, FXAA, DepthOfField, MotionBlur. Each test checks the pass against an independent reference:
 *  - SSAO: analytic cosine-weighted visibility in a floor/wall corner (radius-limited occlusion).
 *  - PhysicalSky: the CPU model (evalPhysicalSky) per pixel; the env map lights a floor as the CPU integral predicts.
 *  - FXAA: a 4x4-supersampled render; the edge error must drop.
 *  - DepthOfField: GBufferRT thin-lens rays (same camera aperture/focus) averaged over many lens samples.
 *  - MotionBlur: GBufferRaster mvec vs analytic reprojection; the blur vs an average of renders over the shutter.
 * Measured numbers are printed as "# ..." lines.
 */

import {
    RenderGraph,
    Scene,
    createPass,
    float2,
    float3,
    float4,
    fovYToFocalLength,
    type Device,
    type Texture,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { createPhysicalSkyEnvMap, evalPhysicalSky, kDefaultPhysicalSky } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

type V3 = [number, number, number];

/** Axis-aligned quad from its 4 corners (counter-clockwise seen from the normal side). */
function quad(c: V3[], n: V3, materialID: number) {
    const uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
    const t = new float4(1, 0, 0, 1);
    return {
        vertices: c.map((p, i) => ({ position: new float3(...p), normal: new float3(...n), tangent: t, texCrd: uv[i]! })),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        materialID,
    };
}

function box(min: V3, max: V3, materialID: number) {
    const [x0, y0, z0] = min, [x1, y1, z1] = max;
    return [
        quad([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0, 1, 0], materialID),
        quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1], materialID),
        quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0, 0, -1], materialID),
        quad([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1, 0, 0], materialID),
        quad([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1, 0, 0], materialID),
    ];
}

const floor = (s: number, m: number) => quad([[-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]], [0, 1, 0], m);
const mat = (r: number, g: number, b: number) => ({ basic: { baseColor: new float4(r, g, b, 1), specular: new float4(0, 1, 0, 0) } });

function setCam(scene: Scene, pos: V3, target: V3, fovYDeg: number): void {
    const cam = scene.camera;
    cam.setPosition(new float3(...pos));
    cam.setTarget(new float3(...target));
    cam.setUpVector(new float3(0, 1, 0));
    cam.setAspectRatio(1);
    cam.setFocalLength(fovYToFocalLength((fovYDeg * Math.PI) / 180, cam.getFrameHeight()));
}

async function read(device: Device, t: Texture): Promise<Float32Array> {
    return new Float32Array((await device.renderContext.readTextureSubresource(t)).buffer);
}

/** Builds graph (passes + edges), marks outputs, executes `frames` times. */
function graphOf(device: Device, scene: Scene | null, size: number, passes: [string, string, Record<string, unknown>?][], edges: [string, string][], outputs: string[]): RenderGraph {
    const g = new RenderGraph(device, "RasterPostTest");
    g.onResize(size, size);
    for (const [type, id, props] of passes) g.addPass(createPass(device, type, props ?? {}), id);
    for (const [a, b] of edges) g.addEdge(a, b);
    for (const o of outputs) g.markOutput(o);
    if (scene) g.setScene(scene);
    return g;
}

const psnr = (a: Float32Array, b: Float32Array, mask?: (i: number) => boolean) => {
    let se = 0, n = 0;
    for (let i = 0; i < a.length / 4; i++) {
        if (mask && !mask(i)) continue;
        for (let c = 0; c < 3; c++) { const d = a[i * 4 + c]! - b[i * 4 + c]!; se += d * d; n++; }
    }
    return 10 * Math.log10(1 / Math.max(se / Math.max(n, 1), 1e-12));
};

gpuTest("SSAO.cornerMatchesAnalytic", async ({ device }) => {
    // floor y = 0 and a wall z = -1 (facing +z). For a point at distance d from the other plane, occluders within R
    // block the cosine-weighted directions with (unit-disk projection) x > d / R: AO = 1 - (acos a - a sqrt(1 - a^2)) / pi.
    const scene = new Scene(device, [floor(20, 0), quad([[-20, 0, -1], [20, 0, -1], [20, 20, -1], [-20, 20, -1]], [0, 0, 1], 0)], [mat(0.8, 0.8, 0.8)]);
    setCam(scene, [0.3, 1.6, 2.6], [0, 0.3, -1], 50);
    const R = 1.0, size = 256;
    const g = graphOf(device, scene, size,
        [["GBufferRaster", "G"], ["SSAO", "AO", { radius: R, falloff: 0.02, sliceCount: 16, stepsPerSlice: 32, maxPixelRadius: 512, noise: false, blur: false }]],
        [["G.posW", "AO.posW"], ["G.normW", "AO.normW"]], ["AO.ao", "G.posW"]);
    g.execute(device.renderContext);
    const ao = await read(device, g.getOutput("AO.ao")!);
    const pos = await read(device, g.getOutput("G.posW")!);
    let err = 0, bias = 0, n = 0, maxErr = 0, bg = 0, flat = 0, flatN = 0;
    for (let i = 0; i < size * size; i++) {
        const x = i % size, y = Math.floor(i / size);
        if (pos[i * 4 + 3] === 0) { bg += Math.abs(ao[i * 4]! - 1); continue; }
        if (x < 4 || y < 4 || x >= size - 4 || y >= size - 4) continue;
        const py = pos[i * 4 + 1]!, pz = pos[i * 4 + 2]!;
        const onFloor = Math.abs(py) < 1e-3;
        const d = onFloor ? pz + 1 : py;
        if (d < 0.03) continue; // the crease itself (a = 0 is a single texel line)
        const a = Math.min(d / R, 1);
        const ref = 1 - (Math.acos(a) - a * Math.sqrt(1 - a * a)) / Math.PI;
        const e = ao[i * 4]! - ref;
        err += Math.abs(e); bias += e; n++; maxErr = Math.max(maxErr, Math.abs(e));
        if (d > 1.2 * R) { flat += Math.abs(ao[i * 4]! - 1); flatN++; }
    }
    console.error(`# SSAO corner: MAE ${(err / n).toFixed(4)} bias ${(bias / n).toFixed(4)} max ${maxErr.toFixed(3)} over ${n} px; |1-AO| beyond R ${(flat / Math.max(flatN, 1)).toExponential(2)}; bg err ${bg}`);
    expectEq(n > 1000, true, "enough corner pixels");
    expectEq(err / n < 0.03, true, `SSAO corner MAE ${err / n}`);
    expectEq(flat / Math.max(flatN, 1) < 1e-3, true, "unoccluded plane AO = 1");
    expectEq(bg, 0, "background AO = 1");
});

gpuTest("PhysicalSky.matchesCpuModelAndLightsScene", async ({ device }) => {
    const scene = new Scene(device, [floor(50, 0)], [mat(0.5, 0.5, 0.5)]);
    const params = { ...kDefaultPhysicalSky, sunDir: [0.4, 0.35, -0.6] as V3, sunAngularRadiusDeg: 2.0 };
    // camera looking towards the sun, horizon in view
    setCam(scene, [0, 1, 0], [0.4, 0.5, -0.6], 70);
    const size = 128;
    const g = graphOf(device, scene, size, [["PhysicalSky", "Sky", { ...params }]], [], ["Sky.dst"]);
    g.execute(device.renderContext);
    const sky = await read(device, g.getOutput("Sky.dst")!);
    const cam = scene.camera;
    const p = cam.getPosition(), t = cam.getTarget();
    const f = [t.x - p.x, t.y - p.y, t.z - p.z]; const fl = Math.hypot(...f); const fwd = f.map((v) => v / fl);
    const r0 = [-fwd[2]!, 0, fwd[0]!]; const rl = Math.hypot(...r0); const right = r0.map((v) => v / rl);
    const up = [right[1]! * fwd[2]! - right[2]! * fwd[1]!, right[2]! * fwd[0]! - right[0]! * fwd[2]!, right[0]! * fwd[1]! - right[1]! * fwd[0]!];
    const tanY = Math.tan(0.5 * cam.getFovY());
    let maxRel = 0, sunPx = 0, mismatchSun = 0;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        const nx = ((x + 0.5) / size) * 2 - 1, ny = ((y + 0.5) / size) * 2 - 1;
        const d = [0, 1, 2].map((k) => fwd[k]! + nx * tanY * right[k]! - ny * tanY * up[k]!);
        const ref = evalPhysicalSky(d, params);
        const i = (y * size + x) * 4;
        const isSun = ref[0] > 10;
        const gpuSun = sky[i]! > 10;
        if (isSun !== gpuSun) { mismatchSun++; continue; } // disk edge texels may flip with fp rounding
        if (isSun) sunPx++;
        for (let c = 0; c < 3; c++) maxRel = Math.max(maxRel, Math.abs(sky[i + c]! - ref[c]!) / Math.max(Math.abs(ref[c]!), 1e-3));
    }
    console.error(`# PhysicalSky vs CPU: max rel err ${maxRel.toExponential(2)}, sun px ${sunPx}, edge flips ${mismatchSun}`);
    expectEq(maxRel < 2e-3, true, `PhysicalSky rel err ${maxRel}`);
    expectEq(sunPx > 0, true, "sun disk visible");
    expectEq(mismatchSun <= 8, true, "sun disk edge flips");

    // Env map: an unoccluded white-ish floor seen from above with only the sky as light must reach
    // albedo / pi * integral(L cos) over the upper hemisphere (CPU quadrature of the same model, no sun).
    const env = createPhysicalSkyEnvMap(device, params, 256, false);
    scene.setEnvMap(env);
    setCam(scene, [0, 5, 0.01], [0, 0, 0], 30);
    const g2 = graphOf(device, scene, 64,
        [["GBufferRaster", "G"], ["MinimalPathTracer", "PT", { maxBounces: 0 }], ["AccumulatePass", "Acc", {}]],
        [["G.vbuffer", "PT.vbuffer"], ["G.viewW", "PT.viewW"], ["PT.color", "Acc.input"]], ["Acc.output"]);
    for (let k = 0; k < 256; k++) g2.execute(device.renderContext);
    const img = await read(device, g2.getOutput("Acc.output")!);
    let E = 0; const nT = 256, nP = 512;
    for (let j = 0; j < nT; j++) {
        const th = ((j + 0.5) / nT) * (Math.PI / 2);
        for (let k = 0; k < nP; k++) {
            const ph = ((k + 0.5) / nP) * 2 * Math.PI;
            const L = evalPhysicalSky([Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)], params, false);
            E += (0.2126 * L[0] + 0.7152 * L[1] + 0.0722 * L[2]) * Math.cos(th) * Math.sin(th) * (Math.PI / 2 / nT) * (2 * Math.PI / nP);
        }
    }
    const refY = (0.5 / Math.PI) * E;
    let Y = 0;
    for (let i = 0; i < 64 * 64; i++) Y += 0.2126 * img[i * 4]! + 0.7152 * img[i * 4 + 1]! + 0.0722 * img[i * 4 + 2]!;
    Y /= 64 * 64;
    console.error(`# PhysicalSky env lighting: floor luminance ${Y.toFixed(4)} vs CPU ${refY.toFixed(4)} (rel ${(Y / refY - 1).toFixed(4)})`);
    expectEq(Math.abs(Y / refY - 1) < 0.1, true, `env-lit floor ${Y} vs ${refY}`);
});

gpuTest("FXAA.reducesEdgeError", async ({ device }) => {
    const scene = new Scene(device, [floor(20, 0), ...box([-0.6, 0, -0.6], [0.6, 1.0, 0.6], 1)], [mat(0.15, 0.15, 0.2), mat(0.95, 0.85, 0.6)]);
    setCam(scene, [2.2, 1.9, 2.9], [0, 0.4, 0], 40);
    const size = 128, ss = 4;
    const lo = graphOf(device, scene, size, [["GBufferRaster", "G"], ["FXAA", "AA"]], [["G.diffuseOpacity", "AA.src"]], ["AA.dst", "G.diffuseOpacity"]);
    lo.execute(device.renderContext);
    const aa = await read(device, lo.getOutput("AA.dst")!);
    const raw = await read(device, lo.getOutput("G.diffuseOpacity")!);
    const hi = graphOf(device, scene, size * ss, [["GBufferRaster", "G"]], [], ["G.diffuseOpacity"]);
    hi.execute(device.renderContext);
    const big = await read(device, hi.getOutput("G.diffuseOpacity")!);
    const ref = new Float32Array(size * size * 4);
    for (let y = 0; y < size * ss; y++) for (let x = 0; x < size * ss; x++)
        for (let c = 0; c < 4; c++) ref[(Math.floor(y / ss) * size + Math.floor(x / ss)) * 4 + c]! += big[(y * size * ss + x) * 4 + c]! / (ss * ss);
    let eRaw = 0, eAA = 0, nEdge = 0, flatChange = 0;
    for (let i = 0; i < size * size; i++) {
        let dr = 0, da = 0;
        for (let c = 0; c < 3; c++) { dr += Math.abs(raw[i * 4 + c]! - ref[i * 4 + c]!); da += Math.abs(aa[i * 4 + c]! - ref[i * 4 + c]!); }
        if (dr > 0.03) { eRaw += dr / 3; eAA += da / 3; nEdge++; } else flatChange = Math.max(flatChange, da / 3 - dr / 3);
    }
    console.error(`# FXAA: edge px ${nEdge}, MAE vs 16x SSAA: no AA ${(eRaw / nEdge).toFixed(4)} -> FXAA ${(eAA / nEdge).toFixed(4)}; PSNR ${psnr(raw, ref).toFixed(2)} -> ${psnr(aa, ref).toFixed(2)} dB`);
    expectEq(nEdge > 50, true, "aliased edges present");
    expectEq(eAA < 0.75 * eRaw, true, `FXAA edge error ${eAA} vs ${eRaw}`);
});

gpuTest("DepthOfField.matchesThinLensRT", async ({ device }) => {
    const scene = new Scene(device, [floor(30, 0), ...box([-1.6, 0, 1.2], [-0.8, 0.8, 2.0], 1), ...box([-0.4, 0, -0.4], [0.4, 1.2, 0.4], 2), ...box([0.9, 0, -4.5], [2.1, 2.0, -3.5], 3)],
        [mat(0.3, 0.3, 0.3), mat(0.9, 0.2, 0.1), mat(0.2, 0.8, 0.3), mat(0.2, 0.3, 0.9)]);
    setCam(scene, [0, 1.4, 6], [0, 0.6, 0], 40);
    scene.camera.setFocalDistance(6.0);
    scene.camera.setApertureRadius(0.12);
    const size = 128;
    const ras = graphOf(device, scene, size, [["GBufferRaster", "G"], ["DepthOfField", "DoF", { maxCoC: 24, radiusScale: 0.5 }]],
        [["G.diffuseOpacity", "DoF.color"], ["G.linearZ", "DoF.depth"]], ["DoF.dst", "G.diffuseOpacity"]);
    ras.execute(device.renderContext);
    const dof = await read(device, ras.getOutput("DoF.dst")!);
    const pin = await read(device, ras.getOutput("G.diffuseOpacity")!);
    const rt = graphOf(device, scene, size, [["GBufferRT", "G", { useDOF: true }]], [], ["G.diffuseOpacity"]);
    const ref = new Float32Array(size * size * 4);
    const N = 256;
    for (let k = 0; k < N; k++) {
        rt.execute(device.renderContext);
        const f = await read(device, rt.getOutput("G.diffuseOpacity")!);
        for (let i = 0; i < ref.length; i++) ref[i]! += f[i]! / N;
    }
    const pPin = psnr(pin, ref), pDof = psnr(dof, ref);
    console.error(`# DoF vs thin-lens RT (${N} lens samples): PSNR pinhole ${pPin.toFixed(2)} dB -> DoF ${pDof.toFixed(2)} dB`);
    expectEq(pDof > pPin + 3, true, `DoF PSNR ${pDof} vs pinhole ${pPin}`);
});

gpuTest("MotionBlur.mvecAndShutterReference", async ({ device }) => {
    const scene = new Scene(device, [floor(30, 0), ...box([-1.6, 0, 1.2], [-0.8, 0.8, 2.0], 1), ...box([-0.4, 0, -0.4], [0.4, 1.2, 0.4], 2), ...box([0.9, 0, -4.5], [2.1, 2.0, -3.5], 3)],
        [mat(0.3, 0.3, 0.3), mat(0.9, 0.2, 0.1), mat(0.2, 0.8, 0.3), mat(0.2, 0.3, 0.9)]);
    const size = 128, fov = 40, shutter = 1.0;
    const A: V3 = [0, 1.4, 6], B: V3 = [0.25, 1.4, 6], tA: V3 = [0, 0.6, 0], tB: V3 = [0.25, 0.6, 0];
    const lerp3 = (a: V3, b: V3, s: number) => a.map((v, i) => v + s * (b[i]! - v)) as V3;
    // reference: average of renders over the shutter interval centred on B (pose(tau) = B + tau (B - A))
    const refG = graphOf(device, scene, size, [["GBufferRaster", "G"]], [], ["G.diffuseOpacity"]);
    const ref = new Float32Array(size * size * 4);
    const N = 64;
    for (let k = 0; k < N; k++) {
        const tau = -shutter / 2 + (shutter * (k + 0.5)) / N;
        setCam(scene, lerp3(B, A, -tau), lerp3(tB, tA, -tau), fov);
        refG.execute(device.renderContext);
        const f = await read(device, refG.getOutput("G.diffuseOpacity")!);
        for (let i = 0; i < ref.length; i++) ref[i]! += f[i]! / N;
    }
    const g = graphOf(device, scene, size, [["GBufferRaster", "G"], ["MotionBlur", "MB", { shutter, tileSize: 20, sampleCount: 31, noise: false }]],
        [["G.diffuseOpacity", "MB.color"], ["G.mvec", "MB.mvec"], ["G.linearZ", "MB.depth"]], ["MB.dst", "G.diffuseOpacity", "G.mvec", "G.posW"]);
    setCam(scene, A, tA, fov);
    const prevVP = scene.camera.getData().viewProjMatNoJitter.clone();
    g.execute(device.renderContext);
    setCam(scene, B, tB, fov);
    g.execute(device.renderContext);
    const mb = await read(device, g.getOutput("MB.dst")!);
    const cur = await read(device, g.getOutput("G.diffuseOpacity")!);
    const mv = await read(device, g.getOutput("G.mvec")!);
    const pos = await read(device, g.getOutput("G.posW")!);
    let mvErr = 0, mvN = 0, mvMax = 0;
    for (let i = 0; i < size * size; i++) {
        if (pos[i * 4 + 3] === 0) continue;
        const p = [pos[i * 4]!, pos[i * 4 + 1]!, pos[i * 4 + 2]!, 1];
        const c = [0, 1, 2, 3].map((r) => p.reduce((s, v, k) => s + prevVP.get(r, k) * v, 0));
        const u = 0.5 * (c[0]! / c[3]!) + 0.5, v = 0.5 - 0.5 * (c[1]! / c[3]!);
        const x = i % size, y = Math.floor(i / size);
        const e = Math.hypot(u - (x + 0.5) / size - mv[i * 2]!, v - (y + 0.5) / size - mv[i * 2 + 1]!) * size;
        mvErr += e; mvN++; mvMax = Math.max(mvMax, Math.hypot(mv[i * 2]!, mv[i * 2 + 1]!) * size);
    }
    const pCur = psnr(cur, ref), pMb = psnr(mb, ref);
    console.error(`# MotionBlur: mvec vs analytic reprojection ${(mvErr / mvN).toExponential(2)} px mean (max |mvec| ${mvMax.toFixed(1)} px); PSNR vs ${N}-pose shutter average: sharp ${pCur.toFixed(2)} dB -> blurred ${pMb.toFixed(2)} dB`);
    expectEq(mvErr / mvN < 1e-2, true, `mvec error ${mvErr / mvN} px`);
    expectEq(mvMax > 3, true, "camera motion produces motion vectors");
    expectEq(pMb > pCur + 3, true, `MotionBlur PSNR ${pMb} vs sharp ${pCur}`);
});
