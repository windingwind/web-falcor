/**
 * VolumetricClouds on the stylize test scene (floor + two boxes, one directional light), with scene-scaled slabs:
 *  1. CPU parity: full resolution, temporal off: every pixel's march (S, T, front depth) matches marchCloudRay() for
 *     (a) a low flat slab around the boxes seen from above (march clamped to the scene surfaces) and (b) a high curved
 *     slab seen by an upward camera (sky only); dst = color * T + S.
 *  2. Identity: coverage 0, density 0 and opacity 0 give dst = color bit for bit on every frame of a moving temporal
 *     shot (layer 0, transmittance 1).
 *  3. Weather: more coverage = more cloud; cloud type changes the image; wind x time moves the clouds (a time shift
 *     equals a camera shift by the wind displacement).
 *  4. Temporal: frame 0 after a reset = the non-temporal render (bitwise); later frames of a static shot change; temporal
 *     off is stateless; half and quarter resolution stay close to full resolution.
 *  5. Cold vs consecutive (full and half resolution): shot X on a fresh graph = shot X after shots A and B on one graph
 *     with a camera reset, a prime render or resetGraphHistory in between (bit for bit); without a cut X differs.
 *  6. DepthComposite route: layer + transmittance + layerDepth through DepthComposite = the pass's own dst.
 */

import { type Device, type RenderGraph, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { cloudCamera, cloudFrameConstants, cloudJitter, cloudPixelRay, marchCloudRay, type VolumetricClouds } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kL, kSize, log, makeScene, type PassDecl, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, kPoseA, kPoseX, renderShot, setPose, warmUp, type Pose } from "./fx-helpers.js";

const kSky: V3 = [0.25, 0.35, 0.55];

/** Low flat slab between the boxes' tops and the camera (the camera looks down through it). */
const kLow = {
    coverage: 0.6, cloudType: 0.7, density: 4, weatherScale: 0.25, weatherAmount: 0.5, typeVariation: 0.3,
    shapeScale: 1.6, octaves: 4, detailScale: 7, detailStrength: 0.35, detailOctaves: 2,
    cloudBottom: 0.5, cloudTop: 1.6, groundHeight: 0, planetRadius: 0, maxDistance: 30, wind: [0.3, 0, 0.1], time: 1.5, seed: 3,
    lightDir: kL, lightIntensity: 2, steps: 32, lightSteps: 5, lightStep: 0.08, lightStepGrowth: 1.4, minTransmittance: 0.01,
    ambientTop: [0.4, 0.45, 0.55], ambientBottom: [0.15, 0.18, 0.24], jitter: 1,
};
/** High curved slab above the scene for an upward camera. */
const kHigh = { ...kLow, cloudBottom: 3, cloudTop: 5, planetRadius: 60, maxDistance: 60, shapeScale: 0.5, detailScale: 2.5, weatherScale: 0.08, density: 1.5, lightStep: 0.25 };
const kPoseUp: Pose = { pos: [0.5, 1.2, 4.5], target: [0.2, 3.2, 0.5] };

function cloudGraph(device: Device, scene: Scene, props: Record<string, unknown>, outputs: string[]): RenderGraph {
    const passes: PassDecl[] = [["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: "None" }], ["VolumetricClouds", "C", props]];
    const edges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["L.color", "C.color"], ["G.posW", "C.posW"]];
    return buildGraph(device, scene, passes, edges, outputs, kSize, kSize);
}

const meanAlpha = (layer: Float32Array) => { let s = 0; for (let i = 3; i < layer.length; i += 4) s += layer[i]!; return s / (layer.length / 4); };
const maxAbs = (a: Float32Array, b: Float32Array) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!)); return m; };
const meanAbs = (a: Float32Array, b: Float32Array) => { let m = 0; for (let i = 0; i < a.length; i++) m += Math.abs(a[i]! - b[i]!); return m / a.length; };

gpuTest("VolumetricClouds.cpuParity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    const outs = ["C.layer", "C.layerDepth", "C.dst", "L.color", "G.posW"];
    for (const [name, props, pose] of [["low flat slab over geometry", kLow, kPoseA], ["high curved slab, sky", kHigh, kPoseUp]] as const) {
        const g = cloudGraph(device, scene, { ...props, temporal: false }, outs);
        const [shot] = await renderShot(device, scene, g, pose, 1, outs);
        const [layer, depth, dst, color, posW] = shot!;
        const cam = cloudCamera(scene);
        const f = cloudFrameConstants({ ...props, temporal: false }, cam, kL);
        const wl = Math.hypot(...cam.W);
        let bad = 0, maxErr = 0, dstErr = 0, depthBad = 0, clouded = 0, overGeo = 0;
        for (let y = 0; y < kSize; y++) for (let x = 0; x < kSize; x++) {
            const i = y * kSize + x;
            const dir = cloudPixelRay(cam, x, y, kSize, kSize);
            const geo = posW![i * 4 + 3] !== 0;
            const maxT = geo ? Math.hypot(posW![i * 4]! - cam.pos[0], posW![i * 4 + 1]! - cam.pos[1], posW![i * 4 + 2]! - cam.pos[2]) : 3e38;
            const r = marchCloudRay(f, dir, maxT, cloudJitter(f, x, y, 0));
            const ref = [r.S[0], r.S[1], r.S[2], 1 - r.T];
            let e = 0;
            for (let k = 0; k < 4; k++) e = Math.max(e, Math.abs(layer![i * 4 + k]! - ref[k]!) / Math.max(1, Math.abs(ref[k]!)));
            maxErr = Math.max(maxErr, e);
            if (!(e <= 2e-3)) bad++;
            const dRef = r.front >= 0 ? (r.front * (dir[0] * cam.W[0] + dir[1] * cam.W[1] + dir[2] * cam.W[2])) / wl : 2.9e38;
            if (!(Math.abs(depth![i * 4]! - dRef) <= 1e-3 * Math.max(1, dRef))) depthBad++;
            if (1 - r.T > 0.05) { clouded++; if (geo) overGeo++; }
            for (let k = 0; k < 3; k++) dstErr = Math.max(dstErr, Math.abs(dst![i * 4 + k]! - (color![i * 4 + k]! * (1 - layer![i * 4 + 3]!) + layer![i * 4 + k]!)));
        }
        const n = kSize * kSize;
        log(`VolumetricClouds parity [${name}]: (S, 1-T) max rel err ${maxErr.toExponential(2)}, px above 2e-3: ${bad}/${n}; front depth off: ${depthBad}/${n}; clouded px ${clouded} (over geometry ${overGeo}); dst = color*T + S max err ${dstErr.toExponential(2)}; mean alpha ${meanAlpha(layer!).toFixed(3)}`);
        expectEq(bad <= 0.01 * n, true, `march = CPU reference (${name})`);
        expectEq(depthBad <= 0.01 * n, true, `front depth = CPU reference (${name})`);
        expectEq(dstErr < 1e-4, true, "dst = color * T + S");
        expectEq(clouded > 0.05 * n, true, "clouds are visible (non-trivial test)");
    }
});

gpuTest("VolumetricClouds.identity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    let ok = true;
    for (const props of [{ coverage: 0 }, { density: 0 }, { opacity: 0 }]) {
        const outs = ["C.dst", "L.color", "C.layer", "C.transmittance"];
        const g = cloudGraph(device, scene, { ...kLow, ...props, temporal: true }, outs);
        const frames = await renderShot(device, scene, g, kPoseA, 3, outs, 0.06);
        for (const [dst, color, layer, tr] of frames) {
            let layersOk = true;
            for (let i = 0; i < layer!.length; i++) if (layer![i] !== 0 || tr![i] !== 1) layersOk = false;
            ok &&= bitwiseEqual(dst!, color!) && layersOk;
        }
        log(`VolumetricClouds identity ${JSON.stringify(props)}: 3 moving temporal frames dst = color bitwise, layer 0, T 1: ${ok}`);
    }
    expectEq(ok, true, "coverage 0 / density 0 / opacity 0 = identity");
});

gpuTest("VolumetricClouds.weather", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    const shot = async (props: Record<string, unknown>, pose: Pose = kPoseUp, extra: string[] = []) =>
        (await renderShot(device, scene, cloudGraph(device, scene, { ...kHigh, temporal: false, ...props }, ["C.layer", ...extra]), pose, 1, ["C.layer", ...extra]))[0]!;
    const cov = [];
    for (const c of [0.3, 0.5, 0.7]) cov.push(meanAlpha((await shot({ coverage: c }))[0]!));
    const stratus = (await shot({ cloudType: 0, typeVariation: 0 }))[0]!, cumulus = (await shot({ cloudType: 1, typeVariation: 0 }))[0]!;
    const t0 = (await shot({ time: 0 }))[0]!, t1 = (await shot({ time: 4 }))[0]!;
    // Time shift by dt with wind w = camera shifted by w * dt (no scene clamp: sky-only pixels, flat slab, horizontal wind).
    const flat = { planetRadius: 0, wind: [0.25, 0, -0.15], jitter: 0 };
    const a = (await shot({ ...flat, time: 0 }, kPoseUp, ["G.posW"]));
    const dtShift = 2;
    const shifted: Pose = { pos: [kPoseUp.pos[0] + 0.5, kPoseUp.pos[1], kPoseUp.pos[2] - 0.3], target: [kPoseUp.target[0] + 0.5, kPoseUp.target[1], kPoseUp.target[2] - 0.3] };
    const b = (await shot({ ...flat, time: dtShift }, shifted, ["G.posW"]));
    let skyErr = 0, skyN = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        if (a[1]![i * 4 + 3] !== 0 || b[1]![i * 4 + 3] !== 0) continue;
        skyN++;
        for (let k = 0; k < 4; k++) skyErr = Math.max(skyErr, Math.abs(a[0]![i * 4 + k]! - b[0]![i * 4 + k]!));
    }
    log(`VolumetricClouds weather: mean alpha at coverage 0.3/0.5/0.7 = ${cov.map((x) => x.toFixed(3)).join("/")}; stratus vs cumulus mean |diff| ${meanAbs(stratus, cumulus).toExponential(2)} (alpha ${meanAlpha(stratus).toFixed(3)} vs ${meanAlpha(cumulus).toFixed(3)}); time 0 vs 4 mean |diff| ${meanAbs(t0, t1).toExponential(2)}; wind time shift vs camera shift max |diff| over ${skyN} sky px ${skyErr.toExponential(2)}`);
    expectEq(cov[0]! < cov[1]! && cov[1]! < cov[2]!, true, "more coverage = more cloud");
    expectEq(meanAbs(stratus, cumulus) > 1e-3, true, "cloud type changes the clouds");
    expectEq(meanAbs(t0, t1) > 1e-3, true, "wind x time moves the clouds");
    expectEq(skyN > 0.2 * kSize * kSize && skyErr < 2e-2, true, "a time shift equals a camera shift by the wind displacement");
});

gpuTest("VolumetricClouds.temporal", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseX);
    scene.camera.resetHistory();
    const gT = cloudGraph(device, scene, { ...kLow, temporal: true }, ["C.dst"]);
    const t = await renderShot(device, scene, gT, kPoseX, 4, ["C.dst"]);
    const c = gT.getPasses().find((p) => p.name === "C")!.pass as unknown as VolumetricClouds;
    const gS = cloudGraph(device, scene, { ...kLow, temporal: false }, ["C.dst"]);
    const s = await renderShot(device, scene, gS, kPoseX, 3, ["C.dst"]);
    const frame0 = bitwiseEqual(t[0]![0]!, s[0]![0]!);
    const changes = [1, 2, 3].map((k) => !bitwiseEqual(t[k]![0]!, t[k - 1]![0]!));
    const stateless = bitwiseEqual(s[0]![0]!, s[1]![0]!) && bitwiseEqual(s[1]![0]!, s[2]![0]!);
    const res: string[] = [];
    let lowOk = true;
    const ref = (await renderShot(device, scene, cloudGraph(device, scene, { ...kLow, temporal: false, jitter: 0, steps: 64 }, ["C.dst"]), kPoseX, 1, ["C.dst"]))[0]![0]!;
    for (const resolution of ["full", "half", "quarter"]) {
        const lo = (await renderShot(device, scene, cloudGraph(device, scene, { ...kLow, temporal: false, resolution }, ["C.dst"]), kPoseX, 1, ["C.dst"]))[0]![0]!;
        const e = meanAbs(lo, ref);
        res.push(`${resolution} ${e.toExponential(2)}`);
        lowOk &&= e < 0.03;
    }
    log(`VolumetricClouds temporal: frame 0 = non-temporal ${frame0}; frames 1-3 change ${changes.join(",")}; frameIndex ${c.frameIndex}; temporal off stateless ${stateless}; mean |dst - 64-step full-res| by resolution: ${res.join(", ")}`);
    expectEq(frame0, true, "first frame after a reset = non-temporal render");
    expectEq(changes.every((x) => x), true, "temporal accumulation changes the static shot over frames");
    expectEq(c.frameIndex, 3, "history counts frames since the reset");
    expectEq(stateless, true, "temporal off: no state");
    expectEq(lowOk, true, "low-resolution marching stays close to full resolution");
});

gpuTest("VolumetricClouds.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device, kSky);
    for (const resolution of ["full", "half"]) {
        const props = { ...kLow, resolution, temporal: true };
        const res = await coldVsConsecutive(device, scene, () => cloudGraph(device, scene, props, ["C.dst", "C.layer"]), ["C.dst", "C.layer"], ["camera-reset", "prime", "explicit", "none"]);
        for (const r of res) log(`VolumetricClouds cold vs consecutive [${resolution}, ${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
        for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `cold = consecutive after ${r.method} (${resolution})`);
        expectEq(res.find((r) => r.method === "none")!.differs[0], true, `without a cut the leaked history changes X (control, ${resolution})`);
    }
    const off = await coldVsConsecutive(device, scene, () => cloudGraph(device, scene, { ...kLow, temporal: false }, ["C.dst"]), ["C.dst"], ["none"]);
    log(`VolumetricClouds cold vs consecutive, temporal off, no cut: ${off[0]!.equalFrames}/${off[0]!.frames} bitwise`);
    expectEq(off[0]!.equalFrames, off[0]!.frames, "temporal off: stateless even without a cut");
});

gpuTest("VolumetricClouds.depthCompositeRoute", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    const passes: PassDecl[] = [["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: "None" }], ["VolumetricClouds", "C", { ...kLow, temporal: false }],
        ["DepthComposite", "D", { alphaMode: "premultiplied" }]];
    const edges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["L.color", "C.color"], ["G.posW", "C.posW"],
        ["L.color", "D.src"], ["G.posW", "D.posW"], ["C.layer", "D.layer"], ["C.layerDepth", "D.layerDepth"], ["C.transmittance", "D.transmittance"]];
    let worst = 0, n = 0;
    for (const pose of [kPoseA, kPoseX]) {
        const g = buildGraph(device, scene, passes, edges, ["C.dst", "D.dst"], kSize, kSize);
        const [[a, b]] = (await renderShot(device, scene, g, pose, 1, ["C.dst", "D.dst"])) as [[Float32Array, Float32Array]];
        worst = Math.max(worst, maxAbs(a, b));
        n += bitwiseEqual(a, b) ? 1 : 0;
    }
    log(`VolumetricClouds DepthComposite route vs own dst: max |diff| ${worst.toExponential(2)}, bitwise equal ${n}/2 poses`);
    expectEq(worst < 1e-5, true, "DepthComposite(layer, transmittance, layerDepth) = own dst");
});
