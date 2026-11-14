/**
 * SSR (Hi-Z screen-space reflections).
 *  1. Mirror parity: on a mirror floor under a sphere, every reflected ray that analytically hits the visible part of
 *     the sphere is found by the Hi-Z march at the CPU-projected hit pixel (+-1.5 px), rays to the sky that pass clearly
 *     away from the sphere miss (rays grazing behind it are ambiguous for a depth buffer and only counted), and the
 *     reflection output equals confidence x colour(hit) x DFG(F0, 0, N.V).
 *  2. Identity: intensity 0 gives the input colour bit for bit on every frame of a moving, temporal, stochastic shot.
 *  3. Temporal accumulation: a static shot with stochastic rays gets less noisy over frames; with temporal off the
 *     output is the same on every frame (no state).
 *  4. Cold vs consecutive: shot X on a fresh graph = shot X after shots A and B on one graph with a camera reset, a
 *     prime render or resetGraphHistory in between (bit for bit); without a cut X differs (control). With temporal off
 *     X is identical even without a cut.
 */

import { LightType, Scene, EnvMap, float2, float3, float4, type Device } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kSize, log, makeScene, read, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, kPoseA, renderShot, setPose, warmUp } from "./fx-helpers.js";

const kSphereC: V3 = [0, 0.6, -0.3];
const kSphereR = 0.5;
/** World-space clearance from the sphere above which a reflected sky ray must miss (thickness 0.03 x view depth ~3). */
const kClearance = 0.15;
const kCam = { pos: [0, 1.1, 3.0] as V3, target: [0, 0.35, -0.2] as V3 };

function sphere(r: number, c: V3, nLon: number, nLat: number, materialID: number) {
    const vertices = [];
    for (let a = 0; a <= nLat; a++) {
        const th = (Math.PI * a) / nLat;
        for (let b = 0; b <= nLon; b++) {
            const ph = (2 * Math.PI * b) / nLon;
            const n: V3 = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
            vertices.push({ position: new float3(c[0] + n[0] * r, c[1] + n[1] * r, c[2] + n[2] * r), normal: new float3(...n), tangent: new float4(-Math.sin(ph), 0, Math.cos(ph), 1), texCrd: new float2(b / nLon, a / nLat) });
        }
    }
    const idx: number[] = [];
    for (let a = 0; a < nLat; a++) {
        for (let b = 0; b < nLon; b++) {
            const i0 = a * (nLon + 1) + b, i1 = i0 + 1, i2 = i0 + nLon + 1, i3 = i2 + 1;
            idx.push(i0, i1, i2, i1, i3, i2);
        }
    }
    return { vertices, indices: new Uint32Array(idx), materialID };
}

const kMirrorScenes = new WeakMap<Device, Scene>();

/** Mirror floor (metallic, roughness 0, F0 0.9) + diffuse sphere, one directional light, constant sky. */
function mirrorScene(device: Device): Scene {
    let scene = kMirrorScenes.get(device);
    if (!scene) {
        const f = 6, uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
        const floor = {
            vertices: ([[-f, 0, f], [f, 0, f], [f, 0, -f], [-f, 0, -f]] as V3[]).map((p, i) => ({ position: new float3(...p), normal: new float3(0, 1, 0), tangent: new float4(1, 0, 0, 1), texCrd: uv[i]! })),
            indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
            materialID: 0,
        };
        const mats = [
            { basic: { baseColor: new float4(0.9, 0.9, 0.9, 1), specular: new float4(0, 0, 1, 0) } },
            { basic: { baseColor: new float4(0.8, 0.2, 0.1, 1), specular: new float4(0, 0.7, 0, 0) } },
        ];
        scene = new Scene(device, [floor, sphere(kSphereR, kSphereC, 128, 64, 1)], mats, [{ type: LightType.Directional, dirW: new float3(0.3, -1, -0.4), intensity: new float3(3, 3, 3) }]);
        scene.setEnvMap(new EnvMap(device, { width: 1, height: 1, data: new Float32Array([0.2, 0.25, 0.35, 1]) } as never));
        kMirrorScenes.set(device, scene);
    }
    scene.camera.setPosition(new float3(...kCam.pos));
    scene.camera.setTarget(new float3(...kCam.target));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
    return scene;
}

const kSsrEdges: [string, string][] = [
    ["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["G.depth", "Z.depth"],
    ["L.color", "S.color"], ["G.depth", "S.depth"], ["G.normW", "S.normW"], ["G.specRough", "S.specRough"], ["Z.hiz", "S.hiz"], ["G.mvec", "S.mvec"],
];

function ssrGraph(device: Device, scene: Scene, props: Record<string, unknown>, outputs: string[]) {
    return buildGraph(device, scene, [["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: "None" }], ["HiZBuild", "Z", {}], ["SSR", "S", props]], kSsrEdges, outputs, kSize, kSize);
}

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const nrm = (a: V3): V3 => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function envBRDFApprox(F0: number, rough: number, NoV: number): number {
    const r = [rough * -1 + 1, rough * -0.0275 + 0.0425, rough * -0.572 + 1.04, rough * 0.022 - 0.04];
    const a004 = Math.min(r[0]! * r[0]!, Math.pow(2, -9.28 * NoV)) * r[0]! + r[1]!;
    return F0 * (-1.04 * a004 + r[2]!) + (1.04 * a004 + r[3]!);
}

gpuTest("SSR.mirrorMatchesAnalyticHits", async ({ device }) => {
    const scene = mirrorScene(device);
    await warmUp(device, scene);
    mirrorScene(device); // warm-up leaves the camera as is; re-apply the pose
    const outs = ["S.hit", "S.reflection", "S.dst", "L.color", "G.posW", "G.specRough"];
    const g = ssrGraph(device, scene, { quality: "mirror", temporal: false, borderFade: 0, intensity: 1 }, outs);
    g.execute(device.renderContext);
    const bufs: Float32Array[] = [];
    for (const o of outs) bufs.push(await read(device, g.getOutput(o)!)); // one readback in flight at a time
    const [hit, refl, dst, col, pos, spr] = bufs;
    const vp = scene.camera.getViewProjMatrix();
    const cam = kCam.pos;
    const project = (q: V3) => {
        const c = [0, 1, 2, 3].map((r) => vp.get(r, 0) * q[0] + vp.get(r, 1) * q[1] + vp.get(r, 2) * q[2] + vp.get(r, 3));
        return [(c[0]! / c[3]! * 0.5 + 0.5) * kSize, (0.5 - c[1]! / c[3]! * 0.5) * kSize];
    };
    let hitN = 0, hitOk = 0, missN = 0, missOk = 0, reflErr = 0, maxPx = 0, grazeHits = 0, grazeMiss = 0;
    const fDiag: string[] = [];
    for (let i = 0; i < kSize * kSize; i++) {
        if (pos![i * 4 + 3] === 0) continue;
        const P: V3 = [pos![i * 4]!, pos![i * 4 + 1]!, pos![i * 4 + 2]!];
        if (Math.abs(P[1]) > 1e-4) continue; // floor pixels only
        const V = nrm(sub(cam, P));
        const R: V3 = [-V[0], V[1], -V[2]];
        // Ray-sphere intersection.
        const oc = sub(P, kSphereC);
        const b = dot(oc, R), c = dot(oc, oc) - kSphereR * kSphereR, disc = b * b - c;
        const conf = hit![i * 4 + 3]!;
        if (disc < 0) {
            // Clearance of the ray from the sphere (perpendicular distance minus radius). A ray that passes just behind
            // the sphere is ambiguous for any screen-space tracer (depth buffer has no thickness); require misses only
            // for rays that pass clearly away from it.
            const clearance = Math.sqrt(Math.max(dot(oc, oc) - b * b, 0)) - kSphereR;
            if (clearance > kClearance) {
                missN++;
                if (conf === 0) missOk++;
                else if (fDiag.length < 12) {
                    const hx = Math.floor(hit![i * 4]! * kSize), hy = Math.floor(hit![i * 4 + 1]! * kSize), j = hy * kSize + hx;
                    fDiag.push(`px(${i % kSize},${Math.floor(i / kSize)})->(${hx},${hy}) hitY ${pos![j * 4 + 1]!.toFixed(3)} hz ${hit![i * 4 + 2]!.toFixed(5)} conf ${conf.toFixed(2)} R ${R.map((x) => x.toFixed(2)).join(",")}`);
                }
            } else if (conf > 0) grazeHits++;
            else grazeMiss++;
            continue;
        }
        const t = -b - Math.sqrt(disc);
        if (t <= 0) continue;
        const Q: V3 = [P[0] + R[0] * t, P[1] + R[1] * t, P[2] + R[2] * t];
        const nQ = nrm(sub(Q, kSphereC));
        // Only hits on the camera-visible part of the sphere, away from its silhouette and the screen border.
        if (dot(nQ, nrm(sub(cam, Q))) < 0.25) continue;
        const [qx, qy] = project(Q);
        if (qx! < 2 || qy! < 2 || qx! > kSize - 2 || qy! > kSize - 2) continue;
        hitN++;
        const ex = hit![i * 4]! * kSize - qx!, ey = hit![i * 4 + 1]! * kSize - qy!;
        const e = Math.hypot(ex, ey);
        if (conf > 0 && e <= 1.5) hitOk++;
        if (conf > 0) maxPx = Math.max(maxPx, e);
        if (conf > 0) {
            // reflection = conf * colour(hit pixel) * DFG(F0, roughness 0, N.V)
            const hx = Math.floor(hit![i * 4]! * kSize), hy = Math.floor(hit![i * 4 + 1]! * kSize), j = hy * kSize + hx;
            const NoV = Math.min(Math.max(V[1], 0), 1);
            for (let ch = 0; ch < 3; ch++) {
                const ref = conf * col![j * 4 + ch]! * envBRDFApprox(spr![i * 4 + ch]!, spr![i * 4 + 3]!, NoV);
                reflErr = Math.max(reflErr, Math.abs(refl![i * 4 + ch]! - ref));
                reflErr = Math.max(reflErr, Math.abs(dst![i * 4 + ch]! - (col![i * 4 + ch]! + refl![i * 4 + ch]!)));
            }
        }
    }
    log(`SSR mirror: analytic sphere hits ${hitN}, found at the right pixel ${hitOk} (${(100 * hitOk / Math.max(hitN, 1)).toFixed(1)} %, max err ${maxPx.toFixed(2)} px); sky rays with clearance > ${kClearance} ${missN}, missed ${missOk} (${(100 * missOk / Math.max(missN, 1)).toFixed(1)} %); grazing sky rays hit/miss ${grazeHits}/${grazeMiss}; reflection/composite max abs err ${reflErr.toExponential(2)}`);
    if (fDiag.length) log(`SSR mirror false hits (first ${fDiag.length}): ${fDiag.join(" | ")}`);
    expectEq(hitN > 300, true, `enough mirror pixels see the sphere (${hitN})`);
    expectEq(hitOk >= 0.95 * hitN, true, "Hi-Z march finds >= 95 % of the analytic hits within 1.5 px");
    expectEq(missN > 1000 && missOk >= 0.98 * missN, true, "rays to the sky that pass clearly away from the sphere miss");
    expectEq(reflErr < 1e-4, true, "reflection = conf x colour(hit) x DFG; dst = colour + reflection");
});

gpuTest("SSR.intensityZeroIsIdentity", async ({ device }) => {
    const scene = makeScene(device);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    const g = ssrGraph(device, scene, { intensity: 0, maxRoughness: 1 }, ["S.dst", "L.color", "S.reflection"]);
    const frames = await renderShot(device, scene, g, kPoseA, 4, ["S.dst", "L.color", "S.reflection"], 0.05);
    const same = frames.every((row) => bitwiseEqual(row[0]!, row[1]!));
    const active = frames.some((row) => row[2]!.some((v, i) => i % 4 === 3 && v > 0));
    log(`SSR intensity 0: dst = color bit for bit on ${frames.length} moving frames ${same}; reflections were traced ${active}`);
    expectEq(active, true, "the shot has reflections");
    expectEq(same, true, "intensity 0 = identity");
});

/** Mean absolute difference between horizontal neighbours of the reflection where both have confidence. */
function roughness(buf: Float32Array): number {
    let s = 0, n = 0;
    for (let y = 0; y < kSize; y++) {
        for (let x = 0; x + 1 < kSize; x++) {
            const i = (y * kSize + x) * 4, j = i + 4;
            if (buf[i + 3]! <= 0 || buf[j + 3]! <= 0) continue;
            for (let c = 0; c < 3; c++) s += Math.abs(buf[i + c]! - buf[j + c]!);
            n++;
        }
    }
    return n ? s / n : 0;
}

gpuTest("SSR.temporalAccumulationAndStateless", async ({ device }) => {
    const scene = makeScene(device);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    const props = { maxRoughness: 1, resolveRadius: 0, temporalAlpha: 0.9 };
    const g = ssrGraph(device, scene, props, ["S.reflection"]);
    const fr = await renderShot(device, scene, g, kPoseA, 12, ["S.reflection"]);
    const n0 = roughness(fr[0]![0]!), n11 = roughness(fr[11]![0]!);
    setPose(scene, kPoseA);
    const g2 = ssrGraph(device, scene, { ...props, temporal: false }, ["S.reflection"]);
    const st = await renderShot(device, scene, g2, kPoseA, 3, ["S.reflection"]);
    const stateless = bitwiseEqual(st[0]![0]!, st[2]![0]!);
    log(`SSR temporal: neighbour noise frame 0 ${n0.toExponential(3)} -> frame 11 ${n11.toExponential(3)}; temporal off: frames 0 and 2 bitwise equal ${stateless}`);
    expectEq(n0 > 0 && n11 < 0.7 * n0, true, "temporal accumulation reduces the stochastic noise");
    expectEq(stateless, true, "temporal off: no state");
});

gpuTest("SSR.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device);
    const mk = (temporal: boolean) => () => ssrGraph(device, scene, { maxRoughness: 1, temporal }, ["S.dst", "S.reflection"]);
    const res = await coldVsConsecutive(device, scene, mk(true), ["S.dst", "S.reflection"], ["camera-reset", "prime", "explicit", "none"]);
    for (const r of res) log(`SSR cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
    for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `cold = consecutive after ${r.method}`);
    expectEq(res.find((r) => r.method === "none")!.differs.some((d) => d), true, "without a cut the history changes X (control)");
    const off = await coldVsConsecutive(device, scene, mk(false), ["S.dst", "S.reflection"], ["none"]);
    log(`SSR temporal off, no cut: X frames bitwise equal ${off[0]!.equalFrames}/${off[0]!.frames}`);
    expectEq(off[0]!.equalFrames, off[0]!.frames, "temporal off: stateless (no cut needed)");
});
