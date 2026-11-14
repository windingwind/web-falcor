/**
 * StyleAovs geometry outputs (ids, posObj, curvature) against CPU references computed from the same G-buffer readbacks:
 *  - ids: raw instance/material/mesh ids agree with the hashed `id` / `materialId` outputs and the scene's materials;
 *  - posObj: equals posW on untransformed meshes and the inverse instance transform of posW on a transformed sphere;
 *  - curvature: bit-for-bit formula parity with a CPU evaluation of the same neighbour estimator, zero on flat faces,
 *    close to 1/R on a sphere of world radius R, and zero when posW is not connected.
 */

import { LightType, Scene, float2, float3, float4, float4x4, type Device } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";
import { camInfo, kSize, log, makeScene, norm3, objectClass, px, readAll, runGraph, type V3 } from "./stylize-helpers.js";

const kEdges: [string, string][] = [["G.vbuffer", "S.vbuffer"], ["G.texC", "S.texC"], ["G.normW", "S.normW"], ["G.posW", "S.posW"]];

/** CPU version of the shader's raw curvature estimator (same neighbours, same validity test). */
function curvatureRef(pos: Float32Array, nrm: Float32Array, camPos: V3, camFwd: V3, thr: number) {
    const depth = (i: number) => (pos[i * 4]! - camPos[0]) * camFwd[0] + (pos[i * 4 + 1]! - camPos[1]) * camFwd[1] + (pos[i * 4 + 2]! - camPos[2]) * camFwd[2];
    const n = (i: number) => norm3([nrm[i * 4]!, nrm[i * 4 + 1]!, nrm[i * 4 + 2]!]);
    return (i: number): number | null => {
        if (pos[i * 4 + 3] === 0) return null;
        const x = i % kSize, y = Math.floor(i / kSize);
        const d = depth(i), ni = n(i);
        let sum = 0, cnt = 0;
        for (const [ox, oy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const qx = x + ox!, qy = y + oy!;
            if (qx < 0 || qy < 0 || qx >= kSize || qy >= kSize) continue;
            const j = qy * kSize + qx;
            if (pos[j * 4 + 3] === 0) continue;
            if (Math.abs(depth(j) - d) > thr * Math.abs(d)) continue;
            const dx = [0, 1, 2].map((c) => pos[j * 4 + c]! - pos[i * 4 + c]!);
            const l2 = dx[0]! * dx[0]! + dx[1]! * dx[1]! + dx[2]! * dx[2]!;
            if (l2 <= 0) continue;
            const nj = n(j);
            sum += ((nj[0] - ni[0]) * dx[0]! + (nj[1] - ni[1]) * dx[1]! + (nj[2] - ni[2]) * dx[2]!) / l2;
            cnt++;
        }
        return cnt > 0 ? sum / cnt : 0;
    };
}

/** Relative curvature mismatch, tolerant to f32 vs f64 rounding (scaled by the local magnitude, floor 1). */
const curvErr = (gpu: number, ref: number) => Math.abs(gpu - ref) / Math.max(1, Math.abs(ref));

gpuTest("StyleAovs.geometryAovs", async ({ device }) => {
    const scene = makeScene(device);
    const outs = ["S.id", "S.materialId", "S.ids", "S.posObj", "S.curvature", "G.posW", "G.normW"];
    let curvFlatMax = 0, creaseMax = 0;
    for (const idSource of ["instance", "mesh"]) {
        const thr = 0.05;
        const g = await runGraph(device, scene, [["StyleAovs", "S", { idSource, curvatureDepthThreshold: thr }]], kEdges, outs);
        const [id, mid, ids, pobj, curv, pos, nrm] = await readAll(device, g, outs);
        const cam = camInfo(scene);
        const ref = curvatureRef(pos, nrm, cam.pos, cam.fwd, thr);
        let bgBad = 0, idBad = 0, mtlBad = 0, posErr = 0, curvBad = 0, curvMax = 0, n = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            const [x, y, z, w] = px(pos, i);
            if (w === 0) {
                if (ids[i * 4 + 3] !== 0 || pobj[i * 4 + 3] !== 0 || curv[i * 4 + 3] !== 0) bgBad++;
                continue;
            }
            n++;
            const sel = idSource === "instance" ? ids[i * 4]! : ids[i * 4 + 2]!;
            if (sel !== id[i * 4 + 1] || ids[i * 4 + 1] !== mid[i * 4 + 1]) idBad++;
            const cls = objectClass(x, y, z);
            if (cls >= 0 && ids[i * 4 + 1] !== cls) mtlBad++;
            posErr = Math.max(posErr, Math.abs(pobj[i * 4]! - x), Math.abs(pobj[i * 4 + 1]! - y), Math.abs(pobj[i * 4 + 2]! - z));
            const r = ref(i)!;
            const c = curv[i * 4]!;
            if (!(curvErr(c, r) < 1e-3)) curvBad++;
            curvMax = Math.max(curvMax, Math.abs(c));
            // Interior of a flat face: every neighbour has the same normal.
            const nn = [nrm[i * 4]!, nrm[i * 4 + 1]!, nrm[i * 4 + 2]!];
            const xi = i % kSize, yi = Math.floor(i / kSize);
            let flat = xi > 0 && yi > 0 && xi < kSize - 1 && yi < kSize - 1;
            for (const j of [i - 1, i + 1, i - kSize, i + kSize]) {
                if (!flat) break;
                if (pos[j * 4 + 3] === 0 || Math.abs(nrm[j * 4]! - nn[0]!) + Math.abs(nrm[j * 4 + 1]! - nn[1]!) + Math.abs(nrm[j * 4 + 2]! - nn[2]!) > 1e-6) flat = false;
            }
            if (flat) curvFlatMax = Math.max(curvFlatMax, Math.abs(c));
            else creaseMax = Math.max(creaseMax, Math.abs(c));
        }
        log(`StyleAovs geometry[${idSource}]: px ${n}, bg bad ${bgBad}, id bad ${idBad}, material bad ${mtlBad}, posObj-posW max ${posErr.toExponential(2)}, curvature mismatches ${curvBad}, |k| max ${curvMax.toFixed(2)}, flat |k| max ${curvFlatMax.toExponential(2)}, crease |k| max ${creaseMax.toFixed(2)}`);
        expectEq(bgBad, 0, "background pixels are zero");
        expectEq(idBad, 0, "raw ids agree with the id/materialId outputs");
        expectEq(mtlBad, 0, "material ids match the scene");
        expectEq(posErr < 1e-5, true, "posObj = posW for untransformed meshes");
        expectEq(curvBad, 0, "curvature = CPU estimator on the same readbacks");
    }
    expectEq(curvFlatMax < 1e-3, true, "curvature ~ 0 inside flat faces");
    expectEq(creaseMax > 1, true, "curvature large across box creases");

    // Without posW the curvature output is 0 (alpha 1 on geometry).
    const g = await runGraph(device, scene, [["StyleAovs", "S", {}]], kEdges.filter(([, b]) => b !== "S.posW"), ["S.curvature", "G.posW"]);
    const [curv, pos] = await readAll(device, g, ["S.curvature", "G.posW"]);
    let nz = 0;
    for (let i = 0; i < kSize * kSize; i++) if (pos[i * 4 + 3] !== 0 && curv[i * 4] !== 0) nz++;
    expectEq(nz, 0, "curvature 0 without posW");
});

/** UV sphere (object radius r) as one mesh. */
function sphere(r: number, nLon: number, nLat: number, materialID: number, transform: float4x4) {
    const vertices = [];
    for (let a = 0; a <= nLat; a++) {
        const th = (Math.PI * a) / nLat;
        for (let b = 0; b <= nLon; b++) {
            const ph = (2 * Math.PI * b) / nLon;
            const n: V3 = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
            vertices.push({ position: new float3(n[0] * r, n[1] * r, n[2] * r), normal: new float3(...n), tangent: new float4(-Math.sin(ph), 0, Math.cos(ph), 1), texCrd: new float2(b / nLon, a / nLat) });
        }
    }
    const idx: number[] = [];
    for (let a = 0; a < nLat; a++) {
        for (let b = 0; b < nLon; b++) {
            const i0 = a * (nLon + 1) + b, i1 = i0 + 1, i2 = i0 + nLon + 1, i3 = i2 + 1;
            idx.push(i0, i1, i2, i1, i3, i2);
        }
    }
    return { vertices, indices: new Uint32Array(idx), materialID, transform };
}

const kSphere = { r: 0.5, s: 1.4, rotY: 0.6, t: [0.3, 0.75, -0.2] as V3 };

function sphereScene(device: Device): Scene {
    const { r, s, rotY, t } = kSphere;
    const c = Math.cos(rotY) * s, sn = Math.sin(rotY) * s;
    const xf = float4x4.fromRows([[c, 0, sn, t[0]], [0, s, 0, t[1]], [-sn, 0, c, t[2]], [0, 0, 0, 1]]);
    const f = 5, uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
    const floor = {
        vertices: ([[-f, 0, f], [f, 0, f], [f, 0, -f], [-f, 0, -f]] as V3[]).map((p, i) => ({ position: new float3(...p), normal: new float3(0, 1, 0), tangent: new float4(1, 0, 0, 1), texCrd: uv[i]! })),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        materialID: 0,
    };
    const mats = [[0.7, 0.7, 0.7], [0.2, 0.5, 0.8]].map((a) => ({ basic: { baseColor: new float4(a[0]!, a[1]!, a[2]!, 1), specular: new float4(0, 0.5, 0, 0) } }));
    const scene = new Scene(device, [floor, sphere(r, 96, 48, 1, xf)], mats, [{ type: LightType.Directional, dirW: new float3(0.3, -1, 0.2), intensity: new float3(3, 3, 3) }]);
    scene.camera.setPosition(new float3(2.0, 1.6, 2.2));
    scene.camera.setTarget(new float3(t[0], t[1] - 0.1, t[2]));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
    return scene;
}

gpuTest("StyleAovs.posObjCurvatureSphere", async ({ device }) => {
    const scene = sphereScene(device);
    const thr = 0.05;
    const outs = ["S.ids", "S.posObj", "S.curvature", "G.posW", "G.normW"];
    const g = await runGraph(device, scene, [["StyleAovs", "S", { curvatureDepthThreshold: thr }]], kEdges, outs);
    const [ids, pobj, curv, pos, nrm] = await readAll(device, g, outs);
    const cam = camInfo(scene);
    const ref = curvatureRef(pos, nrm, cam.pos, cam.fwd, thr);
    const { r, s, rotY, t } = kSphere;
    const R = r * s;
    let n = 0, posErr = 0, curvBad = 0, radErr = 0;
    const ks: number[] = [];
    for (let i = 0; i < kSize * kSize; i++) {
        if (pos[i * 4 + 3] === 0 || ids[i * 4 + 1] !== 1) continue;
        n++;
        // Inverse instance transform of posW: x_obj = R_y(-rotY) (x_w - t) / s.
        const d = [0, 1, 2].map((c) => (pos[i * 4 + c]! - t[c]!) / s);
        const ox = Math.cos(rotY) * d[0]! - Math.sin(rotY) * d[2]!, oz = Math.sin(rotY) * d[0]! + Math.cos(rotY) * d[2]!;
        posErr = Math.max(posErr, Math.abs(pobj[i * 4]! - ox), Math.abs(pobj[i * 4 + 1]! - d[1]!), Math.abs(pobj[i * 4 + 2]! - oz));
        radErr = Math.max(radErr, Math.abs(Math.hypot(pobj[i * 4]!, pobj[i * 4 + 1]!, pobj[i * 4 + 2]!) - r));
        const c = curv[i * 4]!;
        if (!(curvErr(c, ref(i)!) < 1e-3)) curvBad++;
        // Interior pixels only (all four neighbours on the sphere) for the magnitude check.
        const ok = [i - 1, i + 1, i - kSize, i + kSize].every((j) => j >= 0 && j < kSize * kSize && pos[j * 4 + 3] !== 0 && ids[j * 4 + 1] === 1);
        if (ok) ks.push(c);
    }
    ks.sort((a, b) => a - b);
    const med = ks[Math.floor(ks.length / 2)] ?? 0;
    log(`StyleAovs sphere: px ${n}, posObj max err ${posErr.toExponential(2)}, |posObj|-r max ${radErr.toExponential(2)}, curvature mismatches ${curvBad}, interior median k ${med.toFixed(4)} (1/R = ${(1 / R).toFixed(4)})`);
    expectEq(n > 0.1 * kSize * kSize, true, "sphere covers the frame");
    expectEq(posErr < 1e-4, true, "posObj = inverse instance transform of posW");
    expectEq(radErr < 2e-3, true, "posObj lies on the object-space sphere (tessellation tolerance)");
    expectEq(curvBad, 0, "curvature = CPU estimator on the same readbacks");
    expectEq(Math.abs(med * R - 1) < 0.1, true, "median curvature on the sphere ~ 1/R");
});
