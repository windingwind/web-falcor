/**
 * FroxelFog (froxel volumetric fog) on the stylize test scene (floor + two boxes, one directional light).
 *  1. CPU parity: on a small grid (24 x 24 x 16, temporal off) the fog output (S, T) and dst match a CPU reference of
 *     all three dispatches (media + in-scatter with CPU ray/box visibility, front-to-back integration, trilinear apply),
 *     for visibility None and RayTraced, float and half precision; the half reference rounds where the GPU stores.
 *     A second scene (floor, one box, a dim directional light and two point lights) checks otherLights the same way.
 *  2. Shadowed in-scattering: the boxes' shadows cut the fog (RayTraced and ShadowMap in-scatter below None; ShadowMap
 *     agrees with RayTraced on most pixels).
 *  3. Identity: density 0 (and local volumes of density 0) gives dst = color bit for bit on every frame of a moving
 *     temporal shot, fog output (0, 0, 0, 1).
 *  4. Temporal: frame 0 after a reset = the non-temporal render (bitwise); later frames of a static shot change (jittered
 *     accumulation); with temporal off every frame is the same (no state).
 *  5. Cold vs consecutive: shot X on a fresh graph = shot X after shots A and B on one graph with a camera reset, a prime
 *     render or resetGraphHistory in between (bit for bit, ShadowMap visibility, noise, half precision); without a cut X
 *     differs (control). With temporal off X is identical even without a cut.
 */

import { EnvMap, LightType, Scene, float2, float3, float4, type Device, type RenderGraph } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { froxelJitter, type FroxelFog } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kBoxA, kBoxB, kL, kLightDirW, kSize, log, makeScene, type PassDecl, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, kPoseA, kPoseX, renderShot, setPose, warmUp } from "./fx-helpers.js";

const kSky: V3 = [0.03, 0.035, 0.05]; // own scene instance
const kLightI = 3;

function fogGraph(device: Device, scene: Scene, props: Record<string, unknown>, outputs: string[], shadow = "ShadowMap"): RenderGraph {
    const passes: PassDecl[] = [
        ["CascadedShadowMap", "CSM", { cascadeCount: 2, mapSize: 1024 }],
        ["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: shadow }],
        ["FroxelFog", "F", props],
    ];
    const edges: [string, string][] = [
        ["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["CSM.shadowMap", "L.shadowMap"],
        ["L.color", "F.color"], ["G.posW", "F.posW"], ["CSM.shadowMap", "F.shadowMap"],
    ];
    return buildGraph(device, scene, passes, edges, outputs, kSize, kSize);
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU reference (mirrors FroxelFogCommon.slangh, FroxelFogScatter.cs.slang, FroxelFog.cs.slang).

/** Round to the nearest half (ties to even), as the RGBA16F packing; inputs clamped to [0, 65504] first. */
function f16(x: number): number {
    x = Math.min(Math.max(x, 0), 65504);
    if (x === 0) return 0;
    let e = Math.floor(Math.log2(x));
    if (2 ** e > x) e--;
    if (2 ** (e + 1) <= x) e++;
    e = Math.max(e, -14);
    const q = 2 ** (e - 10);
    const t = x / q;
    let r = Math.floor(t);
    const fr = t - r;
    if (fr > 0.5 || (fr === 0.5 && r % 2 === 1)) r++;
    return r * q;
}

function hash(c: V3): number {
    let h = (Math.imul(c[0] >>> 0, 73856093) ^ Math.imul(c[1] >>> 0, 19349663) ^ Math.imul(c[2] >>> 0, 83492791)) >>> 0;
    h = Math.imul((h >>> 16) ^ h, 0x45d9f3b) >>> 0;
    h = Math.imul((h >>> 16) ^ h, 0x45d9f3b) >>> 0;
    h = ((h >>> 16) ^ h) >>> 0;
    return (h & 0xffffff) / 16777215;
}

function valueNoise(p: V3): number {
    const i = p.map(Math.floor) as V3;
    const f = p.map((x, k) => x - i[k]!);
    const w = f.map((t) => t * t * (3 - 2 * t));
    const n = (dx: number, dy: number, dz: number) => hash([i[0] + dx, i[1] + dy, i[2] + dz]);
    const l = (a: number, b: number, t: number) => a + (b - a) * t;
    const x00 = l(n(0, 0, 0), n(1, 0, 0), w[0]!), x10 = l(n(0, 1, 0), n(1, 1, 0), w[0]!);
    const x01 = l(n(0, 0, 1), n(1, 0, 1), w[0]!), x11 = l(n(0, 1, 1), n(1, 1, 1), w[0]!);
    return l(l(x00, x10, w[1]!), l(x01, x11, w[1]!), w[2]!);
}

const fbm = (p: V3) => (0.5 * valueNoise(p) + 0.25 * valueNoise(p.map((x) => x * 2.03 + 17.1) as V3) + 0.125 * valueNoise(p.map((x) => x * 4.01 + 31.7) as V3)) / 0.875;

interface RefParams {
    density: number; heightFalloff: number; heightBase: number; albedo: V3; anisotropy: number; ambient: V3; noise: number; noiseScale: number;
    wind: V3; time: number; far: number; spread: number; grid: V3; skyFog: boolean;
    volumes: { shape: "box" | "sphere"; center: V3; halfExtent: V3; density: number; albedo: V3; softness: number; noise: number; noiseScale: number }[];
}

function blocked(p: V3, L: V3, maxT = Infinity, boxes: [V3, V3][] = [kBoxA, kBoxB]): boolean {
    if (p[1] < 0 && L[1] > 0) { // floor quad (12 x 12 at y = 0)
        const t = -p[1] / L[1];
        if (t < maxT && Math.abs(p[0] + L[0] * t) <= 6 && Math.abs(p[2] + L[2] * t) <= 6) return true;
    }
    for (const [mn, mx] of boxes) {
        let t0 = 0, t1 = maxT;
        for (let a = 0; a < 3; a++) {
            const inv = 1 / L[a]!;
            let ta = (mn[a]! - p[a]!) * inv, tb = (mx[a]! - p[a]!) * inv;
            if (ta > tb) [ta, tb] = [tb, ta];
            t0 = Math.max(t0, ta);
            t1 = Math.min(t1, tb);
        }
        if (t0 <= t1) return true;
    }
    return false;
}

/** Scene lights for the reference: the primary (directional, shadowed by the visibility mode) and point lights. */
interface RefLights { L: V3; I: number; points: { pos: V3; I: V3 }[]; otherScale: number; boxes: [V3, V3][] }
const kStylizeLights: RefLights = { L: kL, I: kLightI, points: [], otherScale: 0, boxes: [kBoxA, kBoxB] };

function cpuFog(scene: Scene, rp: RefParams, vis: "None" | "RayTraced", half: boolean, posW: Float32Array, lights: RefLights = kStylizeLights): Float32Array {
    const d = scene.camera.getData();
    const pos: V3 = [d.posW.x, d.posW.y, d.posW.z], U: V3 = [d.cameraU.x, d.cameraU.y, d.cameraU.z];
    const V: V3 = [d.cameraV.x, d.cameraV.y, d.cameraV.z], W: V3 = [d.cameraW.x, d.cameraW.y, d.cameraW.z];
    const wl = Math.hypot(...W);
    const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const [gx, gy, gz] = rp.grid;
    const st = half ? f16 : (x: number) => Math.max(x, 0);
    const slice = (u: number) => (u <= 0 ? 0 : rp.far * Math.pow(u, rp.spread));
    const ray = (u: number, v: number): V3 => {
        const nx = 2 * u - 1, ny = 1 - 2 * v;
        return [0, 1, 2].map((k) => nx * U[k]! + ny * V[k]! + W[k]!) as V3;
    };
    const noiseF = (p: V3, amount: number, scale: number) =>
        amount > 0 ? Math.max(0, 1 + amount * (2 * fbm(p.map((x, k) => x * scale + rp.wind[k]! * rp.time) as V3) - 1)) : 1;
    const idx = (x: number, y: number, z: number) => (z * gy + y) * gx + x;
    // scatter
    const media = new Float64Array(gx * gy * gz * 4);
    for (let z = 0; z < gz; z++) for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
        const r = ray((x + 0.5) / gx, (y + 0.5) / gy);
        const s = slice((z + 0.5) / gz) / wl;
        const p: V3 = [pos[0] + r[0] * s, pos[1] + r[1] * s, pos[2] + r[2] * s];
        let sT = 0;
        const sS: V3 = [0, 0, 0];
        if (rp.density > 0) {
            const dd = rp.density * Math.exp(Math.min(-rp.heightFalloff * (p[1] - rp.heightBase), 40)) * noiseF(p, rp.noise, rp.noiseScale);
            sT += dd;
            for (let k = 0; k < 3; k++) sS[k] += dd * rp.albedo[k]!;
        }
        for (const v of rp.volumes) {
            const l = p.map((c, k) => (c - v.center[k]!) / v.halfExtent[k]!);
            const rr = v.shape === "sphere" ? Math.hypot(...l) : Math.max(...l.map(Math.abs));
            let w = v.softness > 0 ? Math.min(Math.max((1 - rr) / v.softness, 0), 1) : rr <= 1 ? 1 : 0;
            w = w * w * (3 - 2 * w);
            if (w > 0) {
                const dd = v.density * w * noiseF(p, v.noise, v.noiseScale);
                sT += dd;
                for (let k = 0; k < 3; k++) sS[k] += dd * v.albedo[k]!;
            }
        }
        const o = idx(x, y, z) * 4;
        if (sT > 0) {
            const vn = Math.hypot(p[0] - pos[0], p[1] - pos[1], p[2] - pos[2]);
            const view: V3 = [(p[0] - pos[0]) / vn, (p[1] - pos[1]) / vn, (p[2] - pos[2]) / vn];
            const g = rp.anisotropy;
            const phase = (l: V3) => { const den = 1 + g * g - 2 * g * dot(view, l); return (0.0795774715459 * (1 - g * g)) / (den * Math.sqrt(den)); };
            const visv = vis === "RayTraced" && blocked(p, lights.L, Infinity, lights.boxes) ? 0 : 1;
            const Lin: V3 = [0, 1, 2].map((k) => rp.ambient[k]! + visv * lights.I * phase(lights.L)) as V3;
            if (lights.otherScale > 0) {
                for (const pl of lights.points) {
                    const tl: V3 = [pl.pos[0] - p[0], pl.pos[1] - p[1], pl.pos[2] - p[2]];
                    const d2 = Math.max(dot(tl, tl), 1e-10), dl = Math.sqrt(d2);
                    const l: V3 = [tl[0] / dl, tl[1] / dl, tl[2] / dl];
                    const pv = vis === "RayTraced" && blocked(p, l, dl, lights.boxes) ? 0 : 1;
                    for (let k = 0; k < 3; k++) Lin[k] += (pv * pl.I[k]! / d2) * lights.otherScale * phase(l);
                }
            }
            for (let k = 0; k < 3; k++) media[o + k] = st(sS[k]! * Lin[k]!);
        }
        media[o + 3] = st(sT);
    }
    // integrate
    const integ = new Float64Array(gx * gy * gz * 4);
    for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
        const ps = Math.hypot(...ray((x + 0.5) / gx, (y + 0.5) / gy)) / wl;
        const acc = [0, 0, 0];
        let T = 1, prev = 0;
        for (let z = 0; z < gz; z++) {
            const edge = slice((z + 1) / gz), t = (edge - prev) * ps;
            prev = edge;
            const o = idx(x, y, z) * 4, a = media[o + 3]!;
            const Ts = Math.exp(-a * t);
            for (let k = 0; k < 3; k++) acc[k] += T * (a > 1e-7 ? (media[o + k]! - media[o + k]! * Ts) / a : media[o + k]! * t);
            T *= Ts;
            for (let k = 0; k < 3; k++) integ[o + k] = st(acc[k]!);
            integ[o + 3] = st(T);
        }
    }
    // apply (fog output)
    const tri = (q: V3): number[] => {
        const hi = [gx - 1, gy - 1, gz - 1];
        const c = q.map((v, k) => Math.min(Math.max(v, 0), hi[k]!));
        const a = c.map(Math.floor), f = c.map((v, k) => v - a[k]!), b = a.map((v, k) => Math.min(v + 1, hi[k]!));
        const at = (x: number, y: number, z: number) => Array.from(integ.subarray(idx(x, y, z) * 4, idx(x, y, z) * 4 + 4));
        const l = (u: number[], v: number[], t: number) => u.map((x, k) => x + (v[k]! - x) * t);
        const x00 = l(at(a[0]!, a[1]!, a[2]!), at(b[0]!, a[1]!, a[2]!), f[0]!), x10 = l(at(a[0]!, b[1]!, a[2]!), at(b[0]!, b[1]!, a[2]!), f[0]!);
        const x01 = l(at(a[0]!, a[1]!, b[2]!), at(b[0]!, a[1]!, b[2]!), f[0]!), x11 = l(at(a[0]!, b[1]!, b[2]!), at(b[0]!, b[1]!, b[2]!), f[0]!);
        return l(l(x00, x10, f[1]!), l(x01, x11, f[1]!), f[2]!);
    };
    const out = new Float32Array(kSize * kSize * 4);
    for (let py = 0; py < kSize; py++) for (let px = 0; px < kSize; px++) {
        const i = py * kSize + px;
        const pw: V3 = [posW[i * 4]!, posW[i * 4 + 1]!, posW[i * 4 + 2]!];
        let fog = [0, 0, 0, 1];
        const u = (px + 0.5) / kSize, v = (py + 0.5) / kSize;
        let depth = -1;
        if (posW[i * 4 + 3] !== 0) depth = dot([pw[0] - pos[0], pw[1] - pos[1], pw[2] - pos[2]], W) / wl;
        else if (rp.skyFog) depth = rp.far;
        if (depth >= 0) {
            const s = depth <= 0 ? 0 : gz * Math.pow(Math.min(depth / rp.far, 1), 1 / rp.spread);
            const w = s - 1, xy: [number, number] = [u * gx - 0.5, v * gy - 0.5];
            if (w >= 0) fog = tri([xy[0], xy[1], w]);
            else {
                const v0 = tri([xy[0], xy[1], 0]), e = [0, 0, 0, 1];
                fog = e.map((x, k) => x + (v0[k]! - x) * (w + 1));
            }
        }
        out.set(fog, i * 4);
    }
    return out;
}

const kParity: RefParams = {
    density: 0.06, heightFalloff: 0.6, heightBase: 0.2, albedo: [0.9, 0.95, 1], anisotropy: 0.4, ambient: [0.03, 0.035, 0.05],
    noise: 0.5, noiseScale: 1.3, wind: [0.4, 0, 0.2], time: 1.5, far: 12, spread: 1.6, grid: [24, 24, 16], skyFog: true,
    volumes: [
        { shape: "sphere", center: [0.6, 0.7, 0.6], halfExtent: [0.9, 0.7, 0.9], density: 0.8, albedo: [1, 0.8, 0.6], softness: 0.4, noise: 0.6, noiseScale: 2.5 },
        { shape: "box", center: [-1.2, 0.4, 1.4], halfExtent: [1.0, 0.4, 0.6], density: 0.5, albedo: [0.6, 0.8, 1], softness: 0, noise: 0, noiseScale: 1 },
    ],
};

const passProps = (rp: RefParams, extra: Record<string, unknown>) => ({
    density: rp.density, heightFalloff: rp.heightFalloff, heightBase: rp.heightBase, albedo: rp.albedo, anisotropy: rp.anisotropy,
    ambient: rp.ambient, noise: rp.noise, noiseScale: rp.noiseScale, wind: rp.wind, time: rp.time, far: rp.far, spread: rp.spread,
    gridX: rp.grid[0], gridY: rp.grid[1], gridZ: rp.grid[2], skyFog: rp.skyFog, volumes: rp.volumes, lightIntensity: 1, ...extra,
});

gpuTest("FroxelFog.cpuParity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    const outs = ["F.fog", "F.dst", "L.color", "G.posW"];
    for (const vis of ["None", "RayTraced"] as const) {
        for (const precision of ["float", "half"] as const) {
            const g = fogGraph(device, scene, passProps(kParity, { visMode: vis, precision, temporal: false }), outs, "None");
            const [shot] = await renderShot(device, scene, g, kPoseA, 1, outs);
            const [fog, dst, color, posW] = shot!;
            const ref = cpuFog(scene, kParity, vis, precision === "half", posW!);
            const tol = precision === "half" ? 4e-3 : 2e-4;
            let maxErr = 0, bad = 0, dstErr = 0, sMean = 0, tMin = 1;
            for (let i = 0; i < kSize * kSize; i++) {
                for (let k = 0; k < 4; k++) {
                    const r = ref[i * 4 + k]!, e = Math.abs(fog![i * 4 + k]! - r) / Math.max(1, Math.abs(r));
                    maxErr = Math.max(maxErr, e);
                    if (!(e <= tol)) { bad++; break; }
                }
                for (let k = 0; k < 3; k++) dstErr = Math.max(dstErr, Math.abs(dst![i * 4 + k]! - (color![i * 4 + k]! * fog![i * 4 + 3]! + fog![i * 4 + k]!)));
                sMean += fog![i * 4]! / (kSize * kSize);
                tMin = Math.min(tMin, fog![i * 4 + 3]!);
            }
            log(`FroxelFog parity [${vis}, ${precision}]: fog (S, T) max rel err ${maxErr.toExponential(2)}, px above ${tol}: ${bad}/${kSize * kSize}; dst = color*T + S max err ${dstErr.toExponential(2)}; mean S.r ${sMean.toFixed(4)}, min T ${tMin.toFixed(3)}`);
            expectEq(bad <= 0.002 * kSize * kSize, true, `fog output = CPU reference (${vis}, ${precision})`);
            expectEq(dstErr < 1e-5, true, "dst = color * T + S");
            expectEq(sMean > 1e-3 && tMin < 0.9, true, "the fog is visible (non-trivial test)");
        }
    }
});

const kPoints: { pos: V3; I: V3 }[] = [{ pos: [0.2, 1.4, 1.0], I: [2, 1.6, 1.2] }, { pos: [-1.6, 0.6, -1.5], I: [0.5, 0.8, 1.5] }];
const kPointScenes = new WeakMap<Device, Scene>();

/** Floor + box A, a dim directional light (index 0) and the two point lights of kPoints. */
function pointLightScene(device: Device): Scene {
    let scene = kPointScenes.get(device);
    if (!scene) {
        const uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
        const quad = (c: V3[], n: V3) => ({
            vertices: c.map((p, i) => ({ position: new float3(...p), normal: new float3(...n), tangent: new float4(1, 0, 0, 1), texCrd: uv[i]! })),
            indices: new Uint32Array([0, 1, 2, 0, 2, 3]), materialID: 0,
        });
        const [[x0, y0, z0], [x1, y1, z1]] = kBoxA;
        const meshes = [
            quad([[-6, 0, 6], [6, 0, 6], [6, 0, -6], [-6, 0, -6]], [0, 1, 0]),
            quad([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0, 1, 0]),
            quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1]),
            quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0, 0, -1]),
            quad([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1, 0, 0]),
            quad([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1, 0, 0]),
        ];
        const lights = [{ type: LightType.Directional, dirW: new float3(...kLightDirW), intensity: new float3(0.5, 0.5, 0.5) },
            ...kPoints.map((l) => ({ type: LightType.Point, posW: new float3(...l.pos), intensity: new float3(...l.I) }))];
        scene = new Scene(device, meshes, [{ basic: { baseColor: new float4(0.7, 0.7, 0.7, 1), specular: new float4(0, 0.6, 0, 0) } }], lights);
        scene.setEnvMap(new EnvMap(device, { width: 1, height: 1, data: new Float32Array([0.02, 0.02, 0.03, 1]) } as never));
        kPointScenes.set(device, scene);
    }
    return scene;
}

gpuTest("FroxelFog.otherLightsParity", async ({ device }) => {
    const scene = pointLightScene(device);
    setPose(scene, kPoseA);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    const rp: RefParams = { ...kParity, volumes: kParity.volumes.slice(0, 1), ambient: [0, 0, 0] };
    const outs = ["F.fog", "G.posW"];
    for (const vis of ["None", "RayTraced"] as const) {
        const g = fogGraph(device, scene, passProps(rp, { visMode: vis, precision: "float", temporal: false, otherLights: 0.8 }), outs, "None");
        const [shot] = await renderShot(device, scene, g, kPoseA, 1, outs);
        const [fog, posW] = shot!;
        const ref = cpuFog(scene, rp, vis, false, posW!, { L: kL, I: 0.5, points: kPoints, otherScale: 0.8, boxes: [kBoxA] });
        const off = cpuFog(scene, rp, vis, false, posW!, { L: kL, I: 0.5, points: kPoints, otherScale: 0, boxes: [kBoxA] });
        let maxErr = 0, bad = 0, gain = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            for (let k = 0; k < 4; k++) {
                const r = ref[i * 4 + k]!, e = Math.abs(fog![i * 4 + k]! - r) / Math.max(1, Math.abs(r));
                maxErr = Math.max(maxErr, e);
                if (!(e <= 2e-4)) { bad++; break; }
            }
            gain += (ref[i * 4]! - off[i * 4]!) / (kSize * kSize);
        }
        log(`FroxelFog otherLights parity [${vis}]: max rel err ${maxErr.toExponential(2)}, px above 2e-4: ${bad}/${kSize * kSize}; mean in-scatter added by the point lights ${gain.toFixed(4)}`);
        expectEq(bad <= 0.002 * kSize * kSize, true, `otherLights fog = CPU reference (${vis})`);
        expectEq(gain > 1e-3, true, "the point lights add visible in-scatter (non-trivial test)");
    }
});

gpuTest("FroxelFog.shadowsCutTheFog", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    // Ground fog (dense near the floor, thin above the boxes) so the boxes' shadow volumes are a large share of the medium.
    const rp: RefParams = { ...kParity, volumes: [], noise: 0, heightFalloff: 2.5, heightBase: 0, density: 0.6, far: 8, ambient: [0, 0, 0], grid: [64, 64, 48] };
    const S: Record<string, Float32Array> = {};
    for (const vis of ["None", "RayTraced", "ShadowMap"]) {
        const g = fogGraph(device, scene, passProps(rp, { visMode: vis, temporal: false }), ["F.fog"]);
        S[vis] = (await renderShot(device, scene, g, kPoseA, 1, ["F.fog"]))[0]![0]!;
    }
    let sumN = 0, sumR = 0, sumS = 0, darker = 0, agree = 0, n = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        const a = S.None![i * 4]!, b = S.RayTraced![i * 4]!, c = S.ShadowMap![i * 4]!;
        sumN += a; sumR += b; sumS += c; n++;
        if (b < 0.97 * a) darker++;
        if (Math.abs(c - b) <= 0.05 * Math.max(b, 1e-3)) agree++;
    }
    log(`FroxelFog shadows: mean in-scatter None ${(sumN / n).toFixed(4)}, RayTraced ${(sumR / n).toFixed(4)}, ShadowMap ${(sumS / n).toFixed(4)}; px darkened by shadows ${darker}/${n}; ShadowMap within 5% of RayTraced ${agree}/${n}`);
    expectEq(sumR < 0.98 * sumN && sumS < 0.98 * sumN, true, "shadowed in-scattering is below unshadowed");
    expectEq(darker > 0.05 * n, true, "shadow shafts cut a visible share of the image");
    expectEq(agree > 0.8 * n, true, "ShadowMap visibility agrees with RayTraced on most pixels");
});

gpuTest("FroxelFog.identity", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    let ok = true;
    for (const props of [{ density: 0 }, { density: 0, volumes: [{ shape: "sphere", center: [0, 0.5, 0], halfExtent: 1, density: 0 }], noise: 0.8 }]) {
        const g = fogGraph(device, scene, { ...props, temporal: true }, ["F.dst", "L.color", "F.fog"]);
        const frames = await renderShot(device, scene, g, kPoseA, 3, ["F.dst", "L.color", "F.fog"], 0.06);
        for (const [dst, color, fog] of frames) {
            let fogOk = true;
            for (let i = 0; i < fog!.length; i += 4) if (fog![i] !== 0 || fog![i + 1] !== 0 || fog![i + 2] !== 0 || fog![i + 3] !== 1) fogOk = false;
            ok &&= bitwiseEqual(dst!, color!) && fogOk;
        }
    }
    log(`FroxelFog identity: density 0 (with and without zero-density volumes), 3 moving temporal frames each: dst = color bitwise ${ok}`);
    expectEq(ok, true, "density 0 = identity");
});

gpuTest("FroxelFog.temporal", async ({ device }) => {
    const scene = makeScene(device, kSky);
    await warmUp(device, scene);
    setPose(scene, kPoseX);
    scene.camera.resetHistory();
    const props = passProps({ ...kParity, grid: [48, 48, 32] }, { visMode: "ShadowMap" });
    const gT = fogGraph(device, scene, { ...props, temporal: true }, ["F.dst"]);
    const t = await renderShot(device, scene, gT, kPoseX, 4, ["F.dst"]);
    const f = gT.getPasses().find((p) => p.name === "F")!.pass as unknown as FroxelFog;
    const gS = fogGraph(device, scene, { ...props, temporal: false }, ["F.dst"]);
    const s = await renderShot(device, scene, gS, kPoseX, 3, ["F.dst"]);
    const frame0 = bitwiseEqual(t[0]![0]!, s[0]![0]!);
    const changes = [1, 2, 3].map((k) => !bitwiseEqual(t[k]![0]!, t[k - 1]![0]!));
    const stateless = bitwiseEqual(s[0]![0]!, s[1]![0]!) && bitwiseEqual(s[1]![0]!, s[2]![0]!);
    log(`FroxelFog temporal: frame 0 = non-temporal ${frame0}; frames 1-3 change ${changes.join(",")}; frameIndex ${f.frameIndex}; jitter(1..3) ${[1, 2, 3].map((i) => froxelJitter(i).map((x) => x.toFixed(3)).join("/")).join(" ")}; temporal off stateless ${stateless}`);
    expectEq(frame0, true, "first frame after a reset = non-temporal render");
    expectEq(changes.every((c) => c), true, "temporal accumulation changes the static shot over frames");
    expectEq(f.frameIndex, 3, "history counts frames since the reset");
    expectEq(stateless, true, "temporal off: no state");
});

gpuTest("FroxelFog.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device, kSky);
    const props = passProps({ ...kParity, grid: [48, 48, 32] }, { visMode: "ShadowMap" });
    const res = await coldVsConsecutive(device, scene, () => fogGraph(device, scene, { ...props, temporal: true }, ["F.dst"]), ["F.dst"], ["camera-reset", "prime", "explicit", "none"]);
    for (const r of res) log(`FroxelFog cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
    for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `cold = consecutive after ${r.method}`);
    expectEq(res.find((r) => r.method === "none")!.differs[0], true, "without a cut the leaked history changes X (control)");
    const off = await coldVsConsecutive(device, scene, () => fogGraph(device, scene, { ...props, temporal: false }, ["F.dst"]), ["F.dst"], ["none"]);
    log(`FroxelFog cold vs consecutive, temporal off, no cut: ${off[0]!.equalFrames}/${off[0]!.frames} bitwise`);
    expectEq(off[0]!.equalFrames, off[0]!.frames, "temporal off: stateless even without a cut");
});
