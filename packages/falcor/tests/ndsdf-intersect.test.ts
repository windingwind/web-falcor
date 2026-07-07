/**
 * Faithful CPU port of NDSDFGrid::intersectSDF (VoxelSphereTracing +
 * NumericDiscontinuous config, double math) over the NDSDFGrid.pyscene
 * cheese + camera. Pins the ALGORITHM footprint oracle used by the
 * feature-ndsdf GPU test (tests/oracle/assets/ndsdf-cpu-footprint.bin):
 * the web GPU render matches this footprint PIXEL-EXACTLY (iter 78
 * adjudication: cpuOnly=0 vs web; native is the outlier with 24568
 * spurious dilated hits from its local offset-fetch corruption).
 * Camera: pyscene default focalLength 21 (the scene sets none!).
 */
import { expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { NDSDFGrid } from "../src/Scene/SDFs/NDSDFGrid.js";
import { Camera } from "../src/Scene/Camera/Camera.js";
import { float3 } from "../src/Utils/Math/Vector.js";

const W = 640, H = 360;
const kMinStepSize = 0.001, kMaxSteps = 512, kSolverMaxSteps = 256, FLT_EPS = 1.1920929e-7;

it("intersectSDF CPU port reproduces the pinned footprint oracle", () => {
    const grid = new NDSDFGrid(2.5);
    grid.generateCheeseValues(128, 0);
    const c = grid.coarsestLODGridWidth, nLods = grid.lodCount, maxLOD = nLods - 1;
    const coarseNorm = grid.coarsestLODNormalizationFactor, narrowBand = grid.narrowBandThickness;

    // Corner load: snorm8 decode; OOB -> 0 (native robust access).
    const corner = (lod: number, x: number, y: number, z: number): number => {
        const wv = 1 + (c << lod);
        if (x < 0 || y < 0 || z < 0 || x >= wv || y >= wv || z >= wv) return 0;
        return Math.max(grid.values[lod]![x + wv * (y + wv * z)]! / 127, -1);
    };
    // Corner fetch for a voxel: v0xx = [(0,0,0),(0,0,1),(0,1,0),(0,1,1)], v1xx x+1.
    const loadCorners = (lod: number, vx: number, vy: number, vz: number): [number[], number[]] => [
        [corner(lod, vx, vy, vz), corner(lod, vx, vy, vz + 1), corner(lod, vx, vy + 1, vz), corner(lod, vx, vy + 1, vz + 1)],
        [corner(lod, vx + 1, vy, vz), corner(lod, vx + 1, vy, vz + 1), corner(lod, vx + 1, vy + 1, vz), corner(lod, vx + 1, vy + 1, vz + 1)],
    ];
    const trilin = (v0: number[], v1: number[], u: number, v: number, w: number): number => {
        const cX = [0, 1, 2, 3].map((i) => v0[i]! + (v1[i]! - v0[i]!) * u);
        const cY0 = cX[0]! + (cX[2]! - cX[0]!) * v, cY1 = cX[1]! + (cX[3]! - cX[1]!) * v;
        return cY0 + (cY1 - cY0) * w;
    };
    const containsSurface = (v0: number[], v1: number[]): boolean => {
        const all = [...v0, ...v1];
        return all.some((x) => x <= 0) && all.some((x) => x >= 0);
    };
    // hwSample: linear filter, clamp sampler == lattice pos p*W clamped to [0, W].
    const hwSample = (lod: number, px: number, py: number, pz: number): number => {
        const gw = c << lod;
        const cl = (q: number) => Math.min(Math.max(q * gw, 0), gw);
        const qx = cl(px), qy = cl(py), qz = cl(pz);
        const x0 = Math.min(Math.floor(qx), gw), y0 = Math.min(Math.floor(qy), gw), z0 = Math.min(Math.floor(qz), gw);
        const [v0, v1] = loadCorners(lod, x0, y0, z0);
        return trilin(v0, v1, qx - x0, qy - y0, qz - z0);
    };
    const swSample = (lod: number, px: number, py: number, pz: number): number => {
        const gw = c << lod;
        const pcx = Math.min(Math.max(px, 0), 1) * gw, pcy = Math.min(Math.max(py, 0), 1) * gw, pcz = Math.min(Math.max(pz, 0), 1) * gw;
        const vx = Math.floor(pcx), vy = Math.floor(pcy), vz = Math.floor(pcz);
        const [v0, v1] = loadCorners(lod, vx, vy, vz);
        return trilin(v0, v1, pcx - vx, pcy - vy, pcz - vz);
    };
    const deltaFinerLOD = (d: number): number => Math.floor(Math.log2(1 / Math.abs(d))) + 1;
    const solver = (u0: number, v0c: number, w0: number, d: number[], v0: number[], v1: number[], tMax: number): { hit: boolean; t: number } => {
        let px = u0, py = v0c, pz = w0;
        let lastD = 1, currD = trilin(v0, v1, px, py, pz), clampedD = Math.max(currD, 0.0001), t = 0;
        if (currD <= 0) return { hit: true, t };
        let it2 = 0;
        for (; it2 < kSolverMaxSteps; it2++) {
            t += clampedD;
            if (t > tMax) return { hit: false, t };
            px = Math.min(Math.max(u0 + t * d[0]!, 0), 1); py = Math.min(Math.max(v0c + t * d[1]!, 0), 1); pz = Math.min(Math.max(w0 + t * d[2]!, 0), 1);
            lastD = currD; currD = trilin(v0, v1, px, py, pz); clampedD = Math.max(currD, 0.0001);
            if (currD <= 0) break;
        }
        t += (currD * clampedD) / (currD - lastD);
        return { hit: it2 < kSolverMaxSteps, t };
    };

    const intersect = (ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, tMin: number, tMax: number): boolean => {
        const lox = ox + 0.5, loy = oy + 0.5, loz = oz + 0.5;
        const dirLength = Math.hypot(dx, dy, dz), inv = 1 / dirLength;
        const rdx = dx * inv, rdy = dy * inv, rdz = dz * inv;
        const eps = (q: number) => (Math.abs(q) < FLT_EPS ? (q < 0 ? -FLT_EPS : FLT_EPS) : q);
        const d = [eps(rdx), eps(rdy), eps(rdz)];
        // intersectRayAABB([0,1]^3)
        let near = 0, far = Infinity;
        for (const [o, dd] of [[lox, rdx], [loy, rdy], [loz, rdz]] as const) {
            const i2 = 1 / dd;
            let ta = (0 - o) * i2, tb = (1 - o) * i2;
            if (ta > tb) [ta, tb] = [tb, ta];
            near = Math.max(near, ta); far = Math.min(far, tb);
        }
        if (near > far) return false;
        let t = Math.max(tMin * dirLength, near);
        const tMaxLocal = Math.min(tMax * dirLength, far);
        if (tMaxLocal < t) return false;

        let lod = maxLOD, normF = coarseNorm / (1 << lod);
        let plx = lox + t * rdx, ply = loy + t * rdy, plz = loz + t * rdz;
        let currD = swSample(lod, plx, ply, plz);
        let clampedD = Math.max(currD, kMinStepSize);
        if (currD <= 0) return true;
        let prevT = t, currH = clampedD * normF, nextH = 0;
        let steps = 0, resolved = false;
        for (; steps < kMaxSteps; steps++) {
            prevT = t; t += currH;
            if (t > tMaxLocal) return false;
            plx = lox + t * rdx; ply = loy + t * rdy; plz = loz + t * rdz;
            currD = hwSample(lod, plx, ply, plz);
            if (Math.abs(currD) >= 1) {
                lod = Math.max(lod - 1, 0); normF = coarseNorm / (1 << lod);
                currD = hwSample(lod, plx, ply, plz);
                if (currD <= 0) {
                    if (lod === maxLOD) { nextH = currD * normF; resolved = true; break; }
                    lod = Math.min(lod + deltaFinerLOD(Math.max(currD, 0) || FLT_EPS), maxLOD); normF = coarseNorm / (1 << lod); t = prevT;
                }
            } else {
                if (currD < 1) { lod = Math.min(lod + deltaFinerLOD(currD), maxLOD); normF = coarseNorm / (1 << lod); }
                const gw = c << lod;
                const pcx = Math.min(Math.max(plx, 0), 1) * gw, pcy = Math.min(Math.max(ply, 0), 1) * gw, pcz = Math.min(Math.max(plz, 0), 1) * gw;
                const vx = Math.floor(pcx), vy = Math.floor(pcy), vz = Math.floor(pcz);
                const ux = pcx - vx, uy = pcy - vy, uz = pcz - vz;
                const [v0, v1] = loadCorners(lod, vx, vy, vz);
                currD = trilin(v0, v1, ux, uy, uz);
                if (currD <= 0) {
                    if (lod === maxLOD) { nextH = currD * normF; resolved = true; break; }
                    lod = Math.min(lod + deltaFinerLOD(Math.max(currD, 0) || FLT_EPS), maxLOD); normF = coarseNorm / (1 << lod); t = prevT;
                } else if (containsSurface(v0, v1)) {
                    if (lod === maxLOD) {
                        const tlx = ((d[0]! > 0 ? 1 : 0) - ux) / d[0]!, tly = ((d[1]! > 0 ? 1 : 0) - uy) / d[1]!, tlz = ((d[2]! > 0 ? 1 : 0) - uz) / d[2]!;
                        const tLocalMax = Math.min((tMaxLocal - t) * gw, tlx, tly, tlz);
                        const s = solver(ux, uy, uz, d, v0.map((q) => q / narrowBand), v1.map((q) => q / narrowBand), tLocalMax);
                        if (s.hit) { return true; }
                        currH = Math.max(tLocalMax / gw, currD * normF);
                        continue;
                    }
                    lod = Math.min(lod + deltaFinerLOD(Math.max(currD, 0) || FLT_EPS), maxLOD); normF = coarseNorm / (1 << lod); t = prevT;
                }
            }
            currH = currD * normF;
        }
        return resolved && steps < kMaxSteps;
    };

    const cam = new Camera();
    cam.setPosition(new float3(1, 1, 1));
    cam.setTarget(new float3(0, 0, 0));
    cam.setUpVector(new float3(0, 1, 0));
    cam.setFocalLength(21);
    cam.setAspectRatio(W / H);
    const cd = cam.getData();

    // Regenerate with: replace the compare below with a writeFileSync of `mask`.
    const committed = new Uint8Array(readFileSync(resolve(__dirname, "../../../tests/oracle/assets/ndsdf-cpu-footprint.bin")));
    const mask = new Uint8Array(Math.ceil((W * H) / 8));
    let hits = 0;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const px = (x + 0.5) / W, py = (y + 0.5) / H;
            const nx = 2 * px - 1, ny = -2 * py + 1;
            const dx = nx * cd.cameraU.x + ny * cd.cameraV.x + cd.cameraW.x;
            const dy = nx * cd.cameraU.y + ny * cd.cameraV.y + cd.cameraW.y;
            const dz = nx * cd.cameraU.z + ny * cd.cameraV.z + cd.cameraW.z;
            const len = Math.hypot(dx, dy, dz);
            if (intersect(1, 1, 1, dx / len, dy / len, dz / len, 0.1, 1000)) {
                const idx = y * W + x;
                mask[idx >> 3] = mask[idx >> 3]! | (1 << (idx & 7));
                hits++;
            }
        }
    }
    expect(hits).toBe(34356);
    expect(Buffer.from(mask).equals(Buffer.from(committed))).toBe(true);
}, 600000);
