/**
 * Ripples (wave-equation height field on a water plane).
 *  1. Simulation parity: the GPU field after 5 frames x 2 substeps with rain drops, an oscillating emitter and a mover
 *     equals a CPU simulation of the same steps; the applied normals equal the CPU normals from the GPU field, and
 *     pixels off the plane copy normW bit for bit.
 *  2. Object contact: with contactIds set to one box (StyleAovs.ids), the field after the first injecting step is
 *     non-zero exactly in the cells where that box's visible pixels touch the plane.
 *  3. Identity: strength 0 copies normW bit for bit on every frame of an active simulation; without drivers too.
 *  4. Cold vs consecutive: shot X on a fresh graph = shot X after shots A and B on one graph with a camera reset, a
 *     prime render or resetGraphHistory in between (bit for bit); without a cut X differs (control).
 */

import type { Device, RenderGraph, Scene } from "@web-falcor/falcor";
import { ripplesBasis, dropsAtStep, type Ripples } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kBoxA, kSize, log, makeScene, read, type PassDecl } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, kPoseA, setPose, warmUp } from "./fx-helpers.js";

function pcg(v: number): number {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}
const rnd = (a: number, b: number, salt: number, seed: number) =>
    (pcg((a ^ pcg((Math.imul(b, 9781) + pcg((Math.imul(seed, 6271) + salt) >>> 0)) >>> 0)) >>> 0) >>> 8) / 16777216;
const bump = (d: number, r: number) => (d < r ? 0.5 * (1 + Math.cos((Math.PI * d) / r)) : 0);
const sat = (x: number) => Math.min(Math.max(x, 0), 1);

interface SimParams {
    centre: number[]; size: number; resolution: number; fps: number; substeps: number; damping: number; sponge: number; seed: number;
    dropsPerStep: number; dropRadius: number; dropStrength: number;
    emitters: { position: number[]; radius: number; strength: number; period: number; phase: number }[];
    movers: { from: number[]; to: number[]; start: number; end: number; radius: number; strength: number }[];
}

/** CPU reference of `steps` wave steps (y-up: planar t1/t2 from ripplesBasis). Returns h of the last step. */
function simulateCpu(p: SimParams, steps: number): Float64Array {
    const N = p.resolution, cs = p.size / N, dt = 1 / (p.fps * p.substeps);
    const { t1, t2 } = ripplesBasis([0, 1, 0]);
    const planar = (v: number[]) => [v[0]! * t1[0]! + v[1]! * t1[1]! + v[2]! * t1[2]!, v[0]! * t2[0]! + v[1]! * t2[1]! + v[2]! * t2[2]!];
    const C = planar(p.centre);
    const toCell = (q: number[]) => [(q[0]! - C[0]!) / cs + 0.5 * N, (q[1]! - C[1]!) / cs + 0.5 * N];
    let h = new Float64Array(N * N), hp = new Float64Array(N * N);
    for (let s = 0; s < steps; s++) {
        const nh = new Float64Array(N * N);
        const t = s * dt;
        const drops = Array.from({ length: dropsAtStep(p.dropsPerStep, s) }, (_, k) => [rnd(s, k, 1, p.seed) * N, rnd(s, k, 2, p.seed) * N]);
        const movers = p.movers.filter((m) => t >= m.start && t <= m.end && m.end > m.start).map((m) => {
            const a = planar(m.from), b = planar(m.to), u = (t - m.start) / (m.end - m.start);
            return { c: toCell([a[0]! + (b[0]! - a[0]!) * u, a[1]! + (b[1]! - a[1]!) * u]), r: m.radius / cs, k: m.strength * (Math.hypot(b[0]! - a[0]!, b[1]! - a[1]!) / (m.end - m.start)) * dt };
        });
        for (let y = 0; y < N; y++) {
            for (let x = 0; x < N; x++) {
                const g = (xx: number, yy: number) => h[Math.min(Math.max(yy, 0), N - 1) * N + Math.min(Math.max(xx, 0), N - 1)]!;
                let v = ((g(x, y - 1) + g(x, y + 1) + g(x + 1, y) + g(x - 1, y)) * 0.5 - hp[y * N + x]!) * p.damping;
                const edge = Math.min(x, y, N - 1 - x, N - 1 - y);
                if (p.sponge > 0) v *= 0.9 + 0.1 * sat(edge / p.sponge);
                const cx = x + 0.5, cy = y + 0.5;
                for (const d of drops) v -= p.dropStrength * bump(Math.hypot(cx - d[0]!, cy - d[1]!), p.dropRadius / cs);
                for (const e of p.emitters) {
                    const c = toCell(planar(e.position));
                    v += e.strength * Math.sin(2 * Math.PI * (t / e.period + e.phase)) * bump(Math.hypot(cx - c[0]!, cy - c[1]!), e.radius / cs);
                }
                for (const m of movers) v -= m.k * bump(Math.hypot(cx - m.c[0]!, cy - m.c[1]!), m.r);
                nh[y * N + x] = v;
            }
        }
        hp = h;
        h = nh;
    }
    return h;
}

const kSim: SimParams = {
    centre: [0, 0, 0], size: 8, resolution: 64, fps: 30, substeps: 2, damping: 0.985, sponge: 4, seed: 4,
    dropsPerStep: 0.7, dropRadius: 0.25, dropStrength: 0.01,
    emitters: [{ position: [1.5, 0, 1.0], radius: 0.35, strength: 0.004, period: 0.5, phase: 0.1 }],
    movers: [{ from: [-3, 0, -2], to: [2, 0, 2.5], start: 0, end: 0.2, radius: 0.3, strength: 0.05 }],
};

function ripGraph(device: Device, scene: Scene, props: Record<string, unknown>, outputs: string[], extra: PassDecl[] = [], extraEdges: [string, string][] = []): RenderGraph {
    return buildGraph(device, scene, [...extra, ["Ripples", "P", props]], [["G.posW", "P.posW"], ["G.normW", "P.normW"], ...extraEdges], outputs, kSize, kSize);
}

gpuTest("Ripples.simulationAndApplyParity", async ({ device }) => {
    const scene = makeScene(device);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    const props = { ...kSim, strength: 2, planeTolerance: 0.01, edgeFade: 0.05 };
    const g = ripGraph(device, scene, props, ["P.normWOut", "P.height", "G.normW"]);
    const frames = 5;
    for (let f = 0; f < frames; f++) g.execute(device.renderContext);
    const pass = g.getPass("P") as unknown as Ripples;
    const field = await read(device, pass.fieldTexture!);
    const ref = simulateCpu(kSim, frames * kSim.substeps);
    const N = kSim.resolution;
    let maxErr = 0, maxH = 0;
    for (let i = 0; i < N * N; i++) { maxErr = Math.max(maxErr, Math.abs(field[i * 4]! - ref[i]!)); maxH = Math.max(maxH, Math.abs(ref[i]!)); }
    // apply: CPU normals from the GPU field
    const out = await read(device, g.getOutput("P.normWOut")!), nw = await read(device, g.getOutput("G.normW")!), pos = await read(device, g.getOutput("G.posW")!);
    const cs = kSim.size / N;
    const { t1, t2 } = ripplesBasis([0, 1, 0]);
    const fieldAt = (x: number, y: number) => field[(Math.min(Math.max(y, 0), N - 1) * N + Math.min(Math.max(x, 0), N - 1)) * 4]!;
    const hAt = (cx: number, cy: number) => {
        const gx = cx - 0.5, gy = cy - 0.5, ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
        const a = fieldAt(ix, iy), b = fieldAt(ix + 1, iy), c = fieldAt(ix, iy + 1), d = fieldAt(ix + 1, iy + 1);
        return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
    };
    let nErr = 0, water = 0, offCopy = 0, offPx = 0, bent = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        if (pos[i * 4 + 3] === 0) continue;
        const P = [pos[i * 4]!, pos[i * 4 + 1]!, pos[i * 4 + 2]!];
        const n = [nw[i * 4]!, nw[i * 4 + 1]!, nw[i * 4 + 2]!];
        const nl = Math.hypot(n[0]!, n[1]!, n[2]!);
        const onPlane = n[1]! / nl >= 0.9 && Math.abs(P[1]!) <= 0.01;
        const q = [P[0]! * t1[0]! + P[2]! * t1[2]!, P[0]! * t2[0]! + P[2]! * t2[2]!];
        const cx = q[0]! / cs + 0.5 * N, cy = q[1]! / cs + 0.5 * N;
        const edge = Math.min(cx, cy, N - cx, N - cy);
        const wt = onPlane && edge > 0 ? sat(edge / (0.05 * N)) : 0;
        if (wt === 0) {
            offPx++;
            if (bitwiseEqual(out.subarray(i * 4, i * 4 + 4), nw.subarray(i * 4, i * 4 + 4))) offCopy++;
            continue;
        }
        water++;
        const gx = (hAt(cx + 1, cy) - hAt(cx - 1, cy)) / (2 * cs), gy = (hAt(cx, cy + 1) - hAt(cx, cy - 1)) / (2 * cs);
        const m = [0, 1, 2].map((c) => n[c]! / nl - 2 * wt * (gx * t1[c]! + gy * t2[c]!));
        const ml = Math.hypot(m[0]!, m[1]!, m[2]!);
        for (let c = 0; c < 3; c++) nErr = Math.max(nErr, Math.abs(out[i * 4 + c]! - m[c]! / ml));
        if (Math.abs(out[i * 4 + 1]! - n[1]! / nl) > 1e-4) bent++;
    }
    log(`Ripples parity: field after ${frames * kSim.substeps} steps max |h| ${maxH.toExponential(3)}, GPU vs CPU max abs err ${maxErr.toExponential(2)} (rel ${(maxErr / maxH).toExponential(2)}); water px ${water} (bent ${bent}), normal max err ${nErr.toExponential(2)}; off-water px ${offPx}, copied bitwise ${offCopy}`);
    expectEq(maxH > 1e-3, true, "the drivers moved the water");
    expectEq(maxErr <= 1e-4 * maxH, true, "GPU field = CPU simulation (rel err <= 1e-4)");
    expectEq(water > 1000 && bent > 500, true, "the floor gets ripple normals");
    expectEq(nErr < 1e-4, true, "applied normal = CPU normal from the field");
    expectEq(offCopy, offPx, "pixels off the water copy normW bit for bit");
});

gpuTest("Ripples.objectContact", async ({ device }) => {
    const scene = makeScene(device);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    // Find box A's instance id from StyleAovs.ids.
    const probe = buildGraph(device, scene, [["StyleAovs", "A", {}]], [["G.vbuffer", "A.vbuffer"]], ["A.ids"], kSize, kSize);
    probe.execute(device.renderContext);
    const ids = await read(device, probe.getOutput("A.ids")!), pos0 = await read(device, probe.getOutput("G.posW")!);
    const counts = new Map<number, number>();
    for (let i = 0; i < kSize * kSize; i++) {
        const x = pos0[i * 4]!, y = pos0[i * 4 + 1]!, z = pos0[i * 4 + 2]!;
        if (pos0[i * 4 + 3] === 0 || y < 0.02) continue;
        if (x > kBoxA[0][0] - 1e-3 && x < kBoxA[1][0] + 1e-3 && z > kBoxA[0][2] - 1e-3 && z < kBoxA[1][2] + 1e-3) counts.set(ids[i * 4]!, (counts.get(ids[i * 4]!) ?? 0) + 1);
    }
    const boxId = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]![0];
    const N = 64, size = 8, cs = size / N, dist = 0.08;
    const props = { centre: [0, 0, 0], size, resolution: N, substeps: 1, dropsPerStep: 0, contactIds: [boxId], contactDistance: dist, contactStrength: 0.01, contactPeriod: 1, damping: 0.985, sponge: 0 };
    const g = ripGraph(device, scene, props, ["P.normWOut", "A.ids"], [["StyleAovs", "A", {}]], [["G.vbuffer", "A.vbuffer"], ["A.ids", "P.ids"]]);
    g.execute(device.renderContext); // step 0: t = 0, sin = 0
    g.execute(device.renderContext); // step 1: injection only (h was 0)
    const field = await read(device, (g.getPass("P") as unknown as Ripples).fieldTexture!);
    const pos = await read(device, g.getOutput("G.posW")!), id2 = await read(device, g.getOutput("A.ids")!);
    const { t1, t2 } = ripplesBasis([0, 1, 0]);
    const marked = new Set<number>();
    for (let i = 0; i < kSize * kSize; i++) {
        if (pos[i * 4 + 3] === 0 || Math.round(id2[i * 4]!) !== boxId || Math.abs(pos[i * 4 + 1]!) > dist) continue;
        const cx = Math.floor((pos[i * 4]! * t1[0]! + pos[i * 4 + 2]! * t1[2]!) / cs + 0.5 * N), cy = Math.floor((pos[i * 4]! * t2[0]! + pos[i * 4 + 2]! * t2[2]!) / cs + 0.5 * N);
        if (cx >= 0 && cy >= 0 && cx < N && cy < N) marked.add(cy * N + cx);
    }
    const expectH = 0.01 * Math.sin((2 * Math.PI) / 30);
    let nonzero = 0, match = 0, valErr = 0;
    for (let c = 0; c < N * N; c++) {
        const h = field[c * 4]!;
        if (h !== 0) nonzero++;
        if ((h !== 0) === marked.has(c)) match++;
        if (marked.has(c)) valErr = Math.max(valErr, Math.abs(h - expectH));
    }
    log(`Ripples contact: box instance id ${boxId}, CPU-marked cells ${marked.size}, non-zero cells ${nonzero}, cell agreement ${match}/${N * N}, value err ${valErr.toExponential(2)}`);
    expectEq(marked.size > 3, true, "the box touches the plane in view");
    expectEq(match, N * N, "non-zero exactly in the contact cells");
    expectEq(valErr < 1e-6, true, "injection = contactStrength x sin(2 pi t / period)");
});

gpuTest("Ripples.identity", async ({ device }) => {
    const scene = makeScene(device);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    const res: string[] = [];
    let ok = true;
    for (const [label, props] of [["strength 0", { ...kSim, strength: 0 }], ["no drivers", { ...kSim, dropsPerStep: 0, emitters: [], movers: [] }]] as [string, Record<string, unknown>][]) {
        const g = ripGraph(device, scene, props, ["P.normWOut", "G.normW"]);
        let eq = 0;
        for (let f = 0; f < 3; f++) {
            g.execute(device.renderContext);
            if (bitwiseEqual(await read(device, g.getOutput("P.normWOut")!), await read(device, g.getOutput("G.normW")!))) eq++;
        }
        res.push(`${label} ${eq}/3`);
        ok &&= eq === 3;
    }
    log(`Ripples identity (normWOut = normW bitwise): ${res.join(", ")}`);
    expectEq(ok, true, "identity");
});

gpuTest("Ripples.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device);
    const outs = ["P.normWOut", "P.height"];
    const props = { ...kSim, resolution: 96, strength: 2, planeTolerance: 0.01 };
    const res = await coldVsConsecutive(device, scene, () => ripGraph(device, scene, props, outs), outs, ["camera-reset", "prime", "explicit", "none"]);
    for (const r of res) log(`Ripples cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
    for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `cold = consecutive after ${r.method}`);
    expectEq(res.find((r) => r.method === "none")!.differs.some((d) => d), true, "without a cut the field state changes X (control)");
});
