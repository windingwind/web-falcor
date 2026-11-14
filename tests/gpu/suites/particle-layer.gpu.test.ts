/**
 * ParticleLayer:
 *  - simulation parity: one GPU step from a read-back state = the CPU step (same formulas), splat parity: the GPU layer
 *    (colour, depth) = a CPU splat of the GPU particle state;
 *  - depth-buffer collision keeps falling particles above the visible floor (and without collision they fall through);
 *  - identity: intensity 0 gives an empty layer that DepthComposite passes through bit for bit;
 *  - cold vs consecutive: the particle state is re-seeded per shot, so shot X is bit-identical on a fresh graph and
 *    after shots A and B on one graph, for every cut signal; without a cut X differs.
 */

import { Buffer } from "@web-falcor/falcor";
import { type ParticleLayer } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, camInfo, kSize, log, makeScene, read, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, coverage, setPose, kPoseA } from "./fx-helpers.js";

const kFar = 3.402823466e38;

function pcg(v: number): number {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}
const rnd = (seed: number, i: number, gen: number, salt: number) =>
    (pcg((i ^ pcg((Math.imul(gen, 1013) + Math.imul(salt, 7919) + pcg(seed)) >>> 0)) >>> 0) >>> 8) / 16777216;
const rnd3 = (seed: number, i: number, gen: number, salt: number): V3 => [rnd(seed, i, gen, salt), rnd(seed, i, gen, salt + 1), rnd(seed, i, gen, salt + 2)];

interface P { count: number; seed: number; boundsMin: V3; boundsMax: V3; dt: number; lifetime: number; gravity: V3; wind: V3; drag: number; turbulence: number; turbulenceScale: number; size: number; maxRadiusPx: number; colour: V3; colourJitter: number; intensity: number; twinkle: number; twinkleRate: number; collide: boolean; restitution: number; thickness: number }

const kBase: P = {
    count: 1024, seed: 3, boundsMin: [-1.5, 0.05, -1.5], boundsMax: [1.5, 1.8, 1.5], dt: 1 / 30, lifetime: 3, gravity: [0, -0.3, 0], wind: [0.2, 0, 0],
    drag: 0.5, turbulence: 0.4, turbulenceScale: 2, size: 0.02, maxRadiusPx: 5, colour: [1, 0.8, 0.5], colourJitter: 0.3, intensity: 1.5, twinkle: 0.5,
    twinkleRate: 1, collide: false, restitution: 0.3, thickness: 0.05,
};

async function readState(device: Parameters<typeof buildGraph>[0], b: Buffer): Promise<Float32Array> {
    const raw = await device.renderContext.readBuffer(b);
    return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice();
}

/** CPU step without collision (collide = false), mirroring ParticleLayer.cs.slang `simulate`. */
function stepRef(p: P, s: Float32Array, time: number): Float32Array {
    const out = new Float32Array(s.length);
    for (let i = 0; i < p.count; i++) {
        const pos = [s[i * 8]!, s[i * 8 + 1]!, s[i * 8 + 2]!];
        let v = [s[i * 8 + 4]!, s[i * 8 + 5]!, s[i * 8 + 6]!];
        const age = s[i * 8 + 3]! + p.dt, gen = s[i * 8 + 7]!;
        const ph = rnd3(p.seed, i, gen, 9).map((x) => x * 6.2831853);
        const f = p.turbulenceScale;
        const noise = [Math.sin(f * pos[1]! + 1.7 * time + ph[0]!), Math.sin(f * pos[2]! + 1.3 * time + ph[1]!), Math.sin(f * pos[0]! + 1.1 * time + ph[2]!)];
        v = v.map((x, c) => x + (p.gravity[c]! + p.turbulence * noise[c]!) * p.dt);
        const dr = Math.min(Math.max(p.drag * p.dt, 0), 1);
        v = v.map((x, c) => x + (p.wind[c]! - x) * dr);
        const pn = pos.map((x, c) => x + v[c]! * p.dt);
        const ext = [0, 1, 2].map((c) => (p.boundsMax[c]! - p.boundsMin[c]!) * 0.05 + 1e-4);
        const outside = pn.some((x, c) => x < p.boundsMin[c]! - ext[c]! || x > p.boundsMax[c]! + ext[c]!);
        const life = p.lifetime * (0.5 + rnd(p.seed, i, gen, 6));
        if (age > life || outside) {
            const g1 = gen + 1;
            const sp = rnd3(p.seed, i, g1, 0).map((r, c) => p.boundsMin[c]! + (p.boundsMax[c]! - p.boundsMin[c]!) * r);
            const sv = rnd3(p.seed, i, g1, 3).map((r, c) => p.wind[c]! + (r - 0.5) * 0.1 * p.turbulence);
            out.set([...sp, 0, ...sv, g1], i * 8);
        } else {
            out.set([...pn, age, ...v, gen], i * 8);
        }
    }
    return out;
}

/** CPU splat of a particle state, mirroring `splat` + `resolve` (fixed point 1/4096). */
function splatRef(p: P, s: Float32Array, time: number, scene: ReturnType<typeof makeScene>, pos: Float32Array) {
    const cam = camInfo(scene);
    const vp = scene.camera.getData().viewProjMatNoJitter;
    const ppu = (0.5 * kSize) / Math.tan(0.5 * scene.camera.getFovY());
    const acc = new Float64Array(kSize * kSize * 4), dmin = new Float64Array(kSize * kSize).fill(Infinity);
    const viewDepth = (x: ArrayLike<number>) => (x[0]! - cam.pos[0]) * cam.fwd[0] + (x[1]! - cam.pos[1]) * cam.fwd[1] + (x[2]! - cam.pos[2]) * cam.fwd[2];
    const sceneDepth = (k: number) => (pos[k * 4 + 3] !== 0 ? viewDepth(pos.subarray(k * 4, k * 4 + 3)) : kFar);
    for (let i = 0; i < p.count; i++) {
        const x = s.subarray(i * 8, i * 8 + 3), gen = s[i * 8 + 7]!;
        const d = viewDepth(x);
        const h = [0, 1, 2, 3].map((r) => vp.get(r, 0) * x[0]! + vp.get(r, 1) * x[1]! + vp.get(r, 2) * x[2]! + vp.get(r, 3));
        if (d <= 0 || h[3]! <= 0) continue;
        const cx = (h[0]! / h[3]! * 0.5 + 0.5) * kSize, cy = (-h[1]! / h[3]! * 0.5 + 0.5) * kSize;
        const R = Math.min(Math.max((p.size * ppu) / d, 0.75), p.maxRadiusPx);
        const jit = rnd3(p.seed, i, gen, 12).map((r) => (r - 0.5) * 2 * p.colourJitter);
        const col = p.colour.map((c, k) => Math.max(c * (1 + jit[k]!), 0));
        const tw = 1 - p.twinkle * 0.5 * (1 + Math.sin(6.2831853 * (time * p.twinkleRate + rnd(p.seed, i, gen, 15))));
        const bright = p.intensity * tw;
        for (let py = Math.max(Math.floor(cy - R), 0); py <= Math.min(Math.ceil(cy + R), kSize - 1); py++) {
            for (let px = Math.max(Math.floor(cx - R), 0); px <= Math.min(Math.ceil(cx + R), kSize - 1); px++) {
                const ox = px + 0.5 - cx, oy = py + 0.5 - cy;
                const r2 = (ox * ox + oy * oy) / (R * R);
                if (r2 >= 1) continue;
                const k = py * kSize + px;
                if (d >= sceneDepth(k)) continue;
                const w = (1 - r2) * (1 - r2);
                for (let c = 0; c < 3; c++) acc[k * 4 + c] += Math.floor(col[c]! * w * bright * 4096 + 0.5);
                acc[k * 4 + 3] += Math.floor(w * 4096 + 0.5);
                dmin[k] = Math.min(dmin[k]!, d);
            }
        }
    }
    return { acc, dmin };
}

const plGraph = (device: Parameters<typeof buildGraph>[0], scene: Parameters<typeof buildGraph>[1], props: Partial<P>) =>
    buildGraph(device, scene, [["ParticleLayer", "P", { ...kBase, ...props }], ["DepthComposite", "C", { alphaMode: "additive" }]],
        [["G.posW", "P.posW"], ["G.normW", "P.normW"], ["G.diffuseOpacity", "C.src"], ["G.posW", "C.posW"], ["P.color", "C.layer"], ["P.depth", "C.layerDepth"]],
        ["P.color", "P.depth", "C.dst", "G.diffuseOpacity"], kSize, kSize);

async function settle(device: Parameters<typeof buildGraph>[0], scene: Parameters<typeof buildGraph>[1]) {
    const g = buildGraph(device, scene, [], [], [], kSize, kSize);
    for (let i = 0; i < 5; i++) { g.execute(device.renderContext); if ((await coverage(device, g)) > 0.2) break; }
}

gpuTest("ParticleLayer.simulationAndSplatParity", async ({ device }) => {
    const scene = makeScene(device);
    await settle(device, scene);
    scene.camera.resetHistory();
    const g = plGraph(device, scene, {});
    const pass = g.getPass("P") as unknown as ParticleLayer;
    for (let f = 0; f < 5; f++) g.execute(device.renderContext);
    const s4 = await readState(device, pass.stateBuffer!);
    g.execute(device.renderContext); // frame index 5
    const s5 = await readState(device, pass.stateBuffer!);
    const [color, depth, pos] = [await read(device, g.getOutput("P.color")!), await read(device, g.getOutput("P.depth")!), await read(device, g.getOutput("G.posW")!)];
    const ref = stepRef(kBase, s4, 5 * kBase.dt);
    let simBad = 0, simMax = 0, respawned = 0;
    for (let i = 0; i < kBase.count; i++) {
        if (s5[i * 8 + 7] !== s4[i * 8 + 7]) respawned++;
        let e = 0;
        for (let c = 0; c < 8; c++) e = Math.max(e, Math.abs(s5[i * 8 + c]! - ref[i * 8 + c]!));
        if (e > 1e-4) simBad++;
        else simMax = Math.max(simMax, e);
    }
    const sp = splatRef(kBase, s5, 5 * kBase.dt, scene, pos);
    let colBad = 0, colMax = 0, depthBad = 0, covered = 0;
    for (let k = 0; k < kSize * kSize; k++) {
        let e = 0;
        for (let c = 0; c < 3; c++) e = Math.max(e, Math.abs(color[k * 4 + c]! - sp.acc[k * 4 + c]! / 4096));
        e = Math.max(e, Math.abs(color[k * 4 + 3]! - Math.min(sp.acc[k * 4 + 3]! / 4096, 1)));
        // Per-contribution rounding (1/4096) can differ by one unit between f32 and f64.
        if (e > 4e-3) colBad++;
        else colMax = Math.max(colMax, e);
        const a = depth[k * 4]!, b = sp.dmin[k]!;
        if (Number.isFinite(b)) covered++;
        if (!(Number.isFinite(b) ? Math.abs(a - b) <= 1e-5 * b : a >= 1e38)) depthBad++;
    }
    log(`ParticleLayer sim: ${kBase.count} particles, respawned this step ${respawned}, off-reference ${simBad}, max err ${simMax.toExponential(2)}; splat: covered px ${covered}, colour off-reference px ${colBad}, max err ${colMax.toExponential(2)}, depth mismatches ${depthBad}`);
    expectEq(simBad <= 0.005 * kBase.count, true, "GPU step = CPU step (<= 0.5 % threshold flips)");
    expectEq(covered > 200, true, "particles visible");
    expectEq(colBad <= 0.005 * kSize * kSize, true, "layer colour = CPU splat (<= 0.5 % boundary pixels)");
    expectEq(depthBad <= 0.005 * kSize * kSize, true, "layer depth = CPU splat");
});

gpuTest("ParticleLayer.depthCollision", async ({ device }) => {
    const scene = makeScene(device);
    await settle(device, scene);
    const fall: Partial<P> = { count: 2048, boundsMin: [-1.2, -4.0, -1.2], boundsMax: [1.2, 1.4, 1.2], gravity: [0, -2, 0], wind: [0, 0, 0], drag: 0, turbulence: 0, lifetime: 1000, twinkle: 0, thickness: 0.08 };
    const res: Record<string, number> = {};
    for (const collide of [true, false]) {
        setPose(scene, kPoseA);
        scene.camera.resetHistory();
        const g = plGraph(device, scene, { ...fall, collide });
        const pass = g.getPass("P") as unknown as ParticleLayer;
        g.execute(device.renderContext);
        const s0 = await readState(device, pass.stateBuffer!);
        for (let f = 0; f < 40; f++) g.execute(device.renderContext);
        const s1 = await readState(device, pass.stateBuffer!);
        // Particles that started above the floor and were never respawned: how many ended below it?
        let n = 0, below = 0;
        for (let i = 0; i < 2048; i++) {
            if (s0[i * 8 + 1]! < 0.1 || s1[i * 8 + 7] !== s0[i * 8 + 7]) continue;
            n++;
            if (s1[i * 8 + 1]! < -0.02) below++;
        }
        res[collide ? "collide" : "free"] = below / Math.max(n, 1);
        log(`ParticleLayer collision=${collide}: tracked ${n}, below the floor ${below}`);
    }
    // Off-screen particles cannot collide (screen-space collision), so a small share may still fall through.
    expectEq(res.collide! < 0.25 && res.free! > 0.6, true, "collision keeps particles above the visible floor");
});

gpuTest("ParticleLayer.identity", async ({ device }) => {
    const scene = makeScene(device);
    await settle(device, scene);
    for (const props of [{ intensity: 0 }, { count: 0 }]) {
        const g = plGraph(device, scene, props);
        for (let f = 0; f < 3; f++) g.execute(device.renderContext);
        const [color, depth, dst, src] = [await read(device, g.getOutput("P.color")!), await read(device, g.getOutput("P.depth")!), await read(device, g.getOutput("C.dst")!), await read(device, g.getOutput("G.diffuseOpacity")!)];
        const emptyLayer = color.every((v) => v === 0) && [...depth].every((v, i) => (i % 4 === 0 ? v >= 1e38 : true));
        const same = bitwiseEqual(dst, src);
        log(`ParticleLayer identity ${JSON.stringify(props)}: empty layer ${emptyLayer}, composite bitwise = src ${same}`);
        expectEq(emptyLayer && same, true, `identity ${JSON.stringify(props)}`);
    }
});

gpuTest("ParticleLayer.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device);
    const res = await coldVsConsecutive(device, scene, () => plGraph(device, scene, { collide: true }), ["C.dst", "P.depth"], ["camera-reset", "prime", "explicit", "none"]);
    for (const r of res) log(`ParticleLayer cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
    for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `cold = consecutive after ${r.method}`);
    expectEq(res.find((r) => r.method === "none")!.differs[0], true, "without a cut the carried-over particle state changes X (control)");
});
