/**
 * MotionStats (packages/render-passes/src/MotionStats) against a float64 CPU reference:
 *  - synthetic motion field (non-multiple-of-16 size, mask on half the pixels, angles and magnitudes away from bin
 *    edges): means / rms within 1e-5 relative, max / counts / histograms exact;
 *  - determinism: repeated execution and a fresh pass instance give bitwise equal stats;
 *  - G-buffer motion vectors of a camera move (mask = posW): CPU reference from the readbacks.
 */

import { RenderGraph, ResourceBindFlags, ResourceFormat, createPass, float3, type Device, type Texture } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kSize, makeScene, read, log } from "./stylize-helpers.js";

interface Ref { mean: [number, number]; meanMag: number; maxMag: number; rms: number; movingShare: number; n: number; moving: number; magHist: number[]; dirHist: number[] }

/** float64 reference; magnitude and angle use float32 inputs (m = -mvec * dim rounded to f32 like the shader). */
function cpuStats(mvec: Float32Array, comps: number, mask: Float32Array | null, W: number, H: number, magBins: number, dirBins: number, magRange: number, minMotion: number): Ref {
    let sx = 0, sy = 0, sm = 0, s2 = 0, mx = 0, n = 0, moving = 0;
    const magHist = new Array(magBins).fill(0), dirHist = new Array(dirBins).fill(0);
    for (let i = 0; i < W * H; i++) {
        if (mask && mask[i * 4 + 3] === 0) continue;
        const x = Math.fround(-mvec[i * comps]! * W), y = Math.fround(-mvec[i * comps + 1]! * H);
        const mag = Math.hypot(x, y);
        sx += x; sy += y; sm += mag; s2 += mag * mag; mx = Math.max(mx, mag); n++;
        magHist[Math.min(Math.floor((mag / magRange) * magBins), magBins - 1)]++;
        if (mag > minMotion) {
            moving++;
            const u = (Math.atan2(y, x) + Math.PI) / (2 * Math.PI);
            dirHist[Math.min(Math.floor(u * dirBins), dirBins - 1)]++;
        }
    }
    return { mean: [sx / n, sy / n], meanMag: sm / n, maxMag: mx, rms: Math.sqrt(s2 / n), movingShare: moving / n, n, moving, magHist, dirHist };
}

let graphCount = 0;
function statsGraph(device: Device, W: number, H: number, props: Record<string, unknown>, mvec: Texture, mask: Texture | null): RenderGraph {
    const g = new RenderGraph(device, `MotionStatsTest${++graphCount}`);
    g.onResize(W, H);
    g.addPass(createPass(device, "MotionStats", props), "M");
    g.setInput("M.mvec", mvec);
    if (mask) g.setInput("M.mask", mask);
    g.markOutput("M.stats");
    g.markOutput("M.histogram");
    return g;
}

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-6);

function check(tag: string, s: Float32Array, h: Float32Array, ref: Ref, magBins: number, dirBins: number, exactHist: boolean): void {
    const hw = Math.max(magBins, dirBins);
    const gpuMag = Array.from({ length: magBins }, (_v, b) => h[b * 4 + 1]!);
    const gpuDir = Array.from({ length: dirBins }, (_v, b) => h[(hw + b) * 4 + 1]!);
    const histDiff = gpuMag.reduce((d, c, b) => d + Math.abs(c - ref.magHist[b]), 0) + gpuDir.reduce((d, c, b) => d + Math.abs(c - ref.dirHist[b]), 0);
    const errs = {
        meanX: Math.abs(s[0]! - ref.mean[0]) / Math.max(ref.meanMag, 1e-6), meanY: Math.abs(s[1]! - ref.mean[1]) / Math.max(ref.meanMag, 1e-6),
        meanMag: rel(s[2]!, ref.meanMag), maxMag: rel(s[3]!, ref.maxMag), rms: rel(s[4]!, ref.rms), moving: Math.abs(s[5]! - ref.movingShare),
    };
    log(`MotionStats[${tag}]: n ${s[9]} (ref ${ref.n}), moving ${s[11]} (ref ${ref.moving}), errors ${JSON.stringify(Object.fromEntries(Object.entries(errs).map(([k, v]) => [k, Number(v.toExponential(2))])))}, histogram count diff ${histDiff}`);
    log(`MotionStats[${tag}]: mean (${s[0]!.toFixed(4)}, ${s[1]!.toFixed(4)}) px, |m| mean ${s[2]!.toFixed(4)} max ${s[3]!.toFixed(4)}, dominant angle ${s[6]!.toFixed(4)} share ${s[7]!.toFixed(4)}`);
    expectEq(s[9], ref.n, "pixel count");
    for (const [k, v] of Object.entries(errs)) expectEq(v < 1e-5, true, `${k} error ${v}`);
    if (exactHist) {
        expectEq(s[11], ref.moving, "moving count");
        expectEq(histDiff, 0, "histograms exact");
    } else expectEq(histDiff <= Math.max(2, 1e-3 * ref.n), true, `histograms within bin-edge rounding (${histDiff})`);
    // Dominant direction = centre of the fullest direction bin.
    let best = 0;
    for (let b = 1; b < dirBins; b++) if (ref.dirHist[b] > ref.dirHist[best]) best = b;
    if (ref.moving > 0 && exactHist) expectEq(Math.abs(s[6]! - (-Math.PI + (best + 0.5) * (2 * Math.PI) / dirBins)) < 1e-5, true, "dominant angle");
}

gpuTest("MotionStats.syntheticMatchesCpu", async ({ device }) => {
    const W = 203, H = 117, magBins = 12, dirBins = 16, magRange = 24, minMotion = 0.5;
    const mv = new Float32Array(W * H * 4), mk = new Float32Array(W * H * 4);
    let seed = 12345;
    const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
    for (let i = 0; i < W * H; i++) {
        // Angles at direction-bin centres +-0.3 bin, magnitudes at magnitude-bin centres +-0.3 bin (some static, some overflow).
        const bin = rnd() < 0.4 ? 3 : Math.floor(rnd() * dirBins);
        const ang = -Math.PI + (bin + 0.5 + (rnd() - 0.5) * 0.6) * (2 * Math.PI) / dirBins;
        const r = rnd();
        const mag = r < 0.15 ? 0.1 * rnd() : r < 0.2 ? magRange * (1.2 + rnd()) : (Math.floor(rnd() * magBins) + 0.5 + (rnd() - 0.5) * 0.6) * magRange / magBins;
        mv[i * 4] = -(mag * Math.cos(ang)) / W;
        mv[i * 4 + 1] = -(mag * Math.sin(ang)) / H;
        mk[i * 4 + 3] = (i % W) < W / 2 || rnd() < 0.3 ? 1 : 0;
    }
    const flags = ResourceBindFlags.ShaderResource;
    const mvec = device.createTexture2D(W, H, ResourceFormat.RGBA32Float, 1, 1, mv, flags);
    const mask = device.createTexture2D(W, H, ResourceFormat.RGBA32Float, 1, 1, mk, flags);
    const props = { magBins, dirBins, magRange, minMotion };
    for (const [tag, m] of [["mask", mask], ["all", null]] as const) {
        const g = statsGraph(device, W, H, props, mvec, m);
        g.execute(device.renderContext);
        const s1 = await read(device, g.getOutput("M.stats")!);
        const h1 = await read(device, g.getOutput("M.histogram")!);
        check(`synthetic ${tag}`, s1, h1, cpuStats(mv, 4, m ? mk : null, W, H, magBins, dirBins, magRange, minMotion), magBins, dirBins, true);
        g.execute(device.renderContext);
        const s2 = await read(device, g.getOutput("M.stats")!);
        const h2 = await read(device, g.getOutput("M.histogram")!);
        const g2 = statsGraph(device, W, H, props, mvec, m);
        g2.execute(device.renderContext);
        const s3 = await read(device, g2.getOutput("M.stats")!);
        const same = (a: Float32Array, b: Float32Array) => new Uint32Array(a.buffer).every((v, i) => v === new Uint32Array(b.buffer)[i]);
        log(`MotionStats[synthetic ${tag}]: repeat bitwise ${same(s1, s2) && same(h1, h2)}, fresh pass bitwise ${same(s1, s3)}`);
        expectEq(same(s1, s2) && same(h1, h2), true, "repeated execution is bitwise equal");
        expectEq(same(s1, s3), true, "fresh pass instance is bitwise equal");
    }
});

gpuTest("MotionStats.gbufferCameraMove", async ({ device }) => {
    const scene = makeScene(device, [0.03, 0.03, 0.03]);
    const g = new RenderGraph(device, `MotionStatsTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    g.addPass(createPass(device, "MotionStats", {}), "M");
    g.addEdge("G.mvec", "M.mvec");
    g.addEdge("G.posW", "M.mask");
    for (const o of ["G.mvec", "G.posW", "M.stats", "M.histogram"]) g.markOutput(o);
    g.setScene(scene);
    // First frame (the G-buffer must show the scene), then settle.
    g.execute(device.renderContext);
    const pos0 = await read(device, g.getOutput("G.posW")!);
    let hits = 0;
    for (let i = 3; i < pos0.length; i += 4) if (pos0[i] !== 0) hits++;
    for (let k = 0; k < 2; k++) g.execute(device.renderContext); // prev = current camera
    expectEq(hits > 0.2 * kSize * kSize, true, `G-buffer shows the scene (${hits} px)`);
    const still = await read(device, g.getOutput("M.stats")!);
    log(`MotionStats[static camera]: mean |m| ${still[2]} max ${still[3]} moving ${still[11]}`);
    expectEq(still[3]! < 1e-2 && still[11] === 0, true, "static camera: no motion beyond the G-buffer float noise (~2e-3 px)");
    const p = scene.camera.getPosition();
    scene.camera.setPosition(new float3(p.x + 0.12, p.y - 0.05, p.z + 0.04));
    g.execute(device.renderContext);
    const mvec = await read(device, g.getOutput("G.mvec")!);
    const posW = await read(device, g.getOutput("G.posW")!);
    const s = await read(device, g.getOutput("M.stats")!);
    const h = await read(device, g.getOutput("M.histogram")!);
    const comps = mvec.length / (kSize * kSize);
    const ref = cpuStats(mvec, comps, posW, kSize, kSize, 16, 16, 32, 0.5);
    check("gbuffer", s, h, ref, 16, 16, false);
    expectEq(ref.moving > 0.5 * ref.n, true, "camera move: most geometry pixels move");
    scene.camera.setPosition(p);
});
