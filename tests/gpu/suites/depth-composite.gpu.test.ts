/**
 * DepthComposite against a CPU reference: a ray-traced (analytic SDF) sphere layer with its own colour and depth is
 * composited into the G-buffer scene for every alpha mode, with soft depth, with a transmittance input and with depth
 * from a depth texture; opacity 0 and an empty layer reproduce the input bit for bit.
 */

import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, camInfo, kSize, log, makeScene, read, type V3 } from "./stylize-helpers.js";
import { buildGraph, coverage, rgbaTexture } from "./fx-helpers.js";

const kFar = 3.402823466e38;

/** Sphere layer: premultiplied colour (alpha 0.7 inside, 0 outside), view depth of the hit (kFar outside), transmittance. */
function sphereLayer(scene: ReturnType<typeof makeScene>, centre: V3, radius: number) {
    const cam = camInfo(scene);
    const right = cam.viewRows[0]!, up = cam.viewRows[1]!;
    const t = Math.tan(0.5 * cam.fovY);
    const colour = new Float32Array(kSize * kSize * 4), depth = new Float32Array(kSize * kSize * 4), trans = new Float32Array(kSize * kSize * 4);
    let hits = 0;
    for (let y = 0; y < kSize; y++) {
        for (let x = 0; x < kSize; x++) {
            const i = y * kSize + x;
            const sx = (2 * (x + 0.5) / kSize - 1) * t, sy = (1 - 2 * (y + 0.5) / kSize) * t;
            let d = [0, 1, 2].map((c) => cam.fwd[c]! + sx * right[c]! + sy * up[c]!);
            const l = Math.hypot(d[0]!, d[1]!, d[2]!);
            d = d.map((v) => v / l);
            // Ray-sphere (sphere tracing would converge to the same hit for an SDF sphere).
            const oc = [0, 1, 2].map((c) => cam.pos[c]! - centre[c]!);
            const b = oc[0]! * d[0]! + oc[1]! * d[1]! + oc[2]! * d[2]!;
            const cc = oc[0]! ** 2 + oc[1]! ** 2 + oc[2]! ** 2 - radius * radius;
            const disc = b * b - cc;
            trans.set([1, 1, 1, 1], i * 4);
            depth[i * 4] = kFar;
            depth[i * 4 + 3] = 1;
            if (disc < 0) continue;
            const tt = -b - Math.sqrt(disc);
            if (tt <= 0) continue;
            hits++;
            const p = [0, 1, 2].map((c) => cam.pos[c]! + tt * d[c]!);
            const n = [0, 1, 2].map((c) => (p[c]! - centre[c]!) / radius);
            const shade = 0.25 + 0.75 * Math.max(0, 0.3 * n[0]! + 0.9 * n[1]! + 0.3 * n[2]!);
            const a = 0.7;
            colour.set([0.2 * shade * a, 0.9 * shade * a, 0.5 * shade * a, a], i * 4);
            depth[i * 4] = (p[0]! - cam.pos[0]) * cam.fwd[0] + (p[1]! - cam.pos[1]) * cam.fwd[1] + (p[2]! - cam.pos[2]) * cam.fwd[2];
            trans.set([0.5, 0.6, 0.9, 1], i * 4);
        }
    }
    return { colour, depth, trans, hits };
}

interface Mode { alphaMode: string; opacity: number; softness: number; trans?: boolean; depthTex?: boolean }

function reference(m: Mode, src: Float32Array, pos: Float32Array, lay: ReturnType<typeof sphereLayer>, cam: ReturnType<typeof camInfo>, depthTex?: Float32Array) {
    const out = new Float32Array(src.length), outDepth = new Float32Array(src.length / 4);
    for (let i = 0; i < kSize * kSize; i++) {
        let ds = kFar;
        if (depthTex) ds = depthTex[i * 4]!;
        else if (pos[i * 4 + 3] !== 0) ds = (pos[i * 4]! - cam.pos[0]) * cam.fwd[0] + (pos[i * 4 + 1]! - cam.pos[1]) * cam.fwd[1] + (pos[i * 4 + 2]! - cam.pos[2]) * cam.fwd[2];
        const dl = lay.depth[i * 4]!;
        let vis = dl < ds ? 1 : 0;
        if (m.softness > 0 && ds < kFar) vis = Math.min(Math.max((ds - dl) / m.softness, 0), 1);
        const k = vis * m.opacity;
        const l = lay.colour.subarray(i * 4, i * 4 + 4);
        out.set(src.subarray(i * 4, i * 4 + 4), i * 4);
        outDepth[i] = ds;
        if (!(k > 0) || (l.every((v) => v === 0) && !m.trans)) continue;
        for (let c = 0; c < 3; c++) {
            let add = l[c]!, keep = 1 - l[3]!;
            if (m.alphaMode === "additive") keep = 1;
            if (m.alphaMode === "straight") add = l[c]! * l[3]!;
            if (m.trans) keep = lay.trans[i * 4 + c]!;
            out[i * 4 + c] = src[i * 4 + c]! * (1 + (keep - 1) * k) + add * k;
        }
        if (l[3]! > 0) outDepth[i] = Math.min(ds, dl);
    }
    return { out, outDepth };
}

gpuTest("DepthComposite.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    const cam = camInfo(scene);
    // The sphere intersects box A, so the depth test cuts it along the box faces.
    const lay = sphereLayer(scene, [-0.6, 0.6, -0.1], 0.55);
    const inputs = { "C.layer": rgbaTexture(device, kSize, kSize, lay.colour), "C.layerDepth": rgbaTexture(device, kSize, kSize, lay.depth) };
    const transTex = rgbaTexture(device, kSize, kSize, lay.trans);
    const modes: Mode[] = [
        { alphaMode: "premultiplied", opacity: 1, softness: 0 },
        { alphaMode: "straight", opacity: 0.6, softness: 0 },
        { alphaMode: "additive", opacity: 1, softness: 0 },
        { alphaMode: "premultiplied", opacity: 1, softness: 0.3 },
        { alphaMode: "premultiplied", opacity: 0.8, softness: 0, trans: true },
        { alphaMode: "premultiplied", opacity: 1, softness: 0, depthTex: true },
    ];
    for (const m of modes) {
        const edges: [string, string][] = [["G.diffuseOpacity", "C.src"]];
        const ins: Record<string, ReturnType<typeof rgbaTexture>> = { ...inputs };
        if (m.trans) ins["C.transmittance"] = transTex;
        let depthTex: Float32Array | undefined;
        if (m.depthTex) {
            // Scene depth from a texture: the CPU-computed view depth of posW, same convention as the layer.
            const g0 = buildGraph(device, scene, [], [], [], kSize, kSize);
            for (let i = 0; i < 5; i++) { g0.execute(device.renderContext); if ((await coverage(device, g0)) > 0.2) break; }
            const pos0 = await read(device, g0.getOutput("G.posW")!);
            depthTex = new Float32Array(kSize * kSize * 4);
            for (let i = 0; i < kSize * kSize; i++) depthTex[i * 4] = pos0[i * 4 + 3] !== 0 ? Math.fround((pos0[i * 4]! - cam.pos[0]) * cam.fwd[0] + (pos0[i * 4 + 1]! - cam.pos[1]) * cam.fwd[1] + (pos0[i * 4 + 2]! - cam.pos[2]) * cam.fwd[2]) : kFar;
            ins["C.depth"] = rgbaTexture(device, kSize, kSize, depthTex);
        } else {
            edges.push(["G.posW", "C.posW"]);
        }
        const g = buildGraph(device, scene, [["DepthComposite", "C", { alphaMode: m.alphaMode, opacity: m.opacity, softness: m.softness }]], edges, ["C.dst", "C.dstDepth", "G.diffuseOpacity"], kSize, kSize, ins);
        for (let i = 0; i < 5; i++) { g.execute(device.renderContext); if ((await coverage(device, g)) > 0.2) break; }
        const [dst, dd, src, pos] = [await read(device, g.getOutput("C.dst")!), await read(device, g.getOutput("C.dstDepth")!), await read(device, g.getOutput("G.diffuseOpacity")!), await read(device, g.getOutput("G.posW")!)];
        const ref = reference(m, src, pos, lay, cam, depthTex);
        // f32 (GPU) vs f64 (CPU) view depths can flip the hard test where the sphere crosses a surface: count such pixels.
        let maxErr = 0, badPx = 0, depthBad = 0, changed = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            let e = 0;
            for (let c = 0; c < 4; c++) e = Math.max(e, Math.abs(dst[i * 4 + c]! - ref.out[i * 4 + c]!));
            if (e > 1e-5) badPx++;
            else maxErr = Math.max(maxErr, e);
            const a = dd[i * 4]!, b = ref.outDepth[i]!;
            if (!(Math.abs(a - b) <= 1e-5 * Math.max(1, Math.abs(b)) || (a >= 1e38 && b >= 1e38))) depthBad++;
            if (dst[i * 4] !== src[i * 4]) changed++;
        }
        log(`DepthComposite ${JSON.stringify(m)}: layer px ${lay.hits}, changed px ${changed}, px off reference ${badPx}, max err elsewhere ${maxErr.toExponential(2)}, depth mismatches ${depthBad}`);
        expectEq(badPx <= 0.002 * kSize * kSize, true, "colour = CPU reference (<= 0.2 % depth-tie pixels)");
        expectEq(depthBad <= 0.002 * kSize * kSize, true, "dstDepth = CPU reference (<= 0.2 % depth-tie pixels)");
        expectEq(changed > 0 && changed < lay.hits, true, "layer partly hidden by the scene");
    }

    // Identity: opacity 0, and an empty layer.
    const empty = new Float32Array(kSize * kSize * 4);
    for (const [label, props, layer] of [["opacity 0", { opacity: 0 }, lay.colour], ["empty layer", {}, empty]] as const) {
        const g = buildGraph(device, scene, [["DepthComposite", "C", props]], [["G.diffuseOpacity", "C.src"], ["G.posW", "C.posW"]], ["C.dst", "G.diffuseOpacity"], kSize, kSize,
            { "C.layer": rgbaTexture(device, kSize, kSize, layer), "C.layerDepth": rgbaTexture(device, kSize, kSize, lay.depth) });
        for (let i = 0; i < 5; i++) { g.execute(device.renderContext); if ((await coverage(device, g)) > 0.2) break; }
        const same = bitwiseEqual(await read(device, g.getOutput("C.dst")!), await read(device, g.getOutput("G.diffuseOpacity")!));
        log(`DepthComposite identity (${label}): bitwise ${same}`);
        expectEq(same, true, `identity: ${label}`);
    }
});
