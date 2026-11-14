/**
 * SubsetRaster: re-rasterise a chosen set of geometry instances into separate G-buffer targets.
 *  1. defaultHash: prints hashes of GBufferRaster outputs (run on this branch and on the base commit; must be equal).
 *  2. Where a subset object is visible in the full scene, the subset vbuffer/depth/normW equal the full G-buffer's
 *     bit for bit and subsetMask = 1; hidden subset parts have subsetMask = 2; nothing outside the subset is drawn.
 *  3. Empty mask: nothing drawn.
 *  4. SubsetRaster in the graph leaves the main GBufferRaster outputs unchanged, also with a jittered sample pattern.
 */

import { RenderGraph, createPass, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kSize, bitwiseEqual, log, makeScene, read, type V3 } from "./stylize-helpers.js";

let graphCount = 0;

async function run(device: Device, scene: Scene, passes: [string, string, Record<string, unknown>][], edges: [string, string][],
    outputs: string[], gProps: Record<string, unknown> = { samplePattern: "Center" }, frames = 1): Promise<Record<string, Float32Array>> {
    const g = new RenderGraph(device, `SubsetRasterTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", gProps), "G");
    for (const [type, id, props] of passes) g.addPass(createPass(device, type, props), id);
    for (const [a, b] of edges) g.addEdge(a, b);
    for (const o of new Set([...outputs, "G.posW"])) g.markOutput(o);
    g.setScene(scene);
    for (let tries = 0; ; tries++) {
        g.execute(device.renderContext);
        const pos = await read(device, g.getOutput("G.posW")!);
        let hits = 0;
        for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
        if (hits > 0.2 * kSize * kSize) break;
        if (tries >= 4) throw new Error(`G-buffer still empty after ${tries + 1} frames`);
    }
    for (let f = 0; f < frames; f++) g.execute(device.renderContext);
    const out: Record<string, Float32Array> = {};
    for (const o of new Set([...outputs, "G.posW"])) out[o] = (await read(device, g.getOutput(o)!)).slice();
    return out;
}

function fnv(a: Float32Array): string {
    const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
    let h = 0x811c9dc5;
    for (let i = 0; i < u.length; i++) {
        let v = u[i]!;
        for (let k = 0; k < 4; k++) { h = Math.imul(h ^ (v & 0xff), 0x01000193) >>> 0; v >>>= 8; }
    }
    return h.toString(16).padStart(8, "0");
}

const kChannels = ["posW", "normW", "faceNormalW", "texC", "mvec", "viewW", "linearZ", "mask", "vbuffer", "depth"];

/** Box B (instances 6-10 of the stylize test scene) seen past box A, so A hides part of B. */
const kCam = { pos: [-4.2, 1.2, 1.65] as V3, target: [0.7, 0.8, -0.8] as V3 };
const kMask = [6, 7, 8, 9, 10];

gpuTest("SubsetRaster.defaultHash", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    for (const gProps of [{ samplePattern: "Center" }, { samplePattern: "Halton", sampleCount: 8 }]) {
        const r = await run(device, scene, [], [], kChannels.map((c) => `G.${c}`), gProps, 3);
        log(`defaultHash GBufferRaster ${JSON.stringify(gProps)} ${kChannels.map((c) => `${c}=${fnv(r[`G.${c}`]!)}`).join(" ")}`);
    }
    expectEq(true, true, "hashes printed");
});

gpuTest("SubsetRaster.matchesFullRasterWhereVisible", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const r = await run(device, scene,
        [["SubsetRaster", "Sub", { instanceMask: kMask }], ["StyleAovs", "SA", {}], ["StyleAovs", "SB", {}]],
        [["G.vbuffer", "Sub.sceneVBuffer"], ["G.vbuffer", "SA.vbuffer"], ["Sub.vbuffer", "SB.vbuffer"]],
        ["G.vbuffer", "G.depth", "G.normW", "Sub.vbuffer", "Sub.depth", "Sub.normW", "Sub.subsetMask", "SA.id", "SB.id"]);
    const gv = new Uint32Array(r["G.vbuffer"]!.buffer), sv = new Uint32Array(r["Sub.vbuffer"]!.buffer);
    const gd = new Uint32Array(r["G.depth"]!.buffer), sd = new Uint32Array(r["Sub.depth"]!.buffer);
    const gn = new Uint32Array(r["G.normW"]!.buffer), sn = new Uint32Array(r["Sub.normW"]!.buffer);
    const m = r["Sub.subsetMask"]!, ga = r["SA.id"]!, sb = r["SB.id"]!;
    const inMask = new Set(kMask);
    let visible = 0, hidden = 0, visBad = 0, maskBad = 0, outside = 0, hiddenBad = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        const fullInst = ga[i * 4 + 3] !== 0 ? ga[i * 4 + 1]! : -1;
        const subInst = sb[i * 4 + 3] !== 0 ? sb[i * 4 + 1]! : -1;
        if (subInst >= 0 && !inMask.has(subInst)) outside++;
        const want = subInst < 0 ? 0 : fullInst === subInst ? 1 : 2;
        if (m[i] !== want) maskBad++;
        if (inMask.has(fullInst)) {
            visible++;
            let same = m[i] === 1 && gd[i] === sd[i];
            for (let k = 0; k < 4; k++) same &&= gv[i * 4 + k] === sv[i * 4 + k] && gn[i * 4 + k] === sn[i * 4 + k];
            if (!same) visBad++;
        }
        if (m[i] === 2) {
            hidden++;
            if (inMask.has(fullInst) || !(gd[i]! <= sd[i]!)) hiddenBad++;
        }
    }
    log(`subset: visible px ${visible} (mismatch ${visBad}), hidden px ${hidden} (bad ${hiddenBad}), mask errors ${maskBad}, drawn outside the mask ${outside}`);
    expectEq(visible > 200 && hidden > 100, true, `scene shows visible and hidden subset parts (${visible}/${hidden})`);
    expectEq(visBad, 0, "subset vbuffer/depth/normW = full raster where the subset is visible");
    expectEq(hiddenBad, 0, "hidden pixels: another object is in front in the full scene");
    expectEq(maskBad, 0, "subsetMask classification");
    expectEq(outside, 0, "only instances of the mask are drawn");
});

gpuTest("SubsetRaster.emptyMaskDrawsNothing", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const r = await run(device, scene, [["SubsetRaster", "Sub", {}]], [["G.vbuffer", "Sub.sceneVBuffer"]], ["Sub.depth", "Sub.posW", "Sub.subsetMask"]);
    let drawn = 0;
    for (let i = 0; i < kSize * kSize; i++) if (r["Sub.depth"]![i] !== 1 || r["Sub.posW"]![i * 4 + 3] !== 0 || r["Sub.subsetMask"]![i] !== 0) drawn++;
    log(`empty mask: drawn px ${drawn}`);
    expectEq(drawn, 0, "empty mask draws nothing");
});

gpuTest("SubsetRaster.leavesMainGBufferUnchanged", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    for (const gProps of [{ samplePattern: "Center" }, { samplePattern: "Halton", sampleCount: 8 }]) {
        const outs = ["G.posW", "G.normW", "G.vbuffer", "G.depth", "G.mvec"];
        const a = await run(device, scene, [], [], outs, gProps, 3);
        const b = await run(device, scene, [["SubsetRaster", "Sub", { instanceMask: kMask }]], [["G.vbuffer", "Sub.sceneVBuffer"]], [...outs, "Sub.subsetMask"], gProps, 3);
        const same = outs.every((o) => bitwiseEqual(a[o]!, b[o]!));
        log(`main G-buffer with/without SubsetRaster ${JSON.stringify(gProps)}: bitwise ${same}`);
        expectEq(same, true, `main G-buffer unchanged (${gProps.samplePattern})`);
    }
});
