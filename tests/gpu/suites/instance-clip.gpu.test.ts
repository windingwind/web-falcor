/**
 * GBufferRaster per-instance clip (optional `clipBuffer` input, `clipEdge` output).
 *  1. defaultHash: prints hashes of the GBufferRaster outputs and of the generated WGSL of every compiled variant,
 *     without a clip buffer. Run on this branch and on the base commit; the hashes must be equal.
 *  2. clipEdge connected without a clip buffer: the other outputs are bitwise unchanged, clipEdge = 0.
 *  3. A clip buffer with all thresholds 0 leaves the outputs bitwise unchanged (vbuffer connected, so both pull vertices).
 *  4. Dither mode: the clipped pixels are exactly the CPU 4x4 Bayer prediction; surviving pixels keep their hit bit for
 *     bit; clipEdge matches the CPU value.
 *  5. Object-noise mode: clip decisions match a CPU mirror of the noise (outside a 1e-4 band around the threshold);
 *     instances past the end of the buffer are untouched.
 *  6. No state: the same clip buffer gives bitwise the same outputs on a fresh graph and after other clip settings.
 */

import { Buffer, MemoryType, RenderGraph, ResourceBindFlags, createPass, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kSize, bitwiseEqual, log, makeScene, read, type V3 } from "./stylize-helpers.js";

let graphCount = 0;

/** Box B (instances 6-10 of the stylize test scene) seen past box A (1-5), so A hides part of B. */
const kCam = { pos: [-4.2, 1.2, 1.65] as V3, target: [0.7, 0.8, -0.8] as V3 };
const kBoxA = [1, 2, 3, 4, 5];
const kBoxB = [6, 7, 8, 9, 10];
type ClipEntry = [threshold: number, mode: number, edgeWidth: number, scale: number];

function clipBuffer(device: Device, entries: ClipEntry[]): Buffer {
    const b = new Buffer(device, { size: entries.length * 16, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "InstanceClipTest" });
    b.setBlob(new Float32Array(entries.flat()));
    return b;
}

/** Entries for instances 0..n-1: `set` gets `e`, the rest mode 0. */
function entries(n: number, set: number[], e: ClipEntry): ClipEntry[] {
    return Array.from({ length: n }, (_, i) => (set.includes(i) ? e : [0, 0, 0, 0]));
}

interface RunOpts { gProps?: Record<string, unknown>; frames?: number; clip?: Buffer; }

/** GBufferRaster "G" + StyleAovs "S" (instance ids); returns the marked outputs. With `seq`, several clip buffers in turn on one graph. */
async function run(device: Device, scene: Scene, outputs: string[], o: RunOpts = {}, seq: (Buffer | undefined)[] = []): Promise<Record<string, Float32Array>> {
    const g = new RenderGraph(device, `InstanceClipTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", o.gProps ?? { samplePattern: "Center" }), "G");
    const wantIds = outputs.includes("S.id");
    if (wantIds) {
        g.addPass(createPass(device, "StyleAovs", {}), "S");
        g.addEdge("G.vbuffer", "S.vbuffer");
    }
    for (const out of new Set([...outputs, "G.posW"])) g.markOutput(out);
    g.setScene(scene);
    for (const b of seq) {
        if (b) g.setInput("G.clipBuffer", b);
        g.execute(device.renderContext);
    }
    if (o.clip) g.setInput("G.clipBuffer", o.clip);
    for (let tries = 0; ; tries++) {
        g.execute(device.renderContext);
        const pos = await read(device, g.getOutput("G.posW")!);
        let hits = 0;
        for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
        if (hits > 0.2 * kSize * kSize) break;
        if (tries >= 4) throw new Error(`G-buffer still empty after ${tries + 1} frames`);
    }
    for (let f = 0; f < (o.frames ?? 1); f++) g.execute(device.renderContext);
    const res: Record<string, Float32Array> = {};
    for (const out of new Set([...outputs, "G.posW"])) res[out] = (await read(device, g.getOutput(out)!)).slice();
    const pass = g.getPass("G") as unknown as { variants: Map<string, { state: { vertexKernel: { wgsl: string }; fragmentKernel: { wgsl: string } } }> };
    (res as Record<string, unknown>)["__wgsl"] = [...pass.variants].map(([key, v]) => `${key}=${fnvStr(v.state.vertexKernel.wgsl)}/${fnvStr(v.state.fragmentKernel.wgsl)}`);
    return res;
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
function fnvStr(s: string): string {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
    return `${h.toString(16).padStart(8, "0")}:${s.length}`;
}

const kChannels = ["posW", "normW", "tangentW", "faceNormalW", "texC", "texGrads", "mvec", "mtlData", "guideNormalW", "diffuseOpacity",
    "specRough", "emissive", "viewW", "pnFwidth", "linearZ", "mask", "vbuffer", "depth"];
const kNoPull = ["posW", "normW", "faceNormalW", "texC", "mvec", "depth"];

/** Instance id per pixel from StyleAovs.id (raw id in g, coverage in a), -1 = background. */
const instanceAt = (ids: Float32Array, i: number) => (ids[i * 4 + 3] !== 0 ? ids[i * 4 + 1]! : -1);
const hitEq = (a: Uint32Array, b: Uint32Array, i: number) => a[i * 4] === b[i * 4] && a[i * 4 + 1] === b[i * 4 + 1] && a[i * 4 + 2] === b[i * 4 + 2] && a[i * 4 + 3] === b[i * 4 + 3];
const u32 = (a: Float32Array) => new Uint32Array(a.buffer, a.byteOffset, a.length);
/** One float per pixel from a readback of an R32Float target (handles padded/expanded readbacks). */
const scalarAt = (a: Float32Array, i: number) => a[i * (a.length / (kSize * kSize))]!;

const kBayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
const bayer = (x: number, y: number) => Math.fround((kBayer[(y & 3) * 4 + (x & 3)]! + 0.5) / 16);

/** CPU mirror of clipLatticeValue / clipValueNoise / clipNoise (mode 2), in float32 steps. */
function lattice(cx: number, cy: number, cz: number): number {
    let x = (Math.imul(cx, 1664525) + 1013904223) >>> 0, y = (Math.imul(cy, 1664525) + 1013904223) >>> 0, z = (Math.imul(cz, 1664525) + 1013904223) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0; y = (y + Math.imul(z, x)) >>> 0; z = (z + Math.imul(x, y)) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0; y = (y ^ (y >>> 16)) >>> 0; z = (z ^ (z >>> 16)) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0;
    return Math.fround((x >>> 8) / 16777216);
}
const f = Math.fround;
const lerp = (a: number, b: number, t: number) => f(a + f(f(b - a) * t));
function valueNoise(p: V3): number {
    const fl = p.map(Math.floor) as V3;
    const t = p.map((v, k) => f(v - fl[k]!)) as V3;
    const w = t.map((v) => f(f(v * v) * f(3 - f(2 * v)))) as V3;
    const [cx, cy, cz] = fl;
    const L = (dx: number, dy: number, dz: number) => lattice(cx + dx, cy + dy, cz + dz);
    const x00 = lerp(L(0, 0, 0), L(1, 0, 0), w[0]), x10 = lerp(L(0, 1, 0), L(1, 1, 0), w[0]);
    const x01 = lerp(L(0, 0, 1), L(1, 0, 1), w[0]), x11 = lerp(L(0, 1, 1), L(1, 1, 1), w[0]);
    return lerp(lerp(x00, x10, w[1]), lerp(x01, x11, w[1]), w[2]);
}
function objectNoise(pos: V3, scale: number): number {
    const p = pos.map((v) => f(v * scale)) as V3;
    return f((4 * valueNoise(p) + 2 * valueNoise(p.map((v) => f(2 * v + 17)) as V3) + valueNoise(p.map((v) => f(4 * v + 31)) as V3)) / 7);
}

gpuTest("InstanceClip.defaultHash", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    for (const [label, chans] of [["pulled", kChannels], ["indexed", kNoPull]] as const) {
        for (const gProps of [{ samplePattern: "Center" }, { samplePattern: "Halton", sampleCount: 8 }]) {
            const r = await run(device, scene, chans.map((c) => `G.${c}`), { gProps, frames: 3 });
            log(`defaultHash GBufferRaster ${label} ${JSON.stringify(gProps)} ${chans.map((c) => `${c}=${fnv(r[`G.${c}`]!)}`).join(" ")}`);
            log(`defaultHash wgsl ${label} ${JSON.stringify(gProps)} ${((r as Record<string, unknown>)["__wgsl"] as string[]).join(" ")}`);
        }
    }
    expectEq(true, true, "hashes printed");
});

gpuTest("InstanceClip.unboundIsIdentity", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const base = await run(device, scene, kChannels.map((c) => `G.${c}`));
    const withEdge = await run(device, scene, [...kChannels.map((c) => `G.${c}`), "G.clipEdge"]);
    let diff = 0;
    for (const c of kChannels) if (!bitwiseEqual(base[`G.${c}`]!, withEdge[`G.${c}`]!)) diff++;
    const edgeNonZero = withEdge["G.clipEdge"]!.some((v) => v !== 0);
    log(`unbound: channels differing ${diff}/${kChannels.length}, clipEdge non-zero ${edgeNonZero}`);
    expectEq(diff, 0, "outputs with clipEdge connected but no clip buffer");
    expectEq(edgeNonZero, false, "clipEdge without clip buffer is 0");
});

gpuTest("InstanceClip.zeroThresholdIsIdentity", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const base = await run(device, scene, kChannels.map((c) => `G.${c}`));
    const clip = await run(device, scene, [...kChannels.map((c) => `G.${c}`), "G.clipEdge"], { clip: clipBuffer(device, entries(11, [...kBoxA, ...kBoxB], [0, 2, 0.1, 3])) });
    const differing = kChannels.filter((c) => !bitwiseEqual(base[`G.${c}`]!, clip[`G.${c}`]!));
    log(`zero threshold: differing channels [${differing.join(",")}], clipEdge non-zero ${clip["G.clipEdge"]!.some((v) => v !== 0)}`);
    expectEq(differing.length, 0, "threshold 0 = no clip (bitwise)");
});

gpuTest("InstanceClip.ditherMatchesCpu", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const outs = ["G.vbuffer", "G.depth", "S.id", "G.clipEdge"];
    const base = await run(device, scene, outs);
    for (const [t, w] of [[0.5, 0.25], [0.8, 0.1], [0.2, 0.5]]) {
        const clip = await run(device, scene, outs, { clip: clipBuffer(device, entries(11, kBoxB, [t, 1, w, 1])) });
        const bv = u32(base["G.vbuffer"]!), cv = u32(clip["G.vbuffer"]!);
        const bid = base["S.id"]!, cid = clip["S.id"]!, edge = clip["G.clipEdge"]!;
        let clipped = 0, kept = 0, bad = 0, edgeErr = 0;
        for (let y = 0; y < kSize; y++) for (let x = 0; x < kSize; x++) {
            const i = y * kSize + x;
            const inst = instanceAt(bid, i);
            const n = bayer(x, y);
            if (kBoxB.includes(inst) && n < t) {
                clipped++;
                if (kBoxB.includes(instanceAt(cid, i))) bad++; // the same pixel clips every B face
            } else {
                kept++;
                if (!hitEq(bv, cv, i)) bad++;
                const want = kBoxB.includes(inst) ? Math.min(Math.max(1 - (n - t) / w, 0), 1) : 0;
                edgeErr = Math.max(edgeErr, Math.abs(scalarAt(edge, i) - want));
            }
        }
        log(`dither t=${t} w=${w}: clipped ${clipped}, kept ${kept}, mismatches ${bad}, clipEdge max err ${edgeErr.toExponential(2)}`);
        expectEq(clipped > 0, true, "something clipped");
        expectEq(bad, 0, "dither clip mask = CPU Bayer prediction");
        expectEq(edgeErr < 1e-6, true, "clipEdge = CPU value");
    }
});

gpuTest("InstanceClip.objectNoiseMatchesCpu", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const outs = ["G.vbuffer", "G.posW", "S.id", "G.clipEdge"];
    const base = await run(device, scene, outs);
    // Buffer covers instances 0-5 only: box B (6-10) lies past its end and must stay untouched.
    for (const [t, w, s] of [[0.45, 0.08, 3], [0.6, 0.2, 1.5]]) {
        const clip = await run(device, scene, outs, { clip: clipBuffer(device, entries(6, kBoxA, [t, 2, w, s])) });
        const bv = u32(base["G.vbuffer"]!), cv = u32(clip["G.vbuffer"]!);
        const bid = base["S.id"]!, pos = base["G.posW"]!, edge = clip["G.clipEdge"]!;
        let clipped = 0, kept = 0, bad = 0, band = 0, other = 0, otherBad = 0, edgeErr = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            const inst = instanceAt(bid, i);
            if (!kBoxA.includes(inst)) {
                other++;
                // Non-A pixels: A never covered them in front, so they keep their hit.
                if (!hitEq(bv, cv, i)) otherBad++;
                continue;
            }
            // The test scene's meshes are placed with identity transforms: object-space = world-space position.
            const n = objectNoise([pos[i * 4]!, pos[i * 4 + 1]!, pos[i * 4 + 2]!], s);
            if (Math.abs(n - t) < 1e-4) { band++; continue; }
            const same = hitEq(bv, cv, i);
            if (n < t) { clipped++; if (same) bad++; }
            else {
                kept++;
                if (!same) bad++;
                edgeErr = Math.max(edgeErr, Math.abs(scalarAt(edge, i) - Math.min(Math.max(1 - (n - t) / w, 0), 1)));
            }
        }
        log(`object noise t=${t} w=${w} scale=${s}: clipped ${clipped}, kept ${kept}, band ${band}, mismatches ${bad}, other px ${other} changed ${otherBad}, clipEdge max err ${edgeErr.toExponential(2)}`);
        expectEq(clipped > 0 && kept > 0, true, "partial clip");
        expectEq(bad, 0, "object-noise clip = CPU mirror");
        expectEq(otherBad, 0, "pixels not covered by A are unchanged (B past the buffer end)");
        expectEq(edgeErr < 1e-4, true, "clipEdge = CPU value");
    }
});

gpuTest("InstanceClip.stateless", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const outs = ["G.vbuffer", "G.normW", "G.depth", "G.clipEdge"];
    const X = () => clipBuffer(device, entries(11, [...kBoxA, ...kBoxB], [0.5, 2, 0.1, 3]));
    const A = clipBuffer(device, entries(11, kBoxA, [0.7, 1, 0.3, 1]));
    const B = clipBuffer(device, entries(11, kBoxB, [0.3, 2, 0.05, 5]));
    const cold = await run(device, scene, outs, { clip: X() });
    const warm = await run(device, scene, outs, { clip: X() }, [A, B]);
    const differing = outs.filter((o) => !bitwiseEqual(cold[o]!, warm[o]!));
    log(`stateless: cold vs after A,B differing [${differing.join(",")}]`);
    expectEq(differing.length, 0, "cold == consecutive (bitwise)");
});
