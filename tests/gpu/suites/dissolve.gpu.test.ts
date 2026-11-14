/**
 * DissolveControl + Dissolve (per-object dissolve with a burning edge on the GBufferRaster per-instance clip).
 *  1. DissolveControl's clip buffer = the CPU entries (dissolveClipEntries): the G-buffer with the control pass equals
 *     the G-buffer with the CPU entries bound as an external buffer, bit for bit; entry values follow the time formula.
 *  2. Dissolve output = CPU formula from the lit colour and clipEdge.
 *  3. Identity: threshold 0 gives bitwise the plain lighting chain; edgeIntensity 0 + charAmount 0 copy the lit colour.
 *  4. Time drives the dissolve: visible pixels of the listed objects fall monotonically to 0; reverse runs it backwards.
 *  5. No state: a frame on a fresh graph equals the same frame after other times on one graph (bitwise).
 */

import { Buffer, MemoryType, Properties, RenderGraph, ResourceBindFlags, createPass, type Device, type Scene } from "@web-falcor/falcor";
import { dissolveClipEntries } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kSize, bitwiseEqual, log, makeScene, read, type V3 } from "./stylize-helpers.js";

let graphCount = 0;
const kCam = { pos: [-4.2, 1.2, 1.65] as V3, target: [0.7, 0.8, -0.8] as V3 };
const kBoxA = [1, 2, 3, 4, 5];
const kBoxB = [6, 7, 8, 9, 10];

const kControl = { instances: [...kBoxA, ...kBoxB], threshold: 1, time: 1, startTime: 0, duration: 2, stagger: 0.1, noise: "object", noiseScale: 3, edgeWidth: 0.1 };

interface Opts {
    control?: Record<string, unknown> | null; // DissolveControl props (null: no control pass)
    dissolve?: Record<string, unknown> | null; // Dissolve props (null: no Dissolve pass)
    external?: Buffer; // external G.clipBuffer instead of the control pass
    times?: number[]; // control times executed before the final frame
}

async function run(device: Device, scene: Scene, outputs: string[], o: Opts): Promise<Record<string, Float32Array>> {
    const g = new RenderGraph(device, `DissolveTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    g.addPass(createPass(device, "DeferredLighting", {}), "L");
    g.addEdge("G.vbuffer", "L.vbuffer");
    g.addEdge("G.viewW", "L.viewW");
    g.addPass(createPass(device, "StyleAovs", {}), "S");
    g.addEdge("G.vbuffer", "S.vbuffer");
    if (o.control) {
        g.addPass(createPass(device, "DissolveControl", o.control), "DC");
        g.addEdge("DC.clipBuffer", "G.clipBuffer");
    }
    if (o.dissolve) {
        g.addPass(createPass(device, "Dissolve", o.dissolve), "D");
        g.addEdge("L.color", "D.color");
        g.addEdge("G.clipEdge", "D.clipEdge");
    }
    for (const out of new Set([...outputs, "G.posW"])) g.markOutput(out);
    if (o.external) g.setInput("G.clipBuffer", o.external);
    g.setScene(scene);
    const setTime = (t: number) => g.getPass("DC")!.setProperties(new Properties({ time: t }));
    for (const t of o.times ?? []) {
        setTime(t);
        g.execute(device.renderContext);
    }
    if (o.control) setTime(Number(o.control["time"] ?? 0));
    for (let tries = 0; ; tries++) {
        g.execute(device.renderContext);
        const pos = await read(device, g.getOutput("G.posW")!);
        let hits = 0;
        for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
        if (hits > 0.2 * kSize * kSize) break;
        if (tries >= 4) throw new Error(`G-buffer still empty after ${tries + 1} frames`);
    }
    g.execute(device.renderContext);
    const res: Record<string, Float32Array> = {};
    for (const out of new Set([...outputs, "G.posW"])) res[out] = (await read(device, g.getOutput(out)!)).slice();
    return res;
}

const instanceAt = (ids: Float32Array, i: number) => (ids[i * 4 + 3] !== 0 ? ids[i * 4 + 1]! : -1);
const scalarAt = (a: Float32Array, i: number) => a[i * (a.length / (kSize * kSize))]!;

function countVisible(ids: Float32Array, set: number[]): number {
    let n = 0;
    for (let i = 0; i < kSize * kSize; i++) if (set.includes(instanceAt(ids, i))) n++;
    return n;
}

gpuTest("Dissolve.controlMatchesCpuEntries", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    // Entry formula: instance at list position j, progress saturate((time - start - j * stagger) / duration).
    const e = dissolveClipEntries({ ...kControl, reverse: false });
    let formulaErr = 0;
    kControl.instances.forEach((id, j) => {
        const p = Math.min(Math.max((kControl.time - kControl.startTime - j * kControl.stagger) / kControl.duration, 0), 1);
        formulaErr = Math.max(formulaErr, Math.abs(e[id * 4]! - Math.fround(p)), Math.abs(e[id * 4 + 1]! - 2), Math.abs(e[id * 4 + 2]! - Math.fround(0.1)), Math.abs(e[id * 4 + 3]! - 3));
    });
    expectEq(e.length, 11 * 4, "one entry per instance id up to the largest listed");
    expectEq(e[0], 0, "unlisted instance 0 not clipped");
    expectEq(formulaErr < 1e-7, true, "entries follow the time formula");
    const outs = ["G.vbuffer", "G.normW", "G.depth", "G.clipEdge"];
    for (const t of [0.4, 1.0, 1.7]) {
        const viaPass = await run(device, scene, outs, { control: { ...kControl, time: t } });
        const buf = new Buffer(device, { size: 11 * 16, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "DissolveTest" });
        buf.setBlob(dissolveClipEntries({ ...kControl, time: t, reverse: false }));
        const viaCpu = await run(device, scene, outs, { external: buf });
        const differing = outs.filter((o) => !bitwiseEqual(viaPass[o]!, viaCpu[o]!));
        log(`control t=${t}: differing [${differing.join(",")}]`);
        expectEq(differing.length, 0, "DissolveControl buffer = CPU entries (bitwise G-buffer)");
    }
});

gpuTest("Dissolve.compositeMatchesCpu", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const dp = { edgeColour: [6, 2.4, 0.6], emberColour: [1.2, 0.15, 0.02], edgeIntensity: 1.5, edgeSharpness: 1.5, charAmount: 0.85 };
    const r = await run(device, scene, ["L.color", "G.clipEdge", "D.dst"], { control: { ...kControl, time: 1.2 }, dissolve: dp });
    const c = r["L.color"]!, e = r["G.clipEdge"]!, d = r["D.dst"]!;
    let maxRel = 0, edgePx = 0, copyBad = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        const ev = scalarAt(e, i);
        if (!(ev > 0)) {
            for (let k = 0; k < 4; k++) if (Object.is(c[i * 4 + k], d[i * 4 + k]) === false) copyBad++;
            continue;
        }
        edgePx++;
        const w = ev * ev, s = Math.pow(ev, dp.edgeSharpness);
        for (let k = 0; k < 3; k++) {
            const glow = (dp.emberColour[k]! + (dp.edgeColour[k]! - dp.emberColour[k]!) * w) * s;
            const want = c[i * 4 + k]! * (1 - dp.charAmount * ev) + dp.edgeIntensity * glow;
            maxRel = Math.max(maxRel, Math.abs(d[i * 4 + k]! - want) / Math.max(Math.abs(want), 1e-3));
        }
    }
    log(`composite: edge px ${edgePx}, max rel err ${maxRel.toExponential(2)}, non-edge px not copied ${copyBad}`);
    expectEq(edgePx > 50, true, "edge present");
    expectEq(maxRel < 1e-4, true, "Dissolve = CPU formula");
    expectEq(copyBad, 0, "e = 0 copies the colour");
});

gpuTest("Dissolve.identity", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const plain = await run(device, scene, ["L.color", "G.vbuffer"], { control: null, dissolve: null });
    const zero = await run(device, scene, ["L.color", "G.vbuffer", "D.dst"], { control: { ...kControl, threshold: 0 }, dissolve: {} });
    const a = bitwiseEqual(plain["L.color"]!, zero["D.dst"]!), b = bitwiseEqual(plain["G.vbuffer"]!, zero["G.vbuffer"]!);
    const noGlow = await run(device, scene, ["L.color", "D.dst", "G.clipEdge"], { control: { ...kControl, time: 1.2 }, dissolve: { edgeIntensity: 0, charAmount: 0 } });
    const c = bitwiseEqual(noGlow["L.color"]!, noGlow["D.dst"]!);
    const edgeSeen = noGlow["G.clipEdge"]!.some((v) => v > 0);
    log(`identity: threshold 0 dst == plain lit ${a}, vbuffer == plain ${b}; edgeIntensity 0 + charAmount 0 dst == lit ${c} (edge present ${edgeSeen})`);
    expectEq(a && b, true, "threshold 0 = plain chain (bitwise)");
    expectEq(c && edgeSeen, true, "no glow/char = lit colour (bitwise)");
});

gpuTest("Dissolve.timeProgress", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const set = [...kBoxA, ...kBoxB];
    const base = countVisible((await run(device, scene, ["S.id"], { control: { ...kControl, threshold: 0 } }))["S.id"]!, set);
    const fwd: number[] = [], rev: number[] = [];
    for (const t of [0, 0.5, 1, 1.5, 2, 3.1]) {
        fwd.push(countVisible((await run(device, scene, ["S.id"], { control: { ...kControl, time: t } }))["S.id"]!, set));
        rev.push(countVisible((await run(device, scene, ["S.id"], { control: { ...kControl, time: t, reverse: true } }))["S.id"]!, set));
    }
    log(`time: base ${base}, forward [${fwd.join(",")}], reverse [${rev.join(",")}]`);
    const mono = fwd.every((v, i) => i === 0 || v <= fwd[i - 1]!) && rev.every((v, i) => i === 0 || v >= rev[i - 1]!);
    expectEq(fwd[0], base, "t = start: nothing dissolved");
    expectEq(fwd[fwd.length - 1], 0, "after the end: all listed objects gone");
    expectEq(rev[0], 0, "reverse at start: hidden");
    expectEq(rev[rev.length - 1], base, "reverse after the end: fully back");
    expectEq(mono, true, "monotone in time");
});

gpuTest("Dissolve.stateless", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const outs = ["D.dst", "G.vbuffer", "G.clipEdge"];
    const X = { ...kControl, time: 1.1 };
    const cold = await run(device, scene, outs, { control: X, dissolve: {} });
    const warm = await run(device, scene, outs, { control: X, dissolve: {}, times: [0.3, 1.9, 0.7] });
    const differing = outs.filter((o) => !bitwiseEqual(cold[o]!, warm[o]!));
    log(`stateless: cold vs after other times differing [${differing.join(",")}]`);
    expectEq(differing.length, 0, "cold == consecutive (bitwise)");
});
