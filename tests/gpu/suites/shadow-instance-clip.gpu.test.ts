/**
 * Shadows follow GBufferRaster's per-instance clip: optional `clipBuffer` inputs of CascadedShadowMap (shadow-map
 * casters) and DeferredLighting (ray-traced visibility rays).
 *  1. defaultHash: hashes of the lit colour, the visibility output, the shadow map and the generated WGSL of
 *     DeferredLighting / CascadedShadowMap without a clip buffer (ShadowMap and RayTraced). Run on this branch and on the
 *     base commit; the hashes must be equal.
 *  2. A bound clip buffer with all thresholds 0 leaves every output bitwise unchanged (both shadow modes).
 *  3. Shadow fades with the object: box B is clipped out of the G-buffer; its shadow on the floor/box A shrinks
 *     monotonically as the shadow clip threshold rises and is gone at threshold > 1 (ShadowMap and RayTraced); without
 *     the shadow clip it stays (control). RayTraced dither mode is exact: in B's shadow, visibility = 1 exactly where
 *     the receiving pixel's 4x4 Bayer value is below the threshold; outside B's shadow nothing changes bit for bit.
 *  4. No state: the same clip buffer gives bitwise the same outputs on a fresh graph and after other clip settings.
 */

import { Buffer, MemoryType, ResourceBindFlags, type Device, type RenderGraph, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kSize, log, makeScene, read, type PassDecl, type V3 } from "./stylize-helpers.js";
import { buildGraph } from "./fx-helpers.js";

/** Camera showing both boxes and their shadows on the floor (light dir (0.4, -1, 0.3)). */
const kCam = { pos: [2.6, 2.4, 3.4] as V3, target: [0.2, 0.3, -0.6] as V3 };
const kInstances = 11; // floor 0, box A 1-5, box B 6-10
const kBoxB = [6, 7, 8, 9, 10];
type ClipEntry = [threshold: number, mode: number, edgeWidth: number, scale: number];
type ShadowMode = "ShadowMap" | "RayTraced";

function clipBuffer(device: Device, entries: ClipEntry[]): Buffer {
    const b = new Buffer(device, { size: entries.length * 16, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "ShadowClipTest" });
    b.setBlob(new Float32Array(entries.flat()));
    return b;
}
const onB = (e: ClipEntry): ClipEntry[] => Array.from({ length: kInstances }, (_, i) => (kBoxB.includes(i) ? e : [0, 0, 0, 0]));

function graph(device: Device, scene: Scene, mode: ShadowMode): RenderGraph {
    const passes: PassDecl[] = [["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: mode }]];
    const edges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"]];
    const outs = ["L.color", "L.visibility"];
    if (mode === "ShadowMap") {
        passes.unshift(["CascadedShadowMap", "CSM", { mapSize: 1024 }]);
        edges.push(["CSM.shadowMap", "L.shadowMap"]);
        outs.push("CSM.shadowMap");
    }
    return buildGraph(device, scene, passes, edges, outs, kSize, kSize);
}

/** Binds the G-buffer clip and the shadow clip (CSM / DeferredLighting) buffers (undefined = leave unbound). */
function bind(g: RenderGraph, mode: ShadowMode, gb?: Buffer, shadow?: Buffer): void {
    if (gb) g.setInput("G.clipBuffer", gb);
    if (shadow) g.setInput(mode === "ShadowMap" ? "CSM.clipBuffer" : "L.clipBuffer", shadow);
}

async function render(device: Device, g: RenderGraph, mode: ShadowMode, frames = 2): Promise<Record<string, Float32Array>> {
    for (let i = 0; i < frames; i++) g.execute(device.renderContext);
    const r: Record<string, Float32Array> = {};
    for (const o of ["L.color", "L.visibility", "G.posW", ...(mode === "ShadowMap" ? ["CSM.shadowMap"] : [])]) r[o] = await read(device, g.getOutput(o)!);
    return r;
}

/** One float per pixel from a readback of an R32Float target (handles padded/expanded readbacks). */
const scalarAt = (a: Float32Array, i: number) => a[i * (a.length / (kSize * kSize))]!;
const kBayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
const bayer = (x: number, y: number) => Math.fround((kBayer[(y & 3) * 4 + (x & 3)]! + 0.5) / 16);

function fnv(a: Float32Array): string {
    const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    let h = 0x811c9dc5;
    for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i]!, 0x01000193) >>> 0;
    return h.toString(16).padStart(8, "0");
}
function fnvStr(s: string): string {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
    return `${h.toString(16).padStart(8, "0")}:${s.length}`;
}

/** WGSL of the compiled DeferredLighting kernel and CascadedShadowMap vertex/fragment kernels. */
function wgslHashes(g: RenderGraph, mode: ShadowMode): string {
    const L = g.getPass("L") as unknown as { pass: { kernel: { wgsl: string } } };
    let s = `DL=${fnvStr(L.pass.kernel.wgsl)}`;
    if (mode === "ShadowMap") {
        const C = g.getPass("CSM") as unknown as { state: { vertexKernel: { wgsl: string }; fragmentKernel: { wgsl: string } } };
        s += ` CSM=${fnvStr(C.state.vertexKernel.wgsl)}/${fnvStr(C.state.fragmentKernel.wgsl)}`;
    }
    return s;
}

const kModes: ShadowMode[] = ["ShadowMap", "RayTraced"];

gpuTest("ShadowClip.defaultHash", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    for (const mode of kModes) {
        const g = graph(device, scene, mode);
        const r = await render(device, g, mode, 3);
        log(`defaultHash ShadowClip ${mode} ${Object.entries(r).map(([k, v]) => `${k}=${fnv(v)}`).join(" ")}`);
        log(`defaultHash wgsl ${mode} ${wgslHashes(g, mode)}`);
    }
    expectEq(true, true, "hashes printed");
});

gpuTest("ShadowClip.zeroThresholdIsIdentity", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const res: string[] = [];
    let ok = true;
    for (const mode of kModes) {
        const base = await render(device, graph(device, scene, mode), mode);
        const zero = Array.from({ length: kInstances }, (): ClipEntry => [0, 2, 0.08, 6]);
        const g = graph(device, scene, mode);
        const buf = clipBuffer(device, zero);
        bind(g, mode, buf, buf);
        const r = await render(device, g, mode);
        const eq = Object.keys(base).every((k) => bitwiseEqual(base[k]!, r[k]!));
        res.push(`${mode} ${eq}`);
        ok &&= eq;
    }
    log(`ShadowClip: all thresholds 0 bound to G-buffer + shadows = unbound, bitwise: ${res.join(", ")}`);
    expectEq(ok, true, "zero thresholds change nothing");
});

gpuTest("ShadowClip.shadowFadesWithObject", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const gone = clipBuffer(device, onB([1.01, 2, 0.08, 6])); // box B fully clipped out of the G-buffer
    const thresholds = [0, 0.25, 0.5, 0.75, 1.01];
    let allOk = true;
    for (const mode of kModes) {
        const run = async (shadow?: Buffer) => {
            const g = graph(device, scene, mode);
            bind(g, mode, gone, shadow);
            return render(device, g, mode);
        };
        const vis = (r: Record<string, Float32Array>, i: number) => scalarAt(r["L.visibility"]!, i);
        const noClip = await run();
        const full = await run(gone);
        // B's shadow: shadowed without the shadow clip, lit with B fully clipped from the shadow path.
        const R: number[] = [];
        for (let i = 0; i < kSize * kSize; i++) if (noClip["G.posW"]![i * 4 + 3] !== 0 && vis(noClip, i) < 0.5 && vis(full, i) > 0.5) R.push(i);
        const inR = new Uint8Array(kSize * kSize);
        for (const i of R) inR[i] = 1;
        const frac: number[] = [];
        let outsideChanged = 0;
        for (const t of thresholds) {
            const r = await run(clipBuffer(device, onB([t, 2, 0.08, 6])));
            frac.push(R.reduce((s, i) => s + (1 - vis(r, i)), 0) / Math.max(R.length, 1));
            for (let i = 0; i < kSize * kSize; i++) if (!inR[i] && vis(r, i) !== vis(noClip, i)) outsideChanged++;
        }
        const monotone = frac.every((f, k) => k === 0 || f < frac[k - 1]!);
        const control = R.reduce((s, i) => s + (1 - vis(noClip, i)), 0) / Math.max(R.length, 1);
        let line = `ShadowClip ${mode}: B-shadow px ${R.length}; shadowed share at threshold ${thresholds.join("/")}: ${frac.map((f) => f.toFixed(3)).join("/")}; px changed outside B's shadow ${outsideChanged}; control (G-buffer clip only) share ${control.toFixed(3)}`;
        const last = frac[frac.length - 1]!;
        let ok = R.length > 150 && frac[0] === control && control > 0.5 && monotone && (mode === "RayTraced" ? last === 0 && control === 1 : last < 0.05);
        if (mode === "RayTraced") {
            ok &&= outsideChanged === 0;
            const d = await run(clipBuffer(device, onB([0.5, 1, 0, 1])));
            let mismatch = 0;
            for (const i of R) {
                const x = i % kSize, y = Math.floor(i / kSize);
                if (vis(d, i) !== (bayer(x, y) < 0.5 ? 1 : 0)) mismatch++;
            }
            line += `; dither 0.5 vs CPU Bayer mismatches ${mismatch}/${R.length}`;
            ok &&= mismatch === 0;
        }
        log(line);
        allOk &&= ok;
    }
    expectEq(allOk, true, "B's shadow fades monotonically with the clip threshold and vanishes at > 1 (RayTraced: exact dither, nothing else changes)");
});

gpuTest("ShadowClip.stateless", async ({ device }) => {
    const scene = makeScene(device, null, kCam);
    const res: string[] = [];
    let ok = true;
    for (const mode of kModes) {
        const X = clipBuffer(device, onB([0.5, 2, 0.08, 6]));
        const cold = graph(device, scene, mode);
        bind(cold, mode, X, X);
        const a = await render(device, cold, mode);
        const warm = graph(device, scene, mode);
        for (const t of [0.25, 0.9]) {
            const b = clipBuffer(device, onB([t, 2, 0.08, 6]));
            bind(warm, mode, b, b);
            await render(device, warm, mode, 1);
        }
        bind(warm, mode, X, X);
        const b = await render(device, warm, mode);
        const eq = Object.keys(a).every((k) => bitwiseEqual(a[k]!, b[k]!));
        res.push(`${mode} ${eq}`);
        ok &&= eq;
    }
    log(`ShadowClip cold vs after other clip settings, bitwise: ${res.join(", ")}`);
    expectEq(ok, true, "no state");
});
