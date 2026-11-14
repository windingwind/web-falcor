/**
 * Per-node animation time quantisation (Scene.setAnimationTimeQuantization): a floor and two boxes, box A on animated
 * node 0 and box B on animated node 1 (both slide). Checks:
 *  - default: no quantisation, an empty map and a hold of 0 render bitwise the same posW / mvec / vbuffer frames
 *    (and the frame hash is logged so it can be compared against a checkout without the feature);
 *  - "on twos": with both nodes quantised, frame k equals the unquantised frame 2*floor(k/2) bitwise (posW), held
 *    frames repeat the previous frame bitwise and boundary frames move;
 *  - selectivity: quantising node 0 leaves box B's pixels exactly as in the unquantised render.
 */

import { RenderGraph, Scene, createPass, float2, float3, float4, quatf, LightType, type Device } from "@web-falcor/falcor";
import type { AnimationChannel, SceneNode } from "../../../packages/falcor/src/Scene/Animation/SceneAnimation.js";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

type V3 = [number, number, number];
const kSize = 96;
const kFps = 24;
const kFrames = 8;

function quad(c: V3[], n: V3, materialID: number, nodeID?: number) {
    const uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
    const t = new float4(1, 0, 0, 1);
    return {
        vertices: c.map((p, i) => ({ position: new float3(...p), normal: new float3(...n), tangent: t, texCrd: uv[i]! })),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        materialID,
        ...(nodeID !== undefined ? { nodeID } : {}),
    };
}

/** Box in node-local space (5 faces, open bottom). */
function box(min: V3, max: V3, materialID: number, nodeID: number) {
    const [x0, y0, z0] = min, [x1, y1, z1] = max;
    return [
        quad([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0, 1, 0], materialID, nodeID),
        quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1], materialID, nodeID),
        quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0, 0, -1], materialID, nodeID),
        quad([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1, 0, 0], materialID, nodeID),
        quad([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1, 0, 0], materialID, nodeID),
    ];
}

const slide = (nodeID: number, from: V3, to: V3): AnimationChannel => ({
    nodeID, path: "translation", interp: "LINEAR", times: new Float32Array([0, 1]), values: new Float32Array([...from, ...to]),
});

/** A new scene (nodes 0 and 1 slide). */
function makeScene(device: Device): Scene {
    const s = 5;
    const nodes: SceneNode[] = [
        { parent: -1, t: new float3(-1.5, 0, 0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
        { parent: -1, t: new float3(0.8, 0, -1.2), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
    ];
    const meshes = [
        quad([[-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]], [0, 1, 0], 0),
        ...box([-0.4, 0, -0.4], [0.4, 0.9, 0.4], 1, 0),
        ...box([-0.35, 0, -0.35], [0.35, 1.3, 0.35], 2, 1),
    ];
    const channels = [slide(0, [-1.5, 0, 0], [0.5, 0, 0.6]), slide(1, [0.8, 0, -1.2], [1.8, 0, -0.2])];
    const mats = ([[0.7, 0.7, 0.7], [0.8, 0.2, 0.1], [0.1, 0.3, 0.8]] as V3[]).map((c) => ({ basic: { baseColor: new float4(c[0], c[1], c[2], 1) } }));
    const scene = new Scene(device, meshes, mats, [{ type: LightType.Directional, dirW: new float3(0.3, -1, 0.2), intensity: new float3(2, 2, 2) }],
        undefined, [], nodes, channels);
    scene.camera.setPosition(new float3(0.5, 4.2, 5.0));
    scene.camera.setTarget(new float3(0, 0.3, -0.3));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
    return scene;
}

let graphCount = 0;
async function read(device: Device, t: import("@web-falcor/falcor").Texture): Promise<Float32Array> {
    const raw = await device.renderContext.readTextureSubresource(t);
    return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice();
}

/** Renders frames 0..kFrames-1 at t = k / fps (after one warm-up frame at t = 0); returns posW and mvec per frame. */
async function renderClip(device: Device, scene: Scene): Promise<{ posW: Float32Array[]; mvec: Float32Array[] }> {
    const g = new RenderGraph(device, `AnimTimeQuant${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    g.markOutput("G.posW");
    g.markOutput("G.mvec");
    g.setScene(scene);
    scene.animate(0);
    g.execute(device.renderContext);
    const pos = await read(device, g.getOutput("G.posW")!);
    let hits = 0;
    for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
    if (hits <= 0.2 * kSize * kSize) throw new Error(`G-buffer empty on the warm-up frame (${hits} px)`);
    const posW: Float32Array[] = [], mvec: Float32Array[] = [];
    for (let k = 0; k < kFrames; k++) {
        scene.animate(k / kFps);
        g.execute(device.renderContext);
        posW.push(await read(device, g.getOutput("G.posW")!));
        mvec.push(await read(device, g.getOutput("G.mvec")!));
    }
    return { posW, mvec };
}

function bitwiseEqual(a: Float32Array, b: Float32Array): boolean {
    if (a.length !== b.length) return false;
    const ua = new Uint32Array(a.buffer, a.byteOffset, a.length), ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
    for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
    return true;
}

/** FNV-1a over the bytes of all frames (logged, for comparison with a checkout without the feature). */
function fnv(frames: Float32Array[]): string {
    let h = 0x811c9dc5;
    for (const f of frames) {
        const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
        for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i]!, 16777619) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
}

type QuantScene = Scene & { setAnimationTimeQuantization?: (q: Record<number, { hold: number; phase?: number }> | null) => void };

gpuTest("AnimTimeQuant.defaultIsUnchanged", async ({ device }) => {
    const scene = makeScene(device) as QuantScene;
    const base = await renderClip(device, scene);
    console.log(`# AnimTimeQuant default hash posW ${fnv(base.posW)} mvec ${fnv(base.mvec)}`);
    let moving = 0;
    for (let k = 1; k < kFrames; k++) if (!bitwiseEqual(base.posW[k]!, base.posW[k - 1]!)) moving++;
    expectEq(moving, kFrames - 1, "boxes move on every frame without quantisation");
    if (typeof scene.setAnimationTimeQuantization !== "function") return; // checkout without the feature: hash only
    for (const q of [null, {}, { 0: { hold: 0 } }, { 1: { hold: -1 } }]) {
        scene.setAnimationTimeQuantization(q);
        const r = await renderClip(device, scene);
        const same = r.posW.every((f, k) => bitwiseEqual(f, base.posW[k]!)) && r.mvec.every((f, k) => bitwiseEqual(f, base.mvec[k]!));
        console.log(`# AnimTimeQuant default ${JSON.stringify(q)}: bitwise ${same}`);
        expectEq(same, true, `quantisation ${JSON.stringify(q)} renders bitwise as without it`);
    }
    scene.setAnimationTimeQuantization(null);
});

gpuTest("AnimTimeQuant.onTwosHoldsPose", async ({ device }) => {
    const scene = makeScene(device) as QuantScene;
    expectEq(typeof scene.setAnimationTimeQuantization, "function", "Scene.setAnimationTimeQuantization exists");
    scene.setAnimationTimeQuantization!(null);
    const full = await renderClip(device, scene);
    scene.setAnimationTimeQuantization!({ 0: { hold: 2 / kFps }, 1: { hold: 2 / kFps } });
    const twos = await renderClip(device, scene);
    scene.setAnimationTimeQuantization!(null);
    for (let k = 0; k < kFrames; k++) {
        const held = 2 * Math.floor(k / 2);
        const posOk = bitwiseEqual(twos.posW[k]!, full.posW[held]!);
        // Held frames repeat the previous pose exactly; boundary frames move. (Motion vectors are logged for information.)
        const sameAsPrev = k > 0 && bitwiseEqual(twos.posW[k]!, twos.posW[k - 1]!);
        const maxPx = twos.mvec[k]!.reduce((m, v) => Math.max(m, Math.abs(v) * kSize), 0);
        const fullPx = full.mvec[k]!.reduce((m, v) => Math.max(m, Math.abs(v) * kSize), 0);
        console.log(`# AnimTimeQuant twos frame ${k}: posW == full frame ${held} ${posOk}, == previous frame ${sameAsPrev}; max |mvec| ${maxPx.toExponential(3)} px (unquantised ${fullPx.toExponential(3)} px)`);
        expectEq(posOk, true, `frame ${k} shows the pose of frame ${held}`);
        if (k > 0) expectEq(sameAsPrev, k % 2 === 1, `frame ${k}: pose changes only on hold boundaries`);
    }
});

gpuTest("AnimTimeQuant.onlySelectedNode", async ({ device }) => {
    const scene = makeScene(device) as QuantScene;
    scene.setAnimationTimeQuantization!(null);
    const full = await renderClip(device, scene);
    scene.setAnimationTimeQuantization!({ 0: { hold: 3 / kFps } });
    const q = await renderClip(device, scene);
    scene.setAnimationTimeQuantization!(null);
    // Box B is the only geometry above y = 0.95 (box A is 0.9 high): pixels showing it in both renders must match.
    let bBad = 0, bN = 0, aDiff = 0;
    for (let k = 0; k < kFrames; k++) {
        const a = q.posW[k]!, b = full.posW[k]!;
        for (let i = 0; i < kSize * kSize; i++) {
            const onB = (p: Float32Array) => p[i * 4 + 3] !== 0 && p[i * 4 + 1]! > 0.95;
            if (onB(a) && onB(b)) {
                bN++;
                if (a[i * 4] !== b[i * 4] || a[i * 4 + 1] !== b[i * 4 + 1] || a[i * 4 + 2] !== b[i * 4 + 2]) bBad++;
            }
        }
        if (k % 3 !== 0 && !bitwiseEqual(a, b)) aDiff++;
    }
    console.log(`# AnimTimeQuant selectivity: box-B pixels ${bN}, differing ${bBad}; frames where box A differs ${aDiff}`);
    expectEq(bN > 100, true, "box B visible");
    expectEq(bBad, 0, "unquantised node renders exactly as without quantisation");
    expectEq(aDiff > 0, true, "quantised node holds its pose");
});
