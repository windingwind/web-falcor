/**
 * Several Scenes on one device: each new Scene must render its own geometry in GBufferRaster on its first frame,
 * whatever was created and rendered on the device before it.
 *  - staticThenAnimated: a static scene, then an animated scene (meshes on animated scene-graph nodes), twice (the
 *    second time all programs are cached): the animated meshes are present, move, and the G-buffer is not empty;
 *  - twoStaticScenes: two static scenes with the same content (same scene defines, so every program comes from the
 *    program cache) and a third with different content: every G-buffer is non-empty, the twin renders bitwise like the
 *    first scene, and the third covers exactly its own objects.
 */

import { LightType, RenderGraph, Scene, createPass, float2, float3, float4, quatf, type Device } from "@web-falcor/falcor";
import type { AnimationChannel, SceneNode } from "../../../packages/falcor/src/Scene/Animation/SceneAnimation.js";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

type V3 = [number, number, number];
const kSize = 96;

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
function box(min: V3, max: V3, materialID: number, nodeID?: number) {
    const [x0, y0, z0] = min, [x1, y1, z1] = max;
    return [
        quad([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0, 1, 0], materialID, nodeID),
        quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1], materialID, nodeID),
        quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0, 0, -1], materialID, nodeID),
        quad([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1, 0, 0], materialID, nodeID),
        quad([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1, 0, 0], materialID, nodeID),
    ];
}
const kMats = ([[0.6, 0.65, 0.6], [0.85, 0.3, 0.1], [0.15, 0.35, 0.85]] as V3[]).map((c) => ({ basic: { baseColor: new float4(c[0], c[1], c[2], 1) } }));
const kLight = () => [{ type: LightType.Directional, dirW: new float3(0.2, -1, 0.3), intensity: new float3(2, 2, 2) }];
const kFloor = (s: number) => quad([[-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]], [0, 1, 0], 0);

/** Floor + one box (top at `h`) + optional second box; fresh vertex arrays on every call. */
function staticScene(device: Device, h: number, second: boolean): Scene {
    const meshes = [kFloor(4), ...box([-1.0, 0, -0.5], [0.0, h, 0.5], 1), ...(second ? box([0.4, 0, -1.2], [1.1, 1.6, -0.4], 2) : [])];
    return setCamera(new Scene(device, meshes, kMats, kLight()));
}

/** Floor (static) + box A on node 0 + box B (the only geometry above y = 1) on node 1; both nodes slide over 1 s. */
function animatedScene(device: Device): Scene {
    const nodes: SceneNode[] = [
        { parent: -1, t: new float3(-1.4, 0, 0.2), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
        { parent: -1, t: new float3(0.9, 0, -1.0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
    ];
    const meshes = [kFloor(4), ...box([-0.45, 0, -0.45], [0.45, 0.9, 0.45], 1, 0), ...box([-0.3, 0, -0.3], [0.3, 1.4, 0.3], 2, 1)];
    const slide = (nodeID: number, from: V3, to: V3): AnimationChannel => ({
        nodeID, path: "translation", interp: "LINEAR", times: new Float32Array([0, 1]), values: new Float32Array([...from, ...to]),
    });
    const channels = [slide(0, [-1.4, 0, 0.2], [0.6, 0, 0.7]), slide(1, [0.9, 0, -1.0], [1.7, 0, -0.1])];
    return setCamera(new Scene(device, meshes, kMats, kLight(), undefined, [], nodes, channels));
}

function setCamera(scene: Scene): Scene {
    scene.camera.setPosition(new float3(0.3, 4.0, 5.2));
    scene.camera.setTarget(new float3(0, 0.3, -0.3));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
    return scene;
}

let graphCount = 0;
/** One GBufferRaster frame of `scene` on a new graph (no retries): posW and diffuseOpacity. */
async function gbuffer(device: Device, scene: Scene): Promise<{ posW: Float32Array; diffuse: Float32Array }> {
    const g = new RenderGraph(device, `SecondSceneTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    g.markOutput("G.posW");
    g.markOutput("G.diffuseOpacity");
    g.setScene(scene);
    g.execute(device.renderContext);
    const read = async (name: string) => {
        const raw = await device.renderContext.readTextureSubresource(g.getOutput(name)!);
        return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice();
    };
    const posW = await read("G.posW");
    const diffuse = await read("G.diffuseOpacity");
    return { posW, diffuse };
}

/** Covered pixels, pixels above y = `above`, and the per-pixel max y. */
function coverage(posW: Float32Array, above: number): { px: number; high: number; maxY: number } {
    let px = 0, high = 0, maxY = -Infinity;
    for (let i = 0; i < kSize * kSize; i++) {
        if (posW[i * 4 + 3] === 0) continue;
        px++;
        const y = posW[i * 4 + 1]!;
        if (y > above) high++;
        maxY = Math.max(maxY, y);
    }
    return { px, high, maxY };
}

function bitwiseEqual(a: Float32Array, b: Float32Array): boolean {
    const ua = new Uint32Array(a.buffer, a.byteOffset, a.length), ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
    if (ua.length !== ub.length) return false;
    for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
    return true;
}

/** Number of covered pixels whose diffuse albedo is (close to) material `m`'s base colour. */
function materialPx(gb: { posW: Float32Array; diffuse: Float32Array }, m: number): number {
    const c = kMats[m]!.basic.baseColor;
    let n = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        if (gb.posW[i * 4 + 3] === 0) continue;
        const d = gb.diffuse;
        if (Math.abs(d[i * 4]! - c.x) < 0.02 && Math.abs(d[i * 4 + 1]! - c.y) < 0.02 && Math.abs(d[i * 4 + 2]! - c.z) < 0.02) n++;
    }
    return n;
}


gpuTest("SecondScene.staticThenAnimated", async ({ device }) => {
    // Two rounds of (static scene, animated scene): in the second round every program comes from the program cache.
    for (const round of [1, 2]) {
        const first = staticScene(device, 0.9, false);
        const a = await gbuffer(device, first);
        const ca = coverage(a.posW, 1.0);
        const anim = animatedScene(device);
        const frames: { t: number; c: ReturnType<typeof coverage>; a: number; b: number; posW: Float32Array }[] = [];
        for (const t of [0, 0.25, 0.5]) {
            anim.animate(t);
            const g = await gbuffer(device, anim);
            frames.push({ t, c: coverage(g.posW, 1.0), a: materialPx(g, 1), b: materialPx(g, 2), posW: g.posW });
        }
        console.log(`SecondScene.staticThenAnimated round ${round}: static px ${ca.px}; ${frames.map((f) => `t=${f.t}: px ${f.c.px}, box B px ${f.c.high}, max y ${f.c.maxY.toFixed(3)}, box A/B material px ${f.a}/${f.b}`).join("; ")}`);
        expectEq(ca.px > 0.3 * kSize * kSize, true, `round ${round}: static scene G-buffer non-empty (${ca.px} px)`);
        expectEq(ca.high, 0, `round ${round}: static scene has nothing above y = 1`);
        let moved = 0;
        for (const [k, f] of frames.entries()) {
            expectEq(f.c.px > 0.3 * kSize * kSize, true, `round ${round}: animated scene G-buffer non-empty at t=${f.t} (${f.c.px} px)`);
            expectEq(f.c.high > 50, true, `round ${round}: animated box B present at t=${f.t} (${f.c.high} px above y = 1)`);
            expectEq(Math.abs(f.c.maxY - 1.4) < 1e-3, true, `round ${round}: animated box B top at y = 1.4 at t=${f.t} (${f.c.maxY})`);
            expectEq(f.a > 50 && f.b > 50, true, `round ${round}: both animated boxes shaded with their materials at t=${f.t}`);
            if (k > 0 && !bitwiseEqual(frames[k - 1]!.posW, f.posW)) moved++;
        }
        expectEq(moved, 2, `round ${round}: the animated meshes move between frames`);
    }
});

gpuTest("SecondScene.twoStaticScenes", async ({ device }) => {
    const s1 = staticScene(device, 1.0, true);
    const g1 = await gbuffer(device, s1);
    // Same content and scene defines: every program of the second scene comes from the program cache.
    const s2 = staticScene(device, 1.0, true);
    const g2 = await gbuffer(device, s2);
    // Different content (no second box, lower box): its own geometry only.
    const s3 = staticScene(device, 0.6, false);
    const g3 = await gbuffer(device, s3);
    const [c1, c2, c3] = [g1, g2, g3].map((g) => coverage(g.posW, 1.0));
    console.log(`SecondScene.twoStaticScenes: px ${c1!.px}/${c2!.px}/${c3!.px}, above y=1 ${c1!.high}/${c2!.high}/${c3!.high}, max y ${c1!.maxY}/${c2!.maxY}/${c3!.maxY}`);
    expectEq(c1!.px > 0.3 * kSize * kSize, true, `first scene G-buffer non-empty (${c1!.px} px)`);
    expectEq(c2!.px > 0.3 * kSize * kSize, true, `second scene G-buffer non-empty (${c2!.px} px)`);
    expectEq(bitwiseEqual(g1.posW, g2.posW) && bitwiseEqual(g1.diffuse, g2.diffuse), true, "second scene renders bitwise like the first");
    expectEq(Math.abs(c1!.maxY - 1.6) < 1e-3 && c1!.high > 50, true, "first scene shows its 1.6 high box");
    expectEq(c3!.px > 0.3 * kSize * kSize, true, `third scene G-buffer non-empty (${c3!.px} px)`);
    expectEq(Math.abs(c3!.maxY - 0.6) < 1e-3 && c3!.high === 0, true, `third scene shows only its own geometry (max y ${c3!.maxY})`);
    expectEq(materialPx(g3, 2), 0, "third scene has no pixels of the second box's material");
    expectEq(materialPx(g3, 1) > 50, true, "third scene shows its box");
});
