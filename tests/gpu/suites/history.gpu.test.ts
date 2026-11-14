/**
 * History helpers (Utils/History.ts) and their example user, TemporalBlend:
 *  - HistoryTracker: reset reasons and frame index for fresh, static, moving, primed-cut, Camera.resetHistory,
 *    explicit and key-change frames;
 *  - PingPongTextures: after a reset both textures are bitwise equal to a newly created pair (all zero);
 *  - TemporalBlend: CPU parity of one blended frame, alpha 0 = bitwise pass-through, and cold vs consecutive: shot X
 *    rendered on a fresh graph equals X rendered after shots A and B on one graph, bit for bit, for every cut signal;
 *    without a cut X differs (the test is sensitive to leaked history).
 */

import { ResourceFormat } from "@web-falcor/falcor";
import { HistoryTracker, PingPongTextures } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, kSize, log, makeScene, read } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, kPoseA, kPoseX, renderShot, setPose, warmUp } from "./fx-helpers.js";

gpuTest("History.trackerResets", async ({ device }) => {
    const scene = makeScene(device);
    const cam = scene.camera;
    const t = new HistoryTracker();
    const seq: string[] = [];
    const step = (label: string, key = "k") => {
        const f = t.begin(scene, key);
        seq.push(`${label}:${f.frameIndex}${f.reset ? `/${f.reason}` : ""}`);
        return f;
    };
    cam.beginFrame();
    const f0 = step("fresh");
    cam.beginFrame();
    const f1 = step("static");
    setPose(scene, kPoseA, 0.1);
    cam.beginFrame();
    const f2 = step("moving");
    // Primed cut: the camera jumps to X and another render (one beginFrame) runs before this pass sees X.
    setPose(scene, kPoseX);
    cam.beginFrame();
    cam.beginFrame();
    const f3 = step("primed");
    cam.beginFrame();
    const f4 = step("static");
    cam.resetHistory();
    cam.beginFrame();
    const f5 = step("cameraReset");
    t.reset();
    cam.beginFrame();
    const f6 = step("explicit");
    cam.beginFrame();
    const f7 = step("key", "k2");
    // A jump straight to a new pose without a prime or reset looks like motion (documented behaviour).
    setPose(scene, kPoseA);
    cam.beginFrame();
    const f8 = step("jump", "k2");
    log(`HistoryTracker: ${seq.join(" ")}`);
    expectEq([f0.reason, f1.reason, f2.reason, f3.reason, f4.reason, f5.reason, f6.reason, f7.reason, f8.reason].join(","),
        "fresh,,,camera-cut,,camera-reset,explicit,key,", "reset reasons");
    expectEq([f0, f1, f2, f3, f4, f5, f6, f7, f8].map((f) => f.frameIndex).join(","), "0,1,2,0,1,0,0,0,1", "frame indices");

});

gpuTest("History.pingPongReset", async ({ device }) => {
    // PingPongTextures: dirty both textures, reset, compare with a fresh pair.
    const W = 16, H = 8;
    const ctx = device.renderContext;
    const pp = new PingPongTextures(device, ResourceFormat.RGBA32Float, "test");
    const a = pp.get(ctx, W, H, { frameIndex: 0, reset: true, reason: "fresh" });
    const b = pp.get(ctx, W, H, { frameIndex: 1, reset: false, reason: "" });
    expectEq(a.write === b.read && a.read === b.write, true, "write/read alternate by frame index");
    const junk = new Float32Array(W * H * 4).map((_, i) => (i % 7) + 0.5);
    // Submit the recorded clears first: queue writes are not ordered after commands still in the open encoder.
    await read(device, pp.textures![0]!);
    for (const tex of pp.textures!) tex.setSubresourceBlob(0, 0, junk);
    const dirty = await read(device, pp.textures![0]!);
    const c = pp.get(ctx, W, H, { frameIndex: 0, reset: true, reason: "explicit" });
    const fresh = new PingPongTextures(device, ResourceFormat.RGBA32Float, "fresh");
    const d = fresh.get(ctx, W, H, { frameIndex: 0, reset: true, reason: "fresh" });
    const after = [await read(device, c.read), await read(device, c.write)];
    const ref = [await read(device, d.read), await read(device, d.write)];
    const zero = after.every((x) => x.every((v) => v === 0));
    log(`PingPongTextures: dirty sample ${dirty[1]}, after reset all zero ${zero}, equal to fresh pair ${bitwiseEqual(after[0]!, ref[0]!) && bitwiseEqual(after[1]!, ref[1]!)}`);
    expectEq(dirty[1], 1.5, "textures were dirtied");
    expectEq(zero, true, "reset clears both textures");
    expectEq(bitwiseEqual(after[0]!, ref[0]!) && bitwiseEqual(after[1]!, ref[1]!), true, "state after reset = fresh pair, bit for bit");
});

const kTbOut = ["T.dst", "G.diffuseOpacity", "G.mvec"];
const tbGraph = (device: Parameters<typeof buildGraph>[0], scene: Parameters<typeof buildGraph>[1], alpha: number) =>
    buildGraph(device, scene, [["TemporalBlend", "T", { alpha }]], [["G.diffuseOpacity", "T.src"], ["G.mvec", "T.mvec"]], kTbOut, kSize, kSize);

gpuTest("TemporalBlend.parityAndIdentity", async ({ device }) => {
    const scene = makeScene(device);
    await warmUp(device, scene);
    setPose(scene, kPoseA);
    scene.camera.resetHistory();
    const alpha = 0.75;
    const g = tbGraph(device, scene, alpha);
    const frames = await renderShot(device, scene, g, kPoseA, 3, kTbOut, 0.08);
    // Frame 0: no history -> output = input bit for bit.
    expectEq(bitwiseEqual(frames[0]![0]!, frames[0]![1]!), true, "frame 0 after a reset = input");
    // Frame 2 vs CPU: lerp(src, prev[nearest(uv + mvec)], alpha) where on screen, else src.
    const [dst, src, mvec] = frames[2]!;
    const prev = frames[1]![0]!;
    let maxErr = 0, reproj = 0;
    for (let y = 0; y < kSize; y++) {
        for (let x = 0; x < kSize; x++) {
            const i = y * kSize + x;
            const u = (x + 0.5) / kSize + mvec![i * 2]!, v = (y + 0.5) / kSize + mvec![i * 2 + 1]!;
            for (let c = 0; c < 4; c++) {
                let r = src![i * 4 + c]!;
                if (u >= 0 && v >= 0 && u < 1 && v < 1) {
                    const qx = Math.min(Math.floor(u * kSize), kSize - 1), qy = Math.min(Math.floor(v * kSize), kSize - 1);
                    r = r + (prev[(qy * kSize + qx) * 4 + c]! - r) * alpha;
                    if (c === 0) reproj++;
                }
                maxErr = Math.max(maxErr, Math.abs(dst![i * 4 + c]! - r));
            }
        }
    }
    const moved = [...mvec!].some((m) => m !== 0);
    log(`TemporalBlend parity: max err ${maxErr.toExponential(2)}, reprojected px ${reproj}, mvec non-zero ${moved}`);
    expectEq(moved, true, "camera motion produces motion vectors");
    expectEq(maxErr < 1e-5, true, "frame 2 = CPU blend with reprojected history");

    // alpha = 0: pass-through on every frame of a moving shot.
    setPose(scene, kPoseA);
    const g0 = tbGraph(device, scene, 0);
    const pt = await renderShot(device, scene, g0, kPoseA, 3, kTbOut, 0.08);
    const same = pt.every((row) => bitwiseEqual(row[0]!, row[1]!));
    log(`TemporalBlend alpha 0: bitwise pass-through on 3 frames ${same}`);
    expectEq(same, true, "alpha 0 = identity");
});

gpuTest("TemporalBlend.coldVsConsecutive", async ({ device }) => {
    const scene = makeScene(device);
    const res = await coldVsConsecutive(device, scene, () => tbGraph(device, scene, 0.8), ["T.dst"], ["camera-reset", "prime", "explicit", "none"]);
    for (const r of res) log(`TemporalBlend cold vs consecutive [${r.method}]: X frames bitwise equal ${r.equalFrames}/${r.frames}`);
    for (const r of res.filter((r) => r.method !== "none")) expectEq(r.equalFrames, r.frames, `cold = consecutive after ${r.method}`);
    expectEq(res.find((r) => r.method === "none")!.differs[0], true, "without a cut the leaked history changes X (control)");
});
