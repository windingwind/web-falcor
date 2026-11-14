/**
 * Clouds2D against its CPU reference (evalClouds2D) for several coverage/light/wind settings; identity (coverage 0,
 * opacity 0) returns the input bit for bit; the DepthComposite route (layer + layerDepth composited behind the
 * G-buffer) matches the pass's own posW route; stateless: a shot renders the same on a fresh graph and after other
 * shots, the same `time` reproduces bitwise and a new `time` moves the clouds. Measured numbers are printed as "# ...".
 */

import { Properties, float3, type Device, type RenderGraph, type Scene } from "@web-falcor/falcor";
import { clouds2DSkyFrame, evalClouds2D, type Clouds2DParams } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, log, makeScene, read, type PassDecl, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, coverage, rgbaTexture } from "./fx-helpers.js";

const kN = 96;

function lookAt(scene: Scene, pos: V3, target: V3): void {
    scene.camera.setPosition(new float3(...pos));
    scene.camera.setTarget(new float3(...target));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
}

function viewDirs(scene: Scene, n: number): V3[] {
    const c = scene.camera;
    const p = c.getPosition(), t = c.getTarget();
    const f = [t.x - p.x, t.y - p.y, t.z - p.z]; const fl = Math.hypot(f[0]!, f[1]!, f[2]!); const fwd = f.map((v) => v / fl);
    const r0 = [-fwd[2]!, 0, fwd[0]!]; const rl = Math.hypot(r0[0]!, r0[1]!, r0[2]!); const right = r0.map((v) => v / rl);
    const up = [right[1]! * fwd[2]! - right[2]! * fwd[1]!, right[2]! * fwd[0]! - right[0]! * fwd[2]!, right[0]! * fwd[1]! - right[1]! * fwd[0]!];
    const tanY = Math.tan(0.5 * c.getFovY());
    const dirs: V3[] = [];
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const nx = ((x + 0.5) / n) * 2 - 1, ny = ((y + 0.5) / n) * 2 - 1;
        dirs.push([0, 1, 2].map((k) => fwd[k]! + nx * tanY * right[k]! - ny * tanY * up[k]!) as V3);
    }
    return dirs;
}

async function execUntilScene(device: Device, g: RenderGraph): Promise<void> {
    for (let i = 0; i < 5; i++) { g.execute(device.renderContext); if ((await coverage(device, g)) > 0.05) return; }
}

/** A smooth blue sky gradient as the background input. */
function skyTexture(device: Device) {
    const d = new Float32Array(kN * kN * 4);
    for (let y = 0; y < kN; y++) for (let x = 0; x < kN; x++) d.set([0.2 + 0.002 * x, 0.35 + 0.002 * y, 0.8, 1], (y * kN + x) * 4);
    return { tex: rgbaTexture(device, kN, kN, d), data: d };
}

gpuTest("Clouds2D.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    lookAt(scene, [0, 1, 0], [0.3, 1.35, -1]);
    const dirs = viewDirs(scene, kN);
    const sky = skyTexture(device);
    const light: V3 = [0.48, 0.6, -0.64];
    const cases: Partial<Clouds2DParams>[] = [
        { lightDir: light },
        { lightDir: light, coverage: 0.75, softness: 0.4, scale: 2, time: 12, wind: [0.05, -0.02], seed: 7 },
        { lightDir: [-0.3, 0.2, 0.9], coverage: 0.35, octaves: 7, warp: 1.5, silver: 1.5, shadeColour: [0.2, 0.2, 0.3], brightness: 0.5, opacity: 0.8 },
        { lightDir: light, upAxis: [0.25, 0.9, -0.35], coverage: 0.6 }, // tilted sky frame
    ];
    for (const params of cases) {
        const g = buildGraph(device, scene, [["Clouds2D", "K", params]], [], ["K.dst", "K.layer"], kN, kN, { "K.color": sky.tex });
        g.execute(device.renderContext);
        const dst = await read(device, g.getOutput("K.dst")!), lay = await read(device, g.getOutput("K.layer")!);
        const ln = Math.hypot(...(params.lightDir as V3));
        const toSky = clouds2DSkyFrame(params.upAxis ?? [0, 1, 0]);
        const L = toSky((params.lightDir as V3).map((v) => v / ln));
        let bad = 0, maxErr = 0, cloudy = 0;
        for (let i = 0; i < kN * kN; i++) {
            const r = evalClouds2D(toSky(dirs[i]!), params, L);
            if (r.a > 0.05) cloudy++;
            let e = Math.abs(lay[i * 4 + 3]! - r.a);
            for (let c = 0; c < 3; c++) {
                e = Math.max(e, Math.abs(lay[i * 4 + c]! - r.rgb[c]));
                e = Math.max(e, Math.abs(dst[i * 4 + c]! - (sky.data[i * 4 + c]! * (1 - r.a) + r.rgb[c])));
            }
            if (e > 1e-4) bad++;
            else maxErr = Math.max(maxErr, e);
        }
        log(`Clouds2D ${JSON.stringify(params)}: px off reference ${bad}/${kN * kN}, max err elsewhere ${maxErr.toExponential(2)}, cloudy px (a > 0.05) ${cloudy}`);
        expectEq(bad <= 0.005 * kN * kN, true, "matches CPU reference");
        expectEq(cloudy > 100 && cloudy < kN * kN, true, "partial cloud cover");
    }
});

gpuTest("Clouds2D.identityAndDepthComposite", async ({ device }) => {
    const scene = makeScene(device);
    lookAt(scene, [2.6, 1.2, 3.4], [0, 1.0, -0.3]);
    for (const [label, props] of [["coverage 0", { coverage: 0 }], ["opacity 0", { opacity: 0, coverage: 0.9 }]] as const) {
        for (const withPos of [true, false]) {
            const edges: [string, string][] = [["G.diffuseOpacity", "K.color"]];
            if (withPos) edges.push(["G.posW", "K.posW"]);
            const g = buildGraph(device, scene, [["Clouds2D", "K", props]], edges, ["K.dst", "G.diffuseOpacity"], kN, kN);
            await execUntilScene(device, g);
            const same = bitwiseEqual(await read(device, g.getOutput("K.dst")!), await read(device, g.getOutput("G.diffuseOpacity")!));
            log(`Clouds2D identity (${label}, posW ${withPos}): bitwise ${same}`);
            expectEq(same, true, `identity: ${label}`);
        }
    }
    // Route 1: the pass composites behind geometry itself (posW). Route 2: full-frame layer through DepthComposite.
    const props = { coverage: 0.7, lightDir: [0.48, 0.6, -0.64], time: 3 };
    const passes: PassDecl[] = [["Clouds2D", "K", props], ["Clouds2D", "Q", props], ["DepthComposite", "C", { alphaMode: "premultiplied" }]];
    const edges: [string, string][] = [["G.diffuseOpacity", "K.color"], ["G.posW", "K.posW"], ["G.diffuseOpacity", "C.src"], ["G.posW", "C.posW"],
        ["Q.layer", "C.layer"], ["Q.layerDepth", "C.layerDepth"]];
    const g = buildGraph(device, scene, passes, edges, ["K.dst", "C.dst", "G.diffuseOpacity", "Q.layer"], kN, kN);
    await execUntilScene(device, g);
    const a = await read(device, g.getOutput("K.dst")!), b = await read(device, g.getOutput("C.dst")!), src = await read(device, g.getOutput("G.diffuseOpacity")!);
    const pos = await read(device, g.getOutput("G.posW")!), lay = await read(device, g.getOutput("Q.layer")!);
    let maxErr = 0, sky = 0, cloudy = 0, geo = 0, geomKept = 0;
    for (let i = 0; i < kN * kN; i++) {
        for (let c = 0; c < 3; c++) maxErr = Math.max(maxErr, Math.abs(a[i * 4 + c]! - b[i * 4 + c]!));
        if (pos[i * 4 + 3] === 0) { sky++; if (lay[i * 4 + 3]! > 0.05) cloudy++; }
        else { geo++; if (a[i * 4] === src[i * 4] && a[i * 4 + 1] === src[i * 4 + 1] && a[i * 4 + 2] === src[i * 4 + 2]) geomKept++; }
    }
    log(`Clouds2D DepthComposite route: sky px ${sky} (cloudy ${cloudy}), geometry px ${geo} (kept ${geomKept}), max |posW route - DepthComposite route| ${maxErr.toExponential(2)}`);
    expectEq(sky > 200 && geo > 200 && cloudy > 20, true, "view shows clouds and geometry");
    expectEq(geomKept, geo, "geometry passes through");
    expectEq(maxErr <= 1e-6, true, "DepthComposite route = posW route");
});

gpuTest("Clouds2D.statelessAndAnimated", async ({ device }) => {
    const scene = makeScene(device);
    const props = { coverage: 0.6, time: 5, lightDir: [0.48, 0.6, -0.64] };
    const mk = () => buildGraph(device, scene, [["Clouds2D", "K", props]], [["G.diffuseOpacity", "K.color"], ["G.posW", "K.posW"]], ["K.dst"], kN, kN);
    const res = await coldVsConsecutive(device, scene, mk, ["K.dst"], ["none", "camera-reset"], 2);
    for (const r of res) log(`Clouds2D cold vs consecutive (${r.method}): ${r.equalFrames}/${r.frames} frames bitwise equal`);
    expectEq(res.every((r) => r.equalFrames === r.frames), true, "cold == consecutive (stateless)");
    lookAt(scene, [0, 1, 0], [0.3, 1.35, -1]);
    const g = buildGraph(device, scene, [["Clouds2D", "K", props]], [], ["K.dst"], kN, kN);
    const at = async (t: number) => { g.getPass("K")!.setProperties(new Properties({ time: t })); g.execute(device.renderContext); return read(device, g.getOutput("K.dst")!); };
    const f0 = await at(5), f1 = await at(15), f2 = await at(5);
    let changed = 0;
    for (let i = 0; i < kN * kN; i++) if (f0[i * 4] !== f1[i * 4]) changed++;
    log(`Clouds2D time 5 -> 15: ${changed} px changed; time 5 again bitwise ${bitwiseEqual(f0, f2)}`);
    expectEq(bitwiseEqual(f0, f2), true, "same time reproduces bitwise");
    expectEq(changed > 100, true, "wind moves the clouds");
});
