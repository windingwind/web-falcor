/**
 * ProceduralNightSky against its CPU reference (evalNightSky) for each layer and a combined overlay; identity
 * (composite "add" with no layers / skyScale 0 returns the input bit for bit); the DepthComposite route (layer +
 * layerDepth composited behind the G-buffer) equals the pass's own posW route bit for bit; stateless: a shot renders
 * the same on a fresh graph and after other shots, the same `time` reproduces bitwise and a new `time` animates.
 * Measured numbers are printed as "# ..." lines.
 */

import { Properties, float3, type Device, type RenderGraph, type Scene } from "@web-falcor/falcor";
import { createNightSkyEnvMap, evalNightSky, toSkyFrame, type NightSkyParams } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, log, makeScene, read, type PassDecl, type V3 } from "./stylize-helpers.js";
import { buildGraph, coldVsConsecutive, coverage } from "./fx-helpers.js";

const kN = 96;

function lookAt(scene: Scene, pos: V3, target: V3): void {
    scene.camera.setPosition(new float3(...pos));
    scene.camera.setTarget(new float3(...target));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
}

/** Per-pixel view directions exactly as the pass builds them (PhysicalSky convention). */
function viewDirs(scene: Scene, n: number): { dirs: V3[]; pixelAngle: number } {
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
    return { dirs, pixelAngle: (2 * tanY) / n };
}

async function execUntilScene(device: Device, g: RenderGraph): Promise<void> {
    for (let i = 0; i < 5; i++) { g.execute(device.renderContext); if ((await coverage(device, g)) > 0.05) return; }
}

gpuTest("ProceduralNightSky.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    lookAt(scene, [0, 1, 0], [0.35, 1.45, -1]); // looking up over the horizon
    const { dirs, pixelAngle } = viewDirs(scene, kN);
    const cases: Partial<NightSkyParams>[] = [
        { mode: "none" },
        { mode: "stars", time: 1.3 },
        { mode: "nebula", nebulaIntensity: 0.5 },
        { mode: "aurora", time: 3.2, auroraIntensity: 0.6 },
        { mode: "stars+nebula+aurora", composite: "add", rotation: 30, rotationSpeed: 2, time: 4, seed: 5, skyScale: 2 },
        { mode: "stars+aurora", time: 1, upAxis: [0.25, 0.9, -0.35] }, // tilted sky frame
    ];
    for (const params of cases) {
        const g = buildGraph(device, scene, [["ProceduralNightSky", "S", params]], [], ["S.dst"], kN, kN);
        g.execute(device.renderContext);
        const dst = await read(device, g.getOutput("S.dst")!);
        const add = params.composite === "add";
        const toSky = toSkyFrame(params.upAxis ?? [0, 1, 0]);
        let bad = 0, maxErr = 0, lit = 0, peak = 0;
        for (let i = 0; i < kN * kN; i++) {
            const s = evalNightSky(toSky(dirs[i]!), params, pixelAngle);
            for (let c = 0; c < 3; c++) {
                const ref = (add ? 0 : s.bg[c]) + s.layers[c];
                const e = Math.abs(dst[i * 4 + c]! - ref);
                if (e > 2e-5 + 2e-3 * Math.abs(ref)) { bad++; break; }
                maxErr = Math.max(maxErr, e);
            }
            const l = s.layers[0] + s.layers[1] + s.layers[2];
            if (l > 1e-3) lit++;
            peak = Math.max(peak, l);
        }
        log(`NightSky ${JSON.stringify(params)}: px off reference ${bad}/${kN * kN}, max err elsewhere ${maxErr.toExponential(2)}, px with layer > 1e-3: ${lit}, peak ${peak.toFixed(3)}`);
        expectEq(bad <= 0.005 * kN * kN, true, `matches CPU reference (${params.mode})`);
        if (params.mode !== "none") expectEq(lit > 20, true, `layer visible (${params.mode})`);
    }
    // Env map bake (CPU) runs and covers the same model.
    const env = createNightSkyEnvMap(device, { mode: "stars+nebula" }, 64);
    expectEq(!!env, true, "env map created");
});

gpuTest("ProceduralNightSky.identityAndDepthComposite", async ({ device }) => {
    const scene = makeScene(device);
    lookAt(scene, [2.6, 1.2, 3.4], [0, 1.0, -0.3]); // boxes against the sky
    // Identity: composite add with no layers, or skyScale 0, returns the input bit for bit (with and without posW).
    for (const [label, props, withPos] of [
        ["add none", { mode: "none", composite: "add" }, true],
        ["add none, no posW", { mode: "none", composite: "add" }, false],
        ["add skyScale 0", { mode: "stars+nebula+aurora", composite: "add", skyScale: 0 }, true],
    ] as const) {
        const edges: [string, string][] = [["G.diffuseOpacity", "S.color"]];
        if (withPos) edges.push(["G.posW", "S.posW"]);
        const g = buildGraph(device, scene, [["ProceduralNightSky", "S", props]], edges, ["S.dst", "G.diffuseOpacity"], kN, kN);
        await execUntilScene(device, g);
        const same = bitwiseEqual(await read(device, g.getOutput("S.dst")!), await read(device, g.getOutput("G.diffuseOpacity")!));
        log(`NightSky identity (${label}): bitwise ${same}`);
        expectEq(same, true, `identity: ${label}`);
    }
    // Route 1: the pass composites itself behind the geometry (posW). Route 2: full-frame layer + layerDepth through
    // DepthComposite with the scene's posW. Both must agree bit for bit on rgb.
    for (const props of [{ mode: "stars+aurora", time: 2 }, { mode: "stars+nebula", composite: "add", time: 1 }]) {
        const passes: PassDecl[] = [["ProceduralNightSky", "S", props], ["ProceduralNightSky", "T", props], ["DepthComposite", "C", { alphaMode: "premultiplied" }]];
        const edges: [string, string][] = [["G.diffuseOpacity", "S.color"], ["G.posW", "S.posW"], ["G.diffuseOpacity", "C.src"], ["G.posW", "C.posW"],
            ["T.layer", "C.layer"], ["T.layerDepth", "C.layerDepth"]];
        const g = buildGraph(device, scene, passes, edges, ["S.dst", "C.dst", "G.diffuseOpacity"], kN, kN);
        await execUntilScene(device, g);
        const a = await read(device, g.getOutput("S.dst")!), b = await read(device, g.getOutput("C.dst")!), src = await read(device, g.getOutput("G.diffuseOpacity")!);
        const pos = await read(device, g.getOutput("G.posW")!);
        let diff = 0, sky = 0, geomKept = 0, geo = 0;
        for (let i = 0; i < kN * kN; i++) {
            for (let c = 0; c < 3; c++) if (a[i * 4 + c] !== b[i * 4 + c]) { diff++; break; }
            if (pos[i * 4 + 3] === 0) sky++;
            else { geo++; if (a[i * 4] === src[i * 4] && a[i * 4 + 1] === src[i * 4 + 1] && a[i * 4 + 2] === src[i * 4 + 2]) geomKept++; }
        }
        log(`NightSky DepthComposite route ${JSON.stringify(props)}: sky px ${sky}, geometry px ${geo} (kept ${geomKept}), rgb mismatches ${diff}`);
        expectEq(sky > 200 && geo > 200, true, "view shows sky and geometry");
        expectEq(geomKept, geo, "geometry passes through");
        expectEq(diff, 0, "DepthComposite route = posW route (bitwise rgb)");
    }
});

gpuTest("ProceduralNightSky.statelessAndAnimated", async ({ device }) => {
    const scene = makeScene(device);
    const props = { mode: "stars+nebula+aurora", time: 2.5 };
    const mk = () => buildGraph(device, scene, [["ProceduralNightSky", "S", props]], [["G.diffuseOpacity", "S.color"], ["G.posW", "S.posW"]], ["S.dst"], kN, kN);
    // Stateless: with no cut signal at all, shot X after shots A and B equals X on a fresh graph.
    const res = await coldVsConsecutive(device, scene, mk, ["S.dst"], ["none", "camera-reset"], 2);
    for (const r of res) log(`NightSky cold vs consecutive (${r.method}): ${r.equalFrames}/${r.frames} frames bitwise equal`);
    expectEq(res.every((r) => r.equalFrames === r.frames), true, "cold == consecutive (stateless)");
    // time: same value reproduces bitwise, a new value changes the twinkle/aurora.
    lookAt(scene, [0, 1, 0], [0.35, 1.45, -1]);
    const g = buildGraph(device, scene, [["ProceduralNightSky", "S", props]], [], ["S.dst"], kN, kN);
    const at = async (t: number) => { g.getPass("S")!.setProperties(new Properties({ time: t })); g.execute(device.renderContext); return read(device, g.getOutput("S.dst")!); };
    const f0 = await at(2.5), f1 = await at(3.5), f2 = await at(2.5);
    let changed = 0;
    for (let i = 0; i < kN * kN; i++) if (f0[i * 4] !== f1[i * 4] || f0[i * 4 + 1] !== f1[i * 4 + 1]) changed++;
    log(`NightSky time 2.5 -> 3.5: ${changed} px changed; time 2.5 again bitwise ${bitwiseEqual(f0, f2)}`);
    expectEq(bitwiseEqual(f0, f2), true, "same time reproduces bitwise");
    expectEq(changed > 50, true, "time animates");
});
