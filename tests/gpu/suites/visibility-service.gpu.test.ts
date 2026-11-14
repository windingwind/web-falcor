/**
 * Visibility service (RasterLighting/Visibility.slangh + VisibilityService.ts) through the VisibilityQuery pass:
 *  1. ShadowMap and RayTraced modes reproduce DeferredLighting's visibility output bit for bit (same CSM, same biases);
 *  2. RayTraced agrees with a CPU ray/box shadow test on the stylize test scene (floor + two boxes, one directional light);
 *  3. ShadowMap agrees with RayTraced except along shadow edges; None is all ones; repeated execution is bitwise equal.
 */

import { RenderGraph, createPass, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kBoxA, kBoxB, kL, kSize, makeScene, read, log, bitwiseEqual, type V3 } from "./stylize-helpers.js";

const kMinCosTheta = 1e-6;
let graphCount = 0;

async function render(device: Device, scene: Scene, mode: string, dlShadow: string | null, extra: Record<string, unknown> = {}) {
    const g = new RenderGraph(device, `VisibilityTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    g.addPass(createPass(device, "CascadedShadowMap", { cascadeCount: 2, mapSize: 1024 }), "CSM");
    g.addPass(createPass(device, "VisibilityQuery", { mode, ...extra }), "V");
    g.addEdge("G.vbuffer", "V.vbuffer");
    g.addEdge("CSM.shadowMap", "V.shadowMap");
    g.markOutput("V.visibility");
    g.markOutput("G.posW");
    if (dlShadow) {
        g.addPass(createPass(device, "DeferredLighting", { brdf: "Material", shadowMode: dlShadow, envLighting: false, ...extra }), "DL");
        g.addEdge("G.vbuffer", "DL.vbuffer");
        g.addEdge("CSM.shadowMap", "DL.shadowMap");
        g.markOutput("DL.visibility");
    }
    g.setScene(scene);
    for (let tries = 0; ; tries++) {
        g.execute(device.renderContext);
        const pos = await read(device, g.getOutput("G.posW")!);
        let hits = 0;
        for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
        if (hits > 0.2 * kSize * kSize) break;
        if (tries >= 4) throw new Error("G-buffer still empty");
    }
    g.execute(device.renderContext);
    const vis = await read(device, g.getOutput("V.visibility")!);
    const pos = await read(device, g.getOutput("G.posW")!);
    const dl = dlShadow ? await read(device, g.getOutput("DL.visibility")!) : null;
    g.execute(device.renderContext);
    const again = await read(device, g.getOutput("V.visibility")!);
    return { vis, pos, dl, again };
}

/** CPU: is the ray p + t*L (t > 0) blocked by box A or B? */
function blocked(p: V3, L: V3): boolean {
    for (const [mn, mx] of [kBoxA, kBoxB]) {
        let t0 = 1e-3, t1 = Infinity;
        for (let a = 0; a < 3; a++) {
            const inv = 1 / L[a]!;
            let ta = (mn[a]! - p[a]!) * inv, tb = (mx[a]! - p[a]!) * inv;
            if (ta > tb) [ta, tb] = [tb, ta];
            t0 = Math.max(t0, ta);
            t1 = Math.min(t1, tb);
        }
        if (t0 <= t1) return true;
    }
    return false;
}

gpuTest("Visibility.matchesDeferredLighting", async ({ device }) => {
    const scene = makeScene(device, [0.02, 0.02, 0.02]); // own instance: the default shared scene can render an empty G-buffer after other suites created scenes
    for (const [mode, extra] of [["ShadowMap", {}], ["ShadowMap", { pcfRadius: 2, normalBias: 2.5 }], ["RayTraced", {}]] as const) {
        const r = await render(device, scene, mode, mode, extra);
        let n = 0, bad = 0, lit = 0, shadow = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            if (r.vis[i * 4 + 3] === 0) continue;
            const expected = r.vis[i * 4 + 1]! > kMinCosTheta ? r.vis[i * 4]! : 1; // DeferredLighting reports 1 for back-facing light
            n++;
            if (expected === 1) lit++;
            if (expected === 0) shadow++;
            if (new Uint32Array(Float32Array.of(expected).buffer)[0] !== new Uint32Array(Float32Array.of(r.dl![i]!).buffer)[0]) bad++;
        }
        log(`Visibility[${mode} ${JSON.stringify(extra)}] vs DeferredLighting: ${n} px, ${bad} differ (lit ${lit}, shadowed ${shadow}); repeat bitwise ${bitwiseEqual(r.vis, r.again)}`);
        expectEq(n > 0.5 * kSize * kSize && shadow > 100, true, "geometry and shadows visible");
        expectEq(bad, 0, "visibility bitwise equal to DeferredLighting");
        expectEq(bitwiseEqual(r.vis, r.again), true, "repeated execution bitwise equal");
    }
});

gpuTest("Visibility.rayTracedMatchesCpuAndShadowMap", async ({ device }) => {
    const scene = makeScene(device, [0.02, 0.02, 0.02]); // own instance: the default shared scene can render an empty G-buffer after other suites created scenes
    const rt = await render(device, scene, "RayTraced", null);
    const sm = await render(device, scene, "ShadowMap", null);
    const none = await render(device, scene, "None", null);
    let n = 0, cpuBad = 0, smBad = 0, noneBad = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        if (rt.vis[i * 4 + 3] === 0 || rt.pos[i * 4 + 3] === 0) continue;
        if (none.vis[i * 4] !== 1) noneBad++;
        if (!(rt.vis[i * 4 + 1]! > kMinCosTheta)) continue; // back-facing: self-shadowed, not a light query
        n++;
        const p: V3 = [rt.pos[i * 4]!, rt.pos[i * 4 + 1]!, rt.pos[i * 4 + 2]!];
        const cpu = blocked(p, kL) ? 0 : 1;
        if (rt.vis[i * 4] !== cpu) cpuBad++;
        if (Math.abs(sm.vis[i * 4]! - rt.vis[i * 4]!) > 0.5) smBad++;
    }
    log(`Visibility: ${n} front-lit px; RayTraced vs CPU differ ${cpuBad} (${(100 * cpuBad / n).toFixed(2)}%), ShadowMap vs RayTraced differ ${smBad} (${(100 * smBad / n).toFixed(2)}%), None != 1: ${noneBad}`);
    expectEq(cpuBad <= 0.01 * n, true, "ray-traced visibility = CPU shadow test (edge pixels aside)");
    expectEq(smBad <= 0.03 * n, true, "shadow map = ray traced except along shadow edges");
    expectEq(noneBad, 0, "mode None is all ones");
});
