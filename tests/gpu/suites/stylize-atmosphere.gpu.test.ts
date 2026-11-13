/**
 * GradientFog and LightShafts (packages/render-passes/src/Stylize) against CPU references on the same readbacks;
 * identity parameters must reproduce the input bit for bit.
 */

import { projectDirection } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { bitwiseEqual, camInfo, compareRgb, dot3, kL, kSize, log, makeScene, readAll, readOut, runGraph, type PassDecl, type V3 } from "./stylize-helpers.js";

const kSky: V3 = [0.9, 0.95, 1.1];
const kLit: PassDecl[] = [["DeferredLighting", "L", { shadowMode: "RayTraced" }], ["ToneMapper", "TM", { operator: "Aces", autoExposure: false, exposureCompensation: 0, clamp: true, outputFormat: "RGBA32Float" }]];
const kLitEdges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["L.color", "TM.src"]];
const rgb = (b: Float32Array, i: number): V3 => [b[i * 4]!, b[i * 4 + 1]!, b[i * 4 + 2]!];

gpuTest("GradientFog.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device, kSky);
    const cam = camInfo(scene);
    const cases = [
        { stops: [[0, [0.9, 0.7, 0.5]], [3, [0.6, 0.6, 0.7]], [6, [0.3, 0.4, 0.8]]], density: 0.25, start: 1, heightFalloff: 0, heightRef: 0, up: [0, 1, 0], skyMode: "fog", maxOpacity: 0.9 },
        { stops: [[0, "#f0a060"], [8, "#6040a0"]], density: 0.6, start: 0, heightFalloff: 1.5, heightRef: 0.2, up: [0, 1, 0], skyMode: "keep", maxOpacity: 1 },
    ];
    const hexLin = (h: string): V3 => [0, 2, 4].map((k) => { const x = parseInt(h.slice(1 + k, 3 + k), 16) / 255; return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); }) as V3;
    for (const c of cases) {
        const stops = c.stops.map(([d, col]) => [d as number, typeof col === "string" ? hexLin(col) : (col as V3)] as [number, V3]);
        const g = await runGraph(device, scene, [...kLit, ["GradientFog", "F", c]], [...kLitEdges, ["TM.dst", "F.src"], ["G.posW", "F.posW"]], ["F.dst", "TM.dst", "G.posW"]);
        const [out, src, pos] = await readAll(device, g, ["F.dst", "TM.dst", "G.posW"]);
        const ramp = (d: number): V3 => {
            if (d <= stops[0]![0]) return stops[0]![1];
            for (let k = 1; k < stops.length; k++) if (d <= stops[k]![0]) {
                const t = (d - stops[k - 1]![0]) / Math.max(stops[k]![0] - stops[k - 1]![0], 1e-6);
                return [0, 1, 2].map((ch) => stops[k - 1]![1][ch]! + (stops[k]![1][ch]! - stops[k - 1]![1][ch]!) * t) as V3;
            }
            return stops[stops.length - 1]![1];
        };
        const r = compareRgb(out, (i) => {
            const s = rgb(src, i);
            if (pos[i * 4 + 3] === 0) return c.skyMode === "keep" ? s : (s.map((v, k) => v + (stops[stops.length - 1]![1][k]! - v) * c.maxOpacity) as V3);
            const v = [pos[i * 4]! - cam.pos[0], pos[i * 4 + 1]! - cam.pos[1], pos[i * 4 + 2]! - cam.pos[2]];
            const d = Math.hypot(v[0]!, v[1]!, v[2]!);
            const L = Math.max(d - c.start, 0);
            let tau = c.density * L;
            if (c.heightFalloff > 0 && L > 0) {
                const k = c.heightFalloff, u = v[1]! / d;
                const hs = cam.pos[1] + u * c.start - c.heightRef;
                const ku = k * u * L;
                const gg = Math.abs(ku) < 1e-3 ? L * (1 - 0.5 * ku) : (1 - Math.exp(-ku)) / (k * u);
                tau = c.density * Math.exp(-k * hs) * gg;
            }
            const f = c.maxOpacity * (1 - Math.exp(-tau));
            const col = ramp(d);
            return s.map((x, k) => x + (col[k]! - x) * f) as V3;
        }, 2e-4);
        log(`GradientFog[${c.stops.length} stops, k ${c.heightFalloff}, sky ${c.skyMode}]: max err ${r.maxErr.toExponential(2)}, share > 2e-4 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    }
    const g0 = await runGraph(device, scene, [...kLit, ["GradientFog", "F", { density: 0 }]], [...kLitEdges, ["TM.dst", "F.src"], ["G.posW", "F.posW"]], ["F.dst", "TM.dst"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "F.dst"), await readOut(device, g0, "TM.dst")), true, "density 0 is the identity");
});

gpuTest("LightShafts.matchesCpuReference", async ({ device }) => {
    // camera looking towards the light so that the sun is on screen above the boxes
    const target: V3 = [0, 0.6, -0.6];
    const camPos: V3 = [target[0] - kL[0] * 4, 0.9, target[2] - kL[2] * 4];
    const scene = makeScene(device, kSky, { pos: camPos, target });
    const cam = camInfo(scene);
    const fwdUv = projectDirection(cam.viewProjRows, cam.pos, cam.fwd, cam.fwd);
    log(`projectDirection(camera forward) = ${fwdUv.uv.map((x) => x.toFixed(4)).join(", ")}, facing ${fwdUv.facing}`);
    expectEq(Math.abs(fwdUv.uv[0] - 0.5) < 1e-4 && Math.abs(fwdUv.uv[1] - 0.5) < 1e-4, true, "camera forward projects to the image centre");
    const c = { samples: 48, density: 0.9, decay: 0.96, weight: 0.05, exposure: 1.2, threshold: 0.3, tint: [1, 0.9, 0.7], sunUV: [0.62, 0.1] };
    const g = await runGraph(device, scene, [...kLit, ["LightShafts", "S", c]], [...kLitEdges, ["TM.dst", "S.src"], ["G.posW", "S.posW"]], ["S.dst", "TM.dst", "G.posW"]);
    const [out, src, pos] = await readAll(device, g, ["S.dst", "TM.dst", "G.posW"]);
    const f32 = Math.fround;
    const occ = (u: number, v: number) => {
        const qx = Math.min(Math.max(Math.trunc(u * kSize), 0), kSize - 1), qy = Math.min(Math.max(Math.trunc(v * kSize), 0), kSize - 1);
        const q = qy * kSize + qx;
        if (pos[q * 4 + 3] !== 0) return 0;
        return Math.max(dot3(rgb(src, q), [0.2126, 0.7152, 0.0722]) - c.threshold, 0);
    };
    let lit = 0;
    const r = compareRgb(out, (i) => {
        let u = f32((i % kSize + 0.5) / kSize), v = f32((Math.floor(i / kSize) + 0.5) / kSize);
        const su = f32((c.sunUV[0]! - u) * f32(c.density / c.samples)), sv = f32((c.sunUV[1]! - v) * f32(c.density / c.samples));
        let illum = 1, acc = 0;
        for (let k = 0; k < c.samples; k++) {
            u = f32(u + su); v = f32(v + sv);
            if (u < 0 || v < 0 || u > 1 || v > 1) break;
            acc += occ(u, v) * illum * c.weight;
            illum *= c.decay;
        }
        if (acc > 0.05) lit++;
        return rgb(src, i).map((x, k) => x + c.exposure * acc * c.tint[k]!) as V3;
    }, 1e-3);
    log(`LightShafts: shaft px ${lit}, max err ${r.maxErr.toExponential(2)}, share > 1e-3 ${(100 * r.badShare).toFixed(3)} %`);
    expectEq(lit > 300, true, "shafts present");
    expectEq(r.badShare < 5e-3, true, "matches the CPU reference");
    const g0 = await runGraph(device, scene, [...kLit, ["LightShafts", "S", { exposure: 0 }]], [...kLitEdges, ["TM.dst", "S.src"], ["G.posW", "S.posW"]], ["S.dst", "TM.dst"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "S.dst"), await readOut(device, g0, "TM.dst")), true, "exposure 0 is the identity");
});
