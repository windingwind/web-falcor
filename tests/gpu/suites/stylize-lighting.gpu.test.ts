/**
 * StyleAovs, RimLight, RampShade (packages/render-passes/src/Stylize) against CPU references computed from the same
 * G-buffer readbacks. Identity parameters must reproduce the input bit for bit.
 */

import { kRamps } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import {
    kAlbedo, kL, kSize, bitwiseEqual, camInfo, clamp01, compareRgb, dot3, log, makeScene, norm3, objectClass, px, readAll, readOut, runGraph,
    type V3,
} from "./stylize-helpers.js";

const kG: [string, string][] = [];
const srcEdge = (pass: string): [string, string] => ["G.diffuseOpacity", `${pass}.src`];

function pcgHash(v: number): number {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}
const hash01 = (v: number, seed: number, salt: number) => (((pcgHash((v ^ pcgHash((seed + salt) >>> 0)) >>> 0)) >>> 8) / 16777216);

gpuTest("StyleAovs.idsAlbedoNdotL", async ({ device }) => {
    const scene = makeScene(device);
    const seed = 7;
    for (const idSource of ["instance", "material"]) {
        const g = await runGraph(device, scene, [["StyleAovs", "S", { idSource, hashSeed: seed }]],
            [["G.vbuffer", "S.vbuffer"], ["G.texC", "S.texC"], ["G.normW", "S.normW"], ...kG],
            ["S.id", "S.idColor", "S.materialId", "S.flatAlbedo", "S.ndotl", "G.posW", "G.normW"]);
        const [id, idc, mid, flat, ndl, pos, nrm] = await readAll(device, g, ["S.id", "S.idColor", "S.materialId", "S.flatAlbedo", "S.ndotl", "G.posW", "G.normW"]);
        let bgBad = 0, hashBad = 0, mtlBad = 0, flatErr = 0, ndlErr = 0, n = 0;
        const groups = new Map<string, Set<number>>();
        for (let i = 0; i < kSize * kSize; i++) {
            const [x, y, z, w] = px(pos, i);
            if (w === 0) { if (id[i * 4 + 3] !== 0 || flat[i * 4 + 3] !== 0) bgBad++; continue; }
            n++;
            const raw = id[i * 4 + 1]!, mraw = mid[i * 4 + 1]!;
            if (Math.abs(id[i * 4]! - hash01(raw, seed, 0)) > 1e-7 || Math.abs(idc[i * 4 + 2]! - hash01(raw, seed, 3)) > 1e-7 || Math.abs(mid[i * 4]! - hash01(mraw, seed, 4)) > 1e-7) hashBad++;
            const cls = objectClass(x, y, z);
            if (cls >= 0) {
                if (mraw !== cls) mtlBad++;
                const a = kAlbedo[cls]!;
                flatErr = Math.max(flatErr, ...[0, 1, 2].map((c) => Math.abs(flat[i * 4 + c]! - a[c]!)));
                const nn = norm3([nrm[i * 4]!, nrm[i * 4 + 1]!, nrm[i * 4 + 2]!]);
                const key = `${cls}:${nn.map((v) => Math.round(v)).join(",")}`;
                (groups.get(key) ?? groups.set(key, new Set()).get(key)!).add(raw);
            }
            const nn = norm3([nrm[i * 4]!, nrm[i * 4 + 1]!, nrm[i * 4 + 2]!]);
            ndlErr = Math.max(ndlErr, Math.abs(ndl[i * 4]! - dot3(nn, kL)), Math.abs(ndl[i * 4 + 1]! - (0.5 * dot3(nn, kL) + 0.5)));
        }
        const distinct = new Set([...groups.values()].flatMap((s) => [...s]));
        const constantPerGroup = [...groups.values()].every((s) => s.size === 1);
        log(`StyleAovs[${idSource}]: geometry px ${n}, bg bad ${bgBad}, hash bad ${hashBad}, material-id bad ${mtlBad}, flat albedo max err ${flatErr.toExponential(2)}, ndotl max err ${ndlErr.toExponential(2)}, groups ${groups.size}, distinct ids ${distinct.size}`);
        expectEq(bgBad, 0, "background pixels are zero");
        expectEq(hashBad, 0, "hash outputs match the CPU hash of the raw ids");
        expectEq(mtlBad, 0, "material ids match the scene");
        expectEq(flatErr < 2e-3, true, "flat albedo = material base colour (f16)");
        expectEq(ndlErr < 1e-5, true, "ndotl = dot(normW, L)");
        expectEq(constantPerGroup, true, "id constant within each face group");
        expectEq(idSource === "material" ? distinct.size === 3 : distinct.size >= 3, true, "distinct ids per object");
    }
});

function rimRef(scene: ReturnType<typeof makeScene>, p: Record<string, number | string | number[]>, src: Float32Array, pos: Float32Array, nrm: Float32Array) {
    const cam = camInfo(scene);
    const depth = (i: number) => (pos[i * 4]! - cam.pos[0]) * cam.fwd[0] + (pos[i * 4 + 1]! - cam.pos[1]) * cam.fwd[1] + (pos[i * 4 + 2]! - cam.pos[2]) * cam.fwd[2];
    const colour = p.colour as number[];
    return (i: number): V3 | null => {
        if (pos[i * 4 + 3] === 0) return [src[i * 4]!, src[i * 4 + 1]!, src[i * 4 + 2]!];
        const n = norm3([nrm[i * 4]!, nrm[i * 4 + 1]!, nrm[i * 4 + 2]!]);
        let rim = 0;
        if (p.mode === "fresnel") {
            const V = norm3([cam.pos[0] - pos[i * 4]!, cam.pos[1] - pos[i * 4 + 1]!, cam.pos[2] - pos[i * 4 + 2]!]);
            rim = Math.pow(1 - clamp01(dot3(n, V)), p.power as number) * (1 + (clamp01(n[1]) - 1) * (p.upBias as number));
        } else {
            let dx = dot3(cam.viewRows[0]!, n), dy = -dot3(cam.viewRows[1]!, n);
            const len = Math.hypot(dx, dy);
            if (len > 1e-4) {
                dx /= len; dy /= len;
                const x = i % kSize, y = Math.floor(i / kSize);
                const qx = x + Math.round(dx * (p.widthPx as number)), qy = y + Math.round(dy * (p.widthPx as number));
                if (qx < 0 || qy < 0 || qx >= kSize || qy >= kSize) rim = 1;
                else {
                    const q = qy * kSize + qx;
                    const d = depth(i);
                    rim = pos[q * 4 + 3] === 0 || depth(q) - d > (p.depthThreshold as number) * d ? 1 : 0;
                }
            }
        }
        rim *= 1 + (clamp01(dot3(n, kL)) - 1) * (p.lightFacing as number);
        const t = p.tint as number;
        return [0, 1, 2].map((c) => src[i * 4 + c]! + (p.intensity as number) * rim * colour[c]! * (1 + (src[i * 4 + c]! - 1) * t)) as V3;
    };
}

gpuTest("RimLight.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    const cases = [
        { mode: "fresnel", colour: [1, 0.9, 0.7], intensity: 0.8, power: 3, upBias: 0.6, lightFacing: 0.3, tint: 0.5, widthPx: 3, depthThreshold: 0.05 },
        { mode: "depth_offset", colour: [1, 1, 1], intensity: 0.6, power: 4, upBias: 0, lightFacing: 0, tint: 0, widthPx: 3, depthThreshold: 0.05 },
    ];
    for (const c of cases) {
        const g = await runGraph(device, scene, [["RimLight", "R", c]],
            [srcEdge("R"), ["G.posW", "R.posW"], ["G.normW", "R.normW"], ["G.diffuseOpacity", "R.albedo"]], ["R.dst", "G.posW", "G.normW", "G.diffuseOpacity"]);
        const [out, pos, nrm, src] = await readAll(device, g, ["R.dst", "G.posW", "G.normW", "G.diffuseOpacity"]);
        const r = compareRgb(out, rimRef(scene, c, src, pos, nrm), 1e-4);
        let rimPx = 0;
        for (let i = 0; i < kSize * kSize; i++) if (Math.abs(out[i * 4]! - src[i * 4]!) > 0.05) rimPx++;
        log(`RimLight[${c.mode}]: max err ${r.maxErr.toExponential(2)}, share > 1e-4 ${(100 * r.badShare).toFixed(3)} %, rim px ${rimPx}`);
        expectEq(r.badShare < (c.mode === "fresnel" ? 1e-9 : 5e-3), true, "matches the CPU reference");
        expectEq(rimPx > 50, true, "rim visible");
    }
    const g0 = await runGraph(device, scene, [["RimLight", "R", { intensity: 0 }]], [srcEdge("R"), ["G.posW", "R.posW"], ["G.normW", "R.normW"]], ["R.dst", "G.diffuseOpacity"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "R.dst"), await readOut(device, g0, "G.diffuseOpacity")), true, "intensity 0 is the identity");
});

gpuTest("RampShade.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device);
    const custom = [[0, [0.2, 0.25, 0.4]], [0.5, [0.6, 0.4, 0.35]], [1, [1, 0.95, 0.9]]];
    const cases: Record<string, unknown>[] = [
        { mode: "half_lambert", ramp: "tf2_warm", gamma: 1.4, hardness: 0, overbright: 1.5, ambient: 0.05, mix: 1 },
        { mode: "lambert", ramp: "custom", stops: custom, gamma: 1, hardness: 0.7, overbright: 1, ambient: 0, mix: 0.8 },
        { mode: "half_lambert", ramp: "anime_hard", gamma: 1, hardness: 1, overbright: 1.2, ambient: 0, mix: 1 },
    ];
    const hex = (h: string): V3 => [0, 2, 4].map((k) => { const x = parseInt(h.slice(1 + k, 3 + k), 16) / 255; return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); }) as V3;
    for (const c of cases) {
        const stops = (c.ramp === "custom" ? (c.stops as [number, V3][]) : (kRamps as Record<string, [number, string][]>)[c.ramp as string]!.map(([p, h]) => [p, hex(h)] as [number, V3]));
        const g = await runGraph(device, scene, [["RampShade", "S", c]],
            [srcEdge("S"), ["G.diffuseOpacity", "S.albedo"], ["G.normW", "S.normW"], ["G.posW", "S.posW"]], ["S.dst", "G.posW", "G.normW", "G.diffuseOpacity"]);
        const [out, pos, nrm, alb] = await readAll(device, g, ["S.dst", "G.posW", "G.normW", "G.diffuseOpacity"]);
        const hard = c.hardness as number;
        const ramp = (x: number): V3 => {
            if (x <= stops[0]![0]) return stops[0]![1];
            for (let k = 1; k < stops.length; k++) {
                if (x <= stops[k]![0]) {
                    let t = (x - stops[k - 1]![0]) / Math.max(stops[k]![0] - stops[k - 1]![0], 1e-6);
                    t = clamp01((t - 0.5) / Math.max(1 - hard, 1e-4) + 0.5);
                    return [0, 1, 2].map((ch) => stops[k - 1]![1][ch]! + (stops[k]![1][ch]! - stops[k - 1]![1][ch]!) * t) as V3;
                }
            }
            return stops[stops.length - 1]![1];
        };
        const r = compareRgb(out, (i) => {
            const s: V3 = [alb[i * 4]!, alb[i * 4 + 1]!, alb[i * 4 + 2]!];
            if (pos[i * 4 + 3] === 0) return s;
            const ndl = dot3(norm3([nrm[i * 4]!, nrm[i * 4 + 1]!, nrm[i * 4 + 2]!]), kL);
            let x = c.mode === "half_lambert" ? 0.5 * ndl + 0.5 : clamp01(ndl);
            x = Math.pow(clamp01(x), c.gamma as number);
            const rr = ramp(x);
            const sh = [0, 1, 2].map((ch) => s[ch]! * (rr[ch]! * (c.overbright as number) + (c.ambient as number)));
            return [0, 1, 2].map((ch) => s[ch]! + (sh[ch]! - s[ch]!) * (c.mix as number)) as V3;
        }, 2e-4);
        let changed = 0;
        for (let i = 0; i < kSize * kSize; i++) if (Math.abs(out[i * 4]! - alb[i * 4]!) > 0.02) changed++;
        log(`RampShade[${c.ramp}/${c.mode}]: changed px ${changed}, max err ${r.maxErr.toExponential(2)}, share > 2e-4 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(changed > 1000, true, "ramp shading visible");
        expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    }
    const g0 = await runGraph(device, scene, [["RampShade", "S", { mix: 0 }]],
        [srcEdge("S"), ["G.diffuseOpacity", "S.albedo"], ["G.normW", "S.normW"], ["G.posW", "S.posW"]], ["S.dst", "G.diffuseOpacity"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "S.dst"), await readOut(device, g0, "G.diffuseOpacity")), true, "mix 0 is the identity");
});
