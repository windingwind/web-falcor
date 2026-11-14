/**
 * SelectiveColour, PaletteQuant, DitherV2, PixelateV2 (packages/render-passes/src/Stylize) against CPU references on the
 * same readbacks. Source images: GBuffer albedo, or DeferredLighting output (smooth gradients). Identity parameters must
 * reproduce the input bit for bit.
 */

import { kPalettes, linearSrgbToOklab, voidAndCluster } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import {
    bitwiseEqual, camInfo, compareRgb, dot3, kSize, log, makeScene, norm3, readAll, readOut, runGraph, smoothstep, srgbDec, srgbEnc,
    type PassDecl, type V3,
} from "./stylize-helpers.js";

/** GBuffer + DeferredLighting (ray-traced shadows, constant sky) + ToneMapper (linear output) as the colour source "TM.dst". */
const kLit: PassDecl[] = [["DeferredLighting", "L", { shadowMode: "RayTraced" }], ["ToneMapper", "TM", { operator: "Aces", autoExposure: false, exposureCompensation: 0, clamp: true, outputFormat: "RGBA32Float" }]];
const kLitEdges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["L.color", "TM.src"]];

const rgb = (b: Float32Array, i: number): V3 => [b[i * 4]!, b[i * 4 + 1]!, b[i * 4 + 2]!];

gpuTest("SelectiveColour.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device, [0.3, 0.35, 0.45]);
    const cases = [
        { keepHue: 29, hueWidth: 20, softness: 15, minChroma: 0.05, satKeep: 1.2, satRest: 0, source: "src" },
        { keepHue: 264, hueWidth: 30, softness: 10, minChroma: 0.03, satKeep: 1, satRest: 0.2, source: "albedo" },
    ];
    for (const c of cases) {
        const g = await runGraph(device, scene, [...kLit, ["SelectiveColour", "S", c]], [...kLitEdges, ["TM.dst", "S.src"], ["G.diffuseOpacity", "S.albedo"]], ["S.dst", "TM.dst", "G.diffuseOpacity"]);
        const [out, src, alb] = await readAll(device, g, ["S.dst", "TM.dst", "G.diffuseOpacity"]);
        let keptPx = 0;
        const r = compareRgb(out, (i) => {
            const key = c.source === "albedo" ? rgb(alb, i) : rgb(src, i);
            const lab = linearSrgbToOklab(key.map((v) => Math.max(v, 0)));
            const C = Math.hypot(lab[1], lab[2]);
            const h = (Math.atan2(lab[2], lab[1]) * 180) / Math.PI;
            let dh = Math.abs(h - c.keepHue) % 360;
            dh = Math.min(dh, 360 - dh);
            const m = (1 - smoothstep(c.hueWidth, c.hueWidth + c.softness, dh)) * smoothstep(0.5 * c.minChroma, c.minChroma, C);
            if (m > 0.5) keptPx++;
            const s = c.satRest + (c.satKeep - c.satRest) * m;
            const x = rgb(src, i);
            const luma = dot3(x, [0.2126, 0.7152, 0.0722]);
            return s === 1 ? x : (x.map((v) => luma + (v - luma) * s) as V3);
        }, 1e-4);
        log(`SelectiveColour[hue ${c.keepHue}, ${c.source}]: kept px ${keptPx}, max err ${r.maxErr.toExponential(2)}, share > 1e-4 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(keptPx > 200, true, "key colour present");
        expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    }
    const g0 = await runGraph(device, scene, [...kLit, ["SelectiveColour", "S", { satKeep: 1, satRest: 1 }]], [...kLitEdges, ["TM.dst", "S.src"]], ["S.dst", "TM.dst"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "S.dst"), await readOut(device, g0, "TM.dst")), true, "satKeep = satRest = 1 is the identity");
});

const hexLin = (h: string): V3 => [0, 2, 4].map((k) => srgbDec(parseInt(h.slice(1 + k, 3 + k), 16) / 255)) as V3;
const bayer4 = (x: number, y: number) => { const m = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]; return (m[(y & 3) * 4 + (x & 3)]! + 0.5) / 16 - 0.5; };

gpuTest("PaletteQuant.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device, [0.3, 0.35, 0.45]);
    const cases = [
        { palette: "pico8_16", space: "oklab", dither: 0, mix: 1 },
        { palette: "gameboy4", space: "oklab", dither: 1, mix: 1 },
        { palette: "ega16", space: "rgb", dither: 0.5, mix: 0.7 },
        { palette: "rgb555", space: "oklab", dither: 0, mix: 1 },
    ];
    for (const c of cases) {
        const g = await runGraph(device, scene, [...kLit, ["PaletteQuant", "Q", c]], [...kLitEdges, ["TM.dst", "Q.src"]], ["Q.dst", "TM.dst"]);
        const [out, src] = await readAll(device, g, ["Q.dst", "TM.dst"]);
        const pal = c.palette === "rgb555" ? [] : kPalettes[c.palette]!.map(hexLin);
        const cmp = pal.map((p) => (c.space === "oklab" ? linearSrgbToOklab(p) : p));
        let spacing = 1 / 31;
        if (pal.length) {
            let sum = 0;
            for (let i = 0; i < cmp.length; i++) sum += Math.min(...cmp.map((q, j) => (j === i ? Infinity : Math.hypot(q[0] - cmp[i]![0], q[1] - cmp[i]![1], q[2] - cmp[i]![2]))));
            spacing = sum / cmp.length;
        }
        const used = new Set<number>();
        const r = compareRgb(out, (i) => {
            const x = rgb(src, i);
            const t = c.dither * bayer4(i % kSize, Math.floor(i / kSize)) * spacing;
            let q: V3;
            if (!pal.length) {
                q = x.map((v) => srgbDec(Math.floor(Math.min(Math.max(srgbEnc(v) + t, 0), 1) * 31 + 0.5) / 31)) as V3;
            } else {
                const y = c.space === "oklab" ? linearSrgbToOklab(x.map((v) => Math.max(v, 0))) : [...x];
                if (c.space === "oklab") y[0] += t; else for (let k = 0; k < 3; k++) y[k] += t;
                let best = Infinity, bi = 0;
                cmp.forEach((p, j) => { const d = (y[0]! - p[0]) ** 2 + (y[1]! - p[1]) ** 2 + (y[2]! - p[2]) ** 2; if (d < best) { best = d; bi = j; } });
                used.add(bi);
                q = pal[bi]!;
            }
            return x.map((v, k) => v + (q[k]! - v) * c.mix) as V3;
        }, 2e-4);
        log(`PaletteQuant[${c.palette}/${c.space}/dither ${c.dither}]: palette entries used ${used.size}, max err ${r.maxErr.toExponential(2)}, share > 2e-4 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    }
    const g0 = await runGraph(device, scene, [...kLit, ["PaletteQuant", "Q", { mix: 0 }]], [...kLitEdges, ["TM.dst", "Q.src"]], ["Q.dst", "TM.dst"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "Q.dst"), await readOut(device, g0, "TM.dst")), true, "mix 0 is the identity");
});

gpuTest("DitherV2.matchesCpuReference", async ({ device }) => {
    // blue-noise rank map: a permutation, and uniform at 8x8 scale for every threshold level
    const n = 64, ranks = voidAndCluster(n);
    const perm = new Set(Array.from(ranks, (r) => Math.round(r * n * n)));
    let worstBlock = 0;
    for (const lvl of [0.25, 0.5, 0.75])
        for (let by = 0; by < n; by += 8) for (let bx = 0; bx < n; bx += 8) {
            let on = 0;
            for (let y = by; y < by + 8; y++) for (let x = bx; x < bx + 8; x++) on += ranks[y * n + x]! < lvl ? 1 : 0;
            worstBlock = Math.max(worstBlock, Math.abs(on / 64 - lvl));
        }
    log(`blue noise: ${perm.size} distinct ranks of ${n * n}, worst 8x8 block density deviation ${worstBlock.toFixed(3)}`);
    expectEq(perm.size, n * n, "rank map is a permutation");
    expectEq(worstBlock < 0.1, true, "blue noise is locally uniform");

    const scene = makeScene(device, [0.3, 0.35, 0.45]);
    const cam = camInfo(scene);
    const angScale = kSize / (2 * Math.tan(0.5 * cam.fovY)) / 2;
    const bayer = (cx: number, cy: number, nb: number) => {
        let v = 0;
        for (let k = 0; k < nb; k++) { const bx = (cx >> k) & 1, by = (cy >> k) & 1; v = (v << 2) | ((bx ^ by) << 1) | by; }
        return (v + 0.5) / (1 << (2 * nb)) - 0.5;
    };
    const ps1 = [-4, 0, -3, 1, 2, -2, 3, -1, -3, 1, -4, 0, 3, -1, 2, -2];
    const cases = [
        { pattern: "bayer4", space: "screen", levels: 4, scale: 1, strength: 1, mix: 1 },
        { pattern: "bayer8", space: "screen", levels: 2, scale: 2, strength: 1, mix: 1 },
        { pattern: "ps1", space: "screen", levels: 32, scale: 1, strength: 1, mix: 1 },
        { pattern: "bayer2", space: "view_sphere", levels: 3, scale: 2, strength: 0.8, mix: 1 },
    ];
    for (const c of cases) {
        const g = await runGraph(device, scene, [...kLit, ["DitherV2", "D", c]], [...kLitEdges, ["TM.dst", "D.src"], ["G.posW", "D.posW"], ["G.normW", "D.normW"]], ["D.dst", "TM.dst", "G.posW"]);
        const [out, src, pos] = await readAll(device, g, ["D.dst", "TM.dst", "G.posW"]);
        const r = compareRgb(out, (i) => {
            const x = i % kSize, y = Math.floor(i / kSize);
            let cx = Math.floor(x / c.scale), cy = Math.floor(y / c.scale);
            if (c.space === "view_sphere" && pos[i * 4 + 3] !== 0) {
                const d = norm3([pos[i * 4]! - cam.pos[0], pos[i * 4 + 1]! - cam.pos[1], pos[i * 4 + 2]! - cam.pos[2]]);
                cx = Math.floor(Math.atan2(d[2], d[0]) * angScale); cy = Math.floor(Math.asin(Math.max(-1, Math.min(1, d[1]))) * angScale);
            }
            const e = rgb(src, i).map(srgbEnc);
            let q: number[];
            if (c.pattern === "ps1") {
                const off = c.strength * ps1[(cy & 3) * 4 + (cx & 3)]!;
                q = e.map((v) => Math.min(Math.max(Math.floor(Math.min(Math.max(Math.floor(v * 255 + 0.5) + off, 0), 255) / 8) * 8 / 248, 0), 1));
            } else {
                const nb = c.pattern === "bayer2" ? 1 : c.pattern === "bayer4" ? 2 : 3;
                const t = bayer(cx & 0xffff, cy & 0xffff, nb);
                const L = c.levels - 1;
                q = e.map((v) => Math.min(Math.max(Math.floor(v * L + 0.5 + c.strength * t), 0), L) / L);
            }
            return q.map(srgbDec) as V3;
        }, 2e-4);
        log(`DitherV2[${c.pattern}/${c.space}/L${c.levels}]: max err ${r.maxErr.toExponential(2)}, share > 2e-4 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(r.badShare < 3e-3, true, "matches the CPU reference");
    }
    const g0 = await runGraph(device, scene, [...kLit, ["DitherV2", "D", { mix: 0 }]], [...kLitEdges, ["TM.dst", "D.src"]], ["D.dst", "TM.dst"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "D.dst"), await readOut(device, g0, "TM.dst")), true, "mix 0 is the identity");
    const gb = await runGraph(device, scene, [...kLit, ["DitherV2", "D", { pattern: "blue_noise", levels: 2 }]], [...kLitEdges, ["TM.dst", "D.src"]], ["D.dst"]);
    const ob = await readOut(device, gb, "D.dst");
    const vals = new Set<number>();
    for (let i = 0; i < kSize * kSize; i++) vals.add(Math.round(srgbEnc(ob[i * 4]!) * 1000));
    log(`DitherV2[blue_noise, 2 levels]: encoded red values ${[...vals].sort((a, b) => a - b).join(",")}`);
    expectEq([...vals].every((v) => v === 0 || v === 1000), true, "2 levels give only 0 / 1");
});

gpuTest("PixelateV2.matchesCpuReference", async ({ device }) => {
    const scene = makeScene(device, [0.3, 0.35, 0.45]);
    const cam = camInfo(scene);
    const cases = [
        { block: 4, mode: "avg", offset: [1, 2], outline: 0 },
        { block: 5, mode: "nearest", offset: [0, 0], outline: 0.8, outlineThreshold: 0.05, normalThreshold: 0.3 },
        { block: 6, mode: "majority", offset: [3, 1], outline: 0 },
    ];
    for (const c of cases) {
        const g = await runGraph(device, scene, [...kLit, ["PixelateV2", "P", c]], [...kLitEdges, ["TM.dst", "P.src"], ["G.posW", "P.posW"], ["G.normW", "P.normW"]], ["P.dst", "TM.dst", "G.posW", "G.normW"]);
        const [out, src, pos, nrm] = await readAll(device, g, ["P.dst", "TM.dst", "G.posW", "G.normW"]);
        const B = c.block, [ox, oy] = c.offset as [number, number];
        const lw = Math.ceil((kSize + ox) / B), lh = Math.ceil((kSize + oy) / B);
        const enc = (v: number) => Math.floor(srgbEnc(v) * 15 + 0.5);
        const bin = (i: number) => (enc(src[i * 4]!) << 8) | (enc(src[i * 4 + 1]!) << 4) | enc(src[i * 4 + 2]!);
        const low: V3[] = [], geo: (null | { d: number; n: V3 })[] = [];
        for (let cy = 0; cy < lh; cy++) for (let cx = 0; cx < lw; cx++) {
            const x0 = cx * B - ox, y0 = cy * B - oy;
            const lx = Math.max(x0, 0), ly = Math.max(y0, 0), hx = Math.min(x0 + B, kSize), hy = Math.min(y0 + B, kSize);
            const ccx = Math.min(Math.max(x0 + Math.floor(B / 2), lx), hx - 1), ccy = Math.min(Math.max(y0 + Math.floor(B / 2), ly), hy - 1);
            const centre = ccy * kSize + ccx;
            const idx: number[] = [];
            for (let y = ly; y < hy; y++) for (let x = lx; x < hx; x++) idx.push(y * kSize + x);
            let col: V3;
            if (c.mode === "nearest") col = rgb(src, centre);
            else {
                let members = idx;
                if (c.mode === "majority") {
                    const counts = new Map<number, number>();
                    let bestB = -1, bestC = 0;
                    for (const i of idx) { const b = bin(i); const k = (counts.get(b) ?? 0) + 1; counts.set(b, k); }
                    for (const i of idx) { const k = counts.get(bin(i))!; if (k > bestC) { bestC = k; bestB = bin(i); } }
                    members = idx.filter((i) => bin(i) === bestB);
                }
                col = [0, 1, 2].map((k) => members.reduce((a, i) => a + src[i * 4 + k]!, 0) / members.length) as V3;
            }
            low.push(col);
            const n = norm3([nrm[centre * 4]!, nrm[centre * 4 + 1]!, nrm[centre * 4 + 2]!]);
            geo.push(pos[centre * 4 + 3] === 0 ? null : { d: dot3([pos[centre * 4]! - cam.pos[0], pos[centre * 4 + 1]! - cam.pos[1], pos[centre * 4 + 2]! - cam.pos[2]], cam.fwd), n });
        }
        let darkened = 0;
        const final = low.map((col, k) => {
            const gk = geo[k];
            if (!c.outline || !gk) return col;
            const cx = k % lw, cy = Math.floor(k / lw);
            let e = 0;
            for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
                const qx = cx + dx!, qy = cy + dy!;
                if (qx < 0 || qy < 0 || qx >= lw || qy >= lh) continue;
                const gq = geo[qy * lw + qx];
                if (!gq) { e = 1; continue; }
                if (gq.d < gk.d) continue;
                if (gq.d - gk.d > (c.outlineThreshold ?? 0.05) * gk.d) e = 1;
                else if (1 - dot3(gk.n, gq.n) > (c.normalThreshold ?? 0.3)) e = 1;
            }
            darkened += e;
            return col.map((v) => v * (1 - c.outline * e)) as V3;
        });
        const r = compareRgb(out, (i) => final[Math.floor(((Math.floor(i / kSize) + oy) / B)) * lw + Math.floor(((i % kSize) + ox) / B)]!, 2e-5);
        log(`PixelateV2[${c.mode}, block ${B}, offset ${ox},${oy}, outline ${c.outline}]: outlined cells ${darkened}, max err ${r.maxErr.toExponential(2)}, share > 2e-5 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(r.badShare < 5e-3, true, "matches the CPU reference");
        if (c.outline) expectEq(darkened > 10, true, "low-res outline present");
    }
    const g0 = await runGraph(device, scene, [...kLit, ["PixelateV2", "P", { block: 1, mode: "avg" }]], [...kLitEdges, ["TM.dst", "P.src"]], ["P.dst", "TM.dst"]);
    expectEq(bitwiseEqual(await readOut(device, g0, "P.dst"), await readOut(device, g0, "TM.dst")), true, "block 1 (avg) is the identity");
});
