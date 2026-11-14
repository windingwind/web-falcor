/**
 * GradientMap, ShadowTint, TiltShift, CRT, VHS, HoldFrames (packages/render-passes/src/Stylize): CPU references on the
 * same readbacks (VHS: its deterministic part), frame-hold behaviour, and bit-exact identity parameters.
 */

import { float3 } from "@web-falcor/falcor";
import { kGradientMaps } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import {
    bitwiseEqual, camInfo, clamp01, compareRgb, dot3, kL, kSize, log, makeScene, norm3, readAll, readOut, runGraph, smoothstep, srgbDec, srgbEnc,
    type PassDecl, type V3,
} from "./stylize-helpers.js";

const kLit: PassDecl[] = [["DeferredLighting", "L", { shadowMode: "RayTraced" }], ["ToneMapper", "TM", { operator: "Aces", autoExposure: false, exposureCompensation: 0, clamp: true, outputFormat: "RGBA32Float" }]];
const kLitEdges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["L.color", "TM.src"]];
const rgb = (b: Float32Array, i: number): V3 => [b[i * 4]!, b[i * 4 + 1]!, b[i * 4 + 2]!];
const hexLin = (h: string): V3 => [0, 2, 4].map((k) => srgbDec(parseInt(h.slice(1 + k, 3 + k), 16) / 255)) as V3;
const SKY: V3 = [0.3, 0.35, 0.45];

async function litRun(device: Parameters<typeof makeScene>[0], pass: PassDecl, ports: [string, string][] = []) {
    const scene = makeScene(device, SKY);
    const id = pass[1];
    const g = await runGraph(device, scene, [...kLit, pass], [...kLitEdges, ["TM.dst", `${id}.src`], ...ports], [`${id}.dst`, "TM.dst"]);
    const [out, src] = await readAll(device, g, [`${id}.dst`, "TM.dst"]);
    return { scene, out: out!, src: src!, g };
}

gpuTest("GradientMap.matchesCpuReference", async ({ device }) => {
    for (const preset of ["tritone_sunset", "neon"]) {
        const strength = preset === "neon" ? 0.7 : 1;
        const { out, src } = await litRun(device, ["GradientMap", "M", { preset, strength }]);
        const stops = kGradientMaps[preset]!.map(([p, h]) => [p as number, hexLin(h as string)] as [number, V3]);
        const ramp = (x: number): V3 => {
            if (x <= stops[0]![0]) return stops[0]![1];
            for (let k = 1; k < stops.length; k++) if (x <= stops[k]![0]) {
                const t = (x - stops[k - 1]![0]) / Math.max(stops[k]![0] - stops[k - 1]![0], 1e-6);
                return [0, 1, 2].map((c) => stops[k - 1]![1][c]! + (stops[k]![1][c]! - stops[k - 1]![1][c]!) * t) as V3;
            }
            return stops[stops.length - 1]![1];
        };
        const r = compareRgb(out, (i) => { const s = rgb(src, i); const c = ramp(srgbEnc(dot3(s, [0.2126, 0.7152, 0.0722]))); return s.map((v, k) => v + (c[k]! - v) * strength) as V3; }, 1e-4);
        log(`GradientMap[${preset}]: max err ${r.maxErr.toExponential(2)}, share > 1e-4 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    }
    const { out, src } = await litRun(device, ["GradientMap", "M", { strength: 0 }]);
    expectEq(bitwiseEqual(out, src), true, "strength 0 is the identity");
});

const cbrt = Math.cbrt;
function toOklab(c: V3): V3 {
    const l = cbrt(0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2]);
    const m = cbrt(0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2]);
    const s = cbrt(0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2]);
    return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function fromOklab(L: V3): V3 {
    const l = (L[0] + 0.3963377774 * L[1] + 0.2158037573 * L[2]) ** 3, m = (L[0] - 0.1055613458 * L[1] - 0.0638541728 * L[2]) ** 3, s = (L[0] - 0.0894841775 * L[1] - 1.291485548 * L[2]) ** 3;
    return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
}

gpuTest("ShadowTint.matchesCpuReference", async ({ device }) => {
    const p = { threshold: 0.3, softness: 0.1, hueShift: -25, valueDrop: 0.35, satKeep: 1.2, nearTint: [0.8, 0.9, 1.2], farTint: [1.1, 0.95, 0.8], near: 2, far: 7, brightness: 1.1, strength: 0.9 };
    const scene = makeScene(device, SKY);
    const g = await runGraph(device, scene, [...kLit, ["ShadowTint", "S", p]],
        [...kLitEdges, ["TM.dst", "S.src"], ["G.diffuseOpacity", "S.albedo"], ["G.normW", "S.normW"], ["G.posW", "S.posW"], ["L.visibility", "S.visibility"]],
        ["S.dst", "TM.dst", "G.diffuseOpacity", "G.normW", "G.posW", "L.visibility"]);
    const [out, src, alb, nrm, pos, vis] = await readAll(device, g, ["S.dst", "TM.dst", "G.diffuseOpacity", "G.normW", "G.posW", "L.visibility"]);
    const cam = camInfo(scene);
    const a = (p.hueShift * Math.PI) / 180;
    let shadowPx = 0;
    const r = compareRgb(out!, (i) => {
        const s0 = rgb(src!, i);
        if (pos![i * 4 + 3] === 0) return s0;
        const vs = vis!.length / (kSize * kSize); // visibility may be a 1-channel texture
        const s = clamp01(dot3(norm3(rgb(nrm!, i)), kL)) * clamp01(vis![i * vs]!);
        const m = 1 - smoothstep(p.threshold - p.softness, p.threshold + p.softness + 1e-5, s);
        if (m > 0.5) shadowPx++;
        const lab = toOklab(rgb(alb!, i).map((v) => Math.max(v, 0)) as V3);
        const [A, B] = [Math.cos(a) * lab[1] - Math.sin(a) * lab[2], Math.sin(a) * lab[1] + Math.cos(a) * lab[2]];
        const sh = fromOklab([lab[0] * (1 - p.valueDrop), A * p.satKeep, B * p.satKeep]).map((v) => Math.max(v, 0));
        const d = dot3([pos![i * 4]! - cam.pos[0], pos![i * 4 + 1]! - cam.pos[1], pos![i * 4 + 2]! - cam.pos[2]], cam.fwd);
        const t = smoothstep(p.near, p.far, d);
        return s0.map((v, k) => v + (sh[k]! * (p.nearTint[k]! + (p.farTint[k]! - p.nearTint[k]!) * t) * p.brightness - v) * m * p.strength) as V3;
    }, 2e-4);
    log(`ShadowTint: shadow px ${shadowPx}, max err ${r.maxErr.toExponential(2)}, share > 2e-4 ${(100 * r.badShare).toFixed(3)} %`);
    expectEq(shadowPx > 1000, true, "shadow areas present");
    expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    const g0 = await runGraph(device, scene, [...kLit, ["ShadowTint", "S", { strength: 0 }]],
        [...kLitEdges, ["TM.dst", "S.src"], ["G.diffuseOpacity", "S.albedo"], ["G.normW", "S.normW"], ["G.posW", "S.posW"]], ["S.dst", "TM.dst"]);
    const [o0, s0] = await readAll(device, g0, ["S.dst", "TM.dst"]);
    expectEq(bitwiseEqual(o0!, s0!), true, "strength 0 is the identity");
});

gpuTest("TiltShift.matchesCpuReference", async ({ device }) => {
    const p = { bandCenter: 0.5, bandWidth: 0.2, feather: 0.3, maxBlur: 4, angle: 20 };
    const { out, src } = await litRun(device, ["TiltShift", "T", p]);
    const W = kSize, H = kSize, a = (p.angle * Math.PI) / 180, n = [-Math.sin(a), Math.cos(a)];
    const sigma = (x: number, y: number) => {
        const q = [(x + 0.5 - 0.5 * W) / H, (y + 0.5 - p.bandCenter * H) / H];
        return p.maxBlur * smoothstep(0, p.feather, Math.abs(q[0]! * n[0]! + q[1]! * n[1]!) - 0.5 * p.bandWidth);
    };
    const pass = (img: Float32Array, dx: number, dy: number) => {
        const o = new Float32Array(img.length);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const s = sigma(x, y), i = y * W + x;
            if (s < 0.05) { for (let c = 0; c < 4; c++) o[i * 4 + c] = img[i * 4 + c]!; continue; }
            const R = Math.ceil(3 * s);
            let w = 0; const acc = [0, 0, 0, 0];
            for (let t = -R; t <= R; t++) {
                const k = Math.exp((-0.5 * t * t) / (s * s));
                const qx = Math.min(Math.max(x + dx * t, 0), W - 1), qy = Math.min(Math.max(y + dy * t, 0), H - 1);
                for (let c = 0; c < 4; c++) acc[c] += k * img[(qy * W + qx) * 4 + c]!;
                w += k;
            }
            for (let c = 0; c < 4; c++) o[i * 4 + c] = acc[c]! / w;
        }
        return o;
    };
    const ref = pass(pass(src, 1, 0), 0, 1);
    const r = compareRgb(out, (i) => rgb(ref, i), 1e-4);
    log(`TiltShift: max err ${r.maxErr.toExponential(2)}, share > 1e-4 ${(100 * r.badShare).toFixed(3)} %`);
    expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    const z = await litRun(device, ["TiltShift", "T", { maxBlur: 0 }]);
    expectEq(bitwiseEqual(z.out, z.src), true, "maxBlur 0 is the identity");
});

gpuTest("CRT.matchesCpuReference", async ({ device }) => {
    for (const p of [
        { sourceLines: 40, beamMin: 0.3, beamMax: 0.6, mask: "aperture", maskStrength: 0.5, curvature: 0.15, halation: 0.1, vignette: 0.5, brightness: 1.3, strength: 1 },
        { sourceLines: 64, beamMin: 0.25, beamMax: 0.5, mask: "slot", maskStrength: 0.7, curvature: 0, halation: 0, vignette: 0, brightness: 1, strength: 0.8 },
    ]) {
        const { out, src } = await litRun(device, ["CRT", "C", p]);
        const f32 = Math.fround;
        const fetch = (u: number, v: number): V3 => rgb(src, Math.min(Math.max(Math.trunc(v * kSize), 0), kSize - 1) * kSize + Math.min(Math.max(Math.trunc(u * kSize), 0), kSize - 1));
        const r = compareRgb(out, (i) => {
            const x = i % kSize, y = Math.floor(i / kSize);
            let u = f32((x + 0.5) / kSize), v = f32((y + 0.5) / kSize);
            let cx = u * 2 - 1, cy = v * 2 - 1;
            const k = 1 + p.curvature * (cx * cx + cy * cy) * 0.25;
            cx *= k; cy *= k; u = cx * 0.5 + 0.5; v = cy * 0.5 + 0.5;
            let col: V3 = [0, 0, 0];
            if (u >= 0 && v >= 0 && u <= 1 && v <= 1) {
                const yy = v * p.sourceLines - 0.5, y0 = Math.floor(yy);
                for (let j = 0; j <= 1; j++) {
                    const line = y0 + j, c3 = fetch(u, (line + 0.5) / p.sourceLines);
                    const lum = clamp01(dot3(c3, [0.2126, 0.7152, 0.0722]));
                    const w = p.beamMax + (p.beamMin - p.beamMax) * Math.sqrt(lum);
                    const d = (yy - line) / w;
                    col = col.map((cc, ch) => cc + (c3[ch]! * Math.exp(-0.5 * d * d)) / (w * 2.5066283)) as V3;
                }
                let ph = x % 3;
                if (p.mask === "slot" && ((Math.floor(y / 2) + Math.floor(x / 3)) & 1) !== 0) ph = (ph + 1) % 3;
                if (p.mask === "dot") ph = (x + y) % 3;
                col = col.map((cc, ch) => cc * (1 + ((ch === ph ? 3 : 0) - 1) * p.maskStrength)) as V3;
                if (p.halation > 0) {
                    const g: V3 = [0, 0, 0];
                    for (let j = -3; j <= 3; j++) for (let i2 = -3; i2 <= 3; i2++) { const s = fetch(u + i2 * (2 / kSize), v + j * (2 / kSize)); for (let ch = 0; ch < 3; ch++) g[ch]! += s[ch]!; }
                    col = col.map((cc, ch) => cc + (p.halation * g[ch]!) / 49) as V3;
                }
                const vv = u * (1 - u) * v * (1 - v) * 16;
                col = col.map((cc) => cc * (1 + (clamp01(Math.pow(vv, 0.3)) - 1) * p.vignette) * p.brightness) as V3;
            }
            const s0 = rgb(src, i);
            return s0.map((sv, ch) => sv + (col[ch]! - sv) * p.strength) as V3;
        }, 5e-4);
        log(`CRT[${p.mask}, ${p.sourceLines} lines, curvature ${p.curvature}]: max err ${r.maxErr.toExponential(2)}, share > 5e-4 ${(100 * r.badShare).toFixed(3)} %`);
        expectEq(r.badShare < 5e-3, true, "matches the CPU reference");
    }
    const z = await litRun(device, ["CRT", "C", { strength: 0 }]);
    expectEq(bitwiseEqual(z.out, z.src), true, "strength 0 is the identity");
});

gpuTest("VHS.chromaAndDeterminism", async ({ device }) => {
    // deterministic part: no jitter, no tracking, no noise -> exact YIQ chroma blur + offset
    const p = { chromaBlur: 3, chromaOffset: 2, lineJitter: 0, tracking: 0, noise: 0, strength: 1 };
    const { out, src } = await litRun(device, ["VHS", "V", p]);
    const yiq = (c: V3): V3 => [dot3(c, [0.299, 0.587, 0.114]), dot3(c, [0.596, -0.274, -0.322]), dot3(c, [0.211, -0.523, 0.312])];
    const r = compareRgb(out, (i) => {
        const x = i % kSize, y = Math.floor(i / kSize);
        const at = (xx: number) => rgb(src, y * kSize + Math.min(Math.max(xx, 0), kSize - 1));
        const Y = yiq(at(x))[0];
        let I = 0, Q = 0;
        for (let t = -p.chromaBlur; t <= p.chromaBlur; t++) { const c = yiq(at(x + p.chromaOffset + t)); I += c[1]; Q += c[2]; }
        I /= 2 * p.chromaBlur + 1; Q /= 2 * p.chromaBlur + 1;
        return [Y + 0.956 * I + 0.621 * Q, Y - 0.272 * I - 0.647 * Q, Y - 1.106 * I + 1.703 * Q].map((v) => Math.max(v, 0)) as V3;
    }, 1e-4);
    log(`VHS[deterministic chroma]: max err ${r.maxErr.toExponential(2)}, share > 1e-4 ${(100 * r.badShare).toFixed(3)} %`);
    expectEq(r.badShare < 2e-3, true, "matches the CPU reference");
    // animated part: same frame index -> same image; next frame -> different image
    const a = await litRun(device, ["VHS", "V", { frameOffset: 5, lineJitter: 2, tracking: 1, noise: 0.08 }]);
    const b = await litRun(device, ["VHS", "V", { frameOffset: 5, lineJitter: 2, tracking: 1, noise: 0.08 }]);
    expectEq(bitwiseEqual(a.out, b.out), true, "deterministic per frame index");
    a.g.execute(device.renderContext);
    const a2 = await readOut(device, a.g, "V.dst");
    expectEq(bitwiseEqual(a.out, a2), false, "next frame differs (jitter, tracking, noise)");
    const z = await litRun(device, ["VHS", "V", { strength: 0 }]);
    expectEq(bitwiseEqual(z.out, z.src), true, "strength 0 is the identity");
});

gpuTest("HoldFrames.holdsFrames", async ({ device }) => {
    const scene = makeScene(device);
    for (const [hold, phase] of [[2, 0], [3, 1], [1, 0]] as const) {
        const g = await runGraph(device, scene, [["HoldFrames", "H", { hold, phase }]], [["G.diffuseOpacity", "H.src"]], ["H.dst", "G.diffuseOpacity"], 0);
        // runGraph executed the graph k >= 1 times; restart the hold counter by recreating the pass via a new graph run below
        const frames: Float32Array[] = [], inputs: Float32Array[] = [];
        for (let f = 0; f < 7; f++) {
            scene.camera.setPosition(new float3(2.6 + 0.15 * f, 2.4, 3.4));
            g.execute(device.renderContext);
            frames.push(await readOut(device, g, "H.dst"));
            inputs.push(await readOut(device, g, "G.diffuseOpacity"));
        }
        // every output equals the input of some earlier-or-same frame; a new input appears exactly every `hold` frames
        let fresh = 0;
        const idx: number[] = [];
        for (let f = 0; f < 7; f++) {
            const k = inputs.findIndex((x, j) => j <= f && bitwiseEqual(frames[f]!, x));
            idx.push(k);
            if (k === f) fresh++;
        }
        log(`HoldFrames[hold ${hold}, phase ${phase}]: source frame per output ${idx.join(",")}`);
        expectEq(idx.every((k) => k >= 0), true, "every output is a (possibly held) input");
        expectEq(idx.every((k, f) => f - k < hold), true, "held for less than `hold` frames");
        expectEq(hold === 1 ? fresh === 7 : fresh >= Math.floor(7 / hold) && fresh <= Math.ceil(7 / hold) + 1, true, "fresh frames every `hold` frames");
    }
    makeScene(device); // reset the camera for later tests
});
