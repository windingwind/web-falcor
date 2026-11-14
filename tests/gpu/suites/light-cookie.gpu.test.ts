/**
 * DeferredLighting light cookie (optional `cookie` input) and the StyleAovs per-instance scalar AOV (`custom` output).
 *  1. defaultHash: prints hashes of DeferredLighting / StyleAovs outputs without cookie or custom output. Run on this
 *     branch and on the base commit; the hashes must be equal (the default path is unchanged).
 *  2. A white cookie gives bitwise the same radiance as no cookie.
 *  3. Directional and spot projections match a CPU reference (ratio lit-with-cookie / lit-without = cookie value).
 *  4. The custom AOV = instanceScalars[instance id]; the other StyleAovs outputs are bitwise unchanged by it.
 */

import { RenderGraph, ResourceFormat, createPass, float3, LightType, type Device, type Scene, type SceneLight, type Texture } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kSize, bitwiseEqual, log, makeScene, read, type V3 } from "./stylize-helpers.js";

let graphCount = 0;

/** GBufferRaster "G" + passes; `inputs` are bound with setInput before the first execute. */
async function run(device: Device, scene: Scene, passes: [string, string, Record<string, unknown>][], edges: [string, string][],
    outputs: string[], inputs: [string, Texture][] = []): Promise<Record<string, Float32Array>> {
    const g = new RenderGraph(device, `LightCookieTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    for (const [type, id, props] of passes) g.addPass(createPass(device, type, props), id);
    for (const [a, b] of edges) g.addEdge(a, b);
    for (const o of new Set([...outputs, "G.posW"])) g.markOutput(o);
    for (const [ref, t] of inputs) g.setInput(ref, t);
    g.setScene(scene);
    for (let tries = 0; ; tries++) {
        g.execute(device.renderContext);
        const pos = await read(device, g.getOutput("G.posW")!);
        let hits = 0;
        for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
        if (hits > 0.2 * kSize * kSize) break;
        if (tries >= 4) throw new Error(`G-buffer still empty after ${tries + 1} frames`);
    }
    g.execute(device.renderContext);
    const out: Record<string, Float32Array> = {};
    for (const o of new Set([...outputs, "G.posW"])) out[o] = (await read(device, g.getOutput(o)!)).slice();
    return out;
}

const lightingEdges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"]];

function fnv(a: Float32Array): string {
    const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
    let h = 0x811c9dc5;
    for (let i = 0; i < u.length; i++) {
        let v = u[i]!;
        for (let k = 0; k < 4; k++) { h = Math.imul(h ^ (v & 0xff), 0x01000193) >>> 0; v >>>= 8; }
    }
    return h.toString(16).padStart(8, "0");
}

/** RGBA32Float cookie texture (rgb = given values, a = 1). */
function cookieTexture(device: Device, w: number, h: number, f: (x: number, y: number) => V3): { tex: Texture; data: Float32Array } {
    const data = new Float32Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set([...f(x, y), 1], (y * w + x) * 4);
    return { tex: device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, data), data };
}

/** Deterministic pseudo-random cookie values in [0.25, 1]. */
const rnd = (x: number, y: number, c: number) => 0.25 + 0.75 * ((((x * 73856093) ^ (y * 19349663) ^ (c * 83492791)) >>> 0) % 1000) / 999;

/** CPU mirror of DeferredLighting.cs.slang cookieBasis / cookieLookup. */
function basis(n: V3): [V3, V3] {
    const sgn = n[2] >= 0 ? 1 : -1;
    const a = -1 / (sgn + n[2]);
    const b = n[0] * n[1] * a;
    return [[1 + sgn * n[0] * n[0] * a, sgn * b, -sgn * n[0]], [b, sgn + n[1] * n[1] * a, -n[1]]];
}
function lookup(data: Float32Array, w: number, h: number, u: number, v: number, tile: boolean): V3 {
    const tx = u * w - 0.5, ty = v * h - 0.5;
    const x0 = Math.floor(tx), y0 = Math.floor(ty), fx = tx - x0, fy = ty - y0;
    const at = (x: number, y: number, c: number) => {
        x = tile ? ((x % w) + w) % w : Math.min(Math.max(x, 0), w - 1);
        y = tile ? ((y % h) + h) % h : Math.min(Math.max(y, 0), h - 1);
        return data[(y * w + x) * 4 + c]!;
    };
    return [0, 1, 2].map((c) => {
        const a = at(x0, y0, c) + (at(x0 + 1, y0, c) - at(x0, y0, c)) * fx;
        const b = at(x0, y0 + 1, c) + (at(x0 + 1, y0 + 1, c) - at(x0, y0 + 1, c)) * fx;
        return a + (b - a) * fy;
    }) as V3;
}
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

/** Compares (lit with cookie) / (lit without) against the CPU cookie value on pixels with enough direct light. */
function checkRatio(withC: Float32Array, without: Float32Array, pos: Float32Array, ref: (p: V3) => V3): { n: number; maxErr: number } {
    let n = 0, maxErr = 0;
    for (let i = 0; i < kSize * kSize; i++) {
        if (pos[i * 4 + 3] === 0) continue;
        const p: V3 = [pos[i * 4]!, pos[i * 4 + 1]!, pos[i * 4 + 2]!];
        const c = ref(p);
        for (let k = 0; k < 3; k++) {
            const base = without[i * 4 + k]!;
            if (base < 1e-2) continue;
            n++;
            maxErr = Math.max(maxErr, Math.abs(withC[i * 4 + k]! / base - c[k]!));
        }
    }
    return { n, maxErr };
}

gpuTest("LightCookie.defaultHash", async ({ device }) => {
    const scene = makeScene(device, [0.3, 0.35, 0.4]);
    const configs: Record<string, unknown>[] = [
        { brdf: "Material", shadowMode: "RayTraced" },
        { brdf: "LambertGGX", shadowMode: "ShadowMap" },
        { brdf: "LambertGGX", shadowMode: "None", envLighting: false },
    ];
    for (const props of configs) {
        const passes: [string, string, Record<string, unknown>][] = [["DeferredLighting", "L", props]];
        const edges = [...lightingEdges];
        if (props["shadowMode"] === "ShadowMap") {
            passes.push(["CascadedShadowMap", "CSM", { cascadeCount: 2, mapSize: 1024 }]);
            edges.push(["CSM.shadowMap", "L.shadowMap"]);
        }
        const r = await run(device, scene, passes, edges, ["L.color", "L.visibility", "L.diffuse", "L.specular"]);
        log(`defaultHash DeferredLighting ${JSON.stringify(props)} color=${fnv(r["L.color"]!)} visibility=${fnv(r["L.visibility"]!)} diffuse=${fnv(r["L.diffuse"]!)} specular=${fnv(r["L.specular"]!)}`);
    }
    const outs = ["S.id", "S.idColor", "S.materialId", "S.flatAlbedo", "S.ndotl"];
    const r = await run(device, scene, [["StyleAovs", "S", { hashSeed: 3 }]], [["G.vbuffer", "S.vbuffer"], ["G.texC", "S.texC"], ["G.normW", "S.normW"]], outs);
    log(`defaultHash StyleAovs ${outs.map((o) => `${o}=${fnv(r[o]!)}`).join(" ")}`);
    expectEq(true, true, "hashes printed");
});

gpuTest("LightCookie.whiteCookieIsBitwise", async ({ device }) => {
    const scene = makeScene(device, [0.3, 0.35, 0.4]);
    const white = cookieTexture(device, 4, 4, () => [1, 1, 1]).tex;
    for (const props of [{ brdf: "Material", shadowMode: "RayTraced" }, { brdf: "LambertGGX", shadowMode: "RayTraced" }]) {
        const a = await run(device, scene, [["DeferredLighting", "L", props]], lightingEdges, ["L.color"]);
        const b = await run(device, scene, [["DeferredLighting", "L", { ...props, cookieWorldSize: 0.7, cookieOffset: [0.2, 0.4] }]], lightingEdges, ["L.color"], [["L.cookie", white]]);
        const same = bitwiseEqual(a["L.color"]!, b["L.color"]!);
        log(`white cookie ${JSON.stringify(props)}: bitwise ${same}`);
        expectEq(same, true, `white cookie leaves the radiance bitwise unchanged (${props.brdf})`);
    }
});

gpuTest("LightCookie.directionalProjectionMatchesCpu", async ({ device }) => {
    const scene = makeScene(device);
    const W = 8, H = 6, size = 1.7, off: [number, number] = [0.3, -0.15];
    const ck = cookieTexture(device, W, H, (x, y) => [rnd(x, y, 0), rnd(x, y, 1), rnd(x, y, 2)]);
    const props = { brdf: "LambertGGX", shadowMode: "None", envLighting: false, emissive: false };
    const a = await run(device, scene, [["DeferredLighting", "L", props]], lightingEdges, ["L.color"]);
    const b = await run(device, scene, [["DeferredLighting", "L", { ...props, cookieWorldSize: size, cookieOffset: off }]], lightingEdges, ["L.color"], [["L.cookie", ck.tex]]);
    const d = scene.getLight(0).dirW!;
    const len = Math.hypot(d.x, d.y, d.z);
    const [T, B] = basis([d.x / len, d.y / len, d.z / len]);
    const res = checkRatio(b["L.color"]!, a["L.color"]!, a["G.posW"]!, (p) => {
        let u = dot(p, T) / size + off[0], v = dot(p, B) / size + off[1];
        u -= Math.floor(u); v -= Math.floor(v);
        return lookup(ck.data, W, H, u, v, true);
    });
    log(`directional cookie: channels ${res.n}, max |ratio - cpu| ${res.maxErr.toExponential(2)}`);
    expectEq(res.n > 10000, true, `enough lit pixels (${res.n})`);
    expectEq(res.maxErr < 2e-3, true, `directional projection matches the CPU (${res.maxErr})`);
});

gpuTest("LightCookie.spotProjectionMatchesCpu", async ({ device }) => {
    const scene = makeScene(device);
    const light = scene.getLight(0) as SceneLight;
    const saved = { type: light.type, posW: light.posW, dirW: light.dirW, openingAngle: light.openingAngle, intensity: light.intensity };
    const lp: V3 = [0.3, 4.0, 0.8];
    const dir: V3 = (() => { const v: V3 = [-0.25, -1, -0.35]; const l = Math.hypot(...v); return v.map((x) => x / l) as V3; })();
    const angle = 0.45;
    try {
        light.type = LightType.Point;
        light.openingAngle = angle;
        light.posW = new float3(...lp);
        light.dirW = new float3(...dir);
        light.intensity = new float3(30, 30, 30);
        const W = 7, H = 9;
        const ck = cookieTexture(device, W, H, (x, y) => [rnd(x, y, 3), rnd(x, y, 4), rnd(x, y, 5)]);
        const props = { brdf: "LambertGGX", shadowMode: "None", envLighting: false, emissive: false };
        const a = await run(device, scene, [["DeferredLighting", "L", props]], lightingEdges, ["L.color"]);
        const b = await run(device, scene, [["DeferredLighting", "L", { ...props, cookieLight: 0 }]], lightingEdges, ["L.color"], [["L.cookie", ck.tex]]);
        const [T, B] = basis(dir);
        const t = Math.tan(angle);
        const res = checkRatio(b["L.color"]!, a["L.color"]!, a["G.posW"]!, (p) => {
            const w: V3 = [p[0] - lp[0], p[1] - lp[1], p[2] - lp[2]];
            const z = dot(w, dir);
            if (z <= 0) return [0, 0, 0];
            const u = 0.5 + 0.5 * dot(w, T) / (z * t), v = 0.5 + 0.5 * dot(w, B) / (z * t);
            if (u < 0 || v < 0 || u > 1 || v > 1) return [0, 0, 0];
            return lookup(ck.data, W, H, u, v, false);
        });
        log(`spot cookie: channels ${res.n}, max |ratio - cpu| ${res.maxErr.toExponential(2)}`);
        expectEq(res.n > 3000, true, `enough lit pixels in the cone (${res.n})`);
        expectEq(res.maxErr < 2e-3, true, `spot projection matches the CPU (${res.maxErr})`);
    } finally {
        light.type = saved.type;
        light.openingAngle = saved.openingAngle;
        light.posW = saved.posW;
        light.dirW = saved.dirW;
        light.intensity = saved.intensity;
    }
});

gpuTest("StyleAovs.customInstanceScalar", async ({ device }) => {
    const scene = makeScene(device);
    const outs = ["S.id", "S.idColor", "S.materialId", "S.flatAlbedo", "S.ndotl"];
    const edges: [string, string][] = [["G.vbuffer", "S.vbuffer"], ["G.texC", "S.texC"], ["G.normW", "S.normW"]];
    const scalars = [0.5, -2, 3.25, 0.125, 7, 1e-3, 42, -0.75, 9.5]; // fewer than the instances: the rest read 0
    const a = await run(device, scene, [["StyleAovs", "S", {}]], edges, outs);
    const b = await run(device, scene, [["StyleAovs", "S", { instanceScalars: scalars }]], edges, [...outs, "S.custom"]);
    let n = 0, bad = 0, bgBad = 0, past = 0;
    const c = b["S.custom"]!, id = b["S.id"]!;
    for (let i = 0; i < kSize * kSize; i++) {
        if (b["G.posW"]![i * 4 + 3] === 0) { if (c[i * 4] !== 0 || c[i * 4 + 3] !== 0) bgBad++; continue; }
        n++;
        const inst = id[i * 4 + 1]!;
        if (inst >= scalars.length) past++;
        const want = inst < scalars.length ? Math.fround(scalars[inst]!) : 0;
        if (c[i * 4] !== want || c[i * 4 + 3] !== 1) bad++;
    }
    const unchanged = outs.every((o) => bitwiseEqual(a[o]!, b[o]!));
    log(`custom AOV: geometry px ${n} (past the list ${past}), mismatches ${bad}, background bad ${bgBad}, other outputs bitwise unchanged ${unchanged}`);
    expectEq(n > 5000 && past > 0, true, "covers instances inside and past the scalar list");
    expectEq(bad, 0, "custom = instanceScalars[instance id]");
    expectEq(bgBad, 0, "background custom = 0");
    expectEq(unchanged, true, "other StyleAovs outputs unchanged by the custom output");
});
