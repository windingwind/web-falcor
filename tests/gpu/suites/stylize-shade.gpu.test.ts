/**
 * StyleShade (deferred re-shading with a per-material style table).
 * Scene: floor (material 0) + UV sphere (material 1), one directional light, constant sky.
 *  1. Identity: styleMix 0, and the default record at styleMix 1, give DeferredLighting's LambertGGX colour (to fp32 rounding)
 *     (shadow map visibility passed through, and without shadows); hardSpec leaves the diffuse output unchanged.
 *  2. CPU parity of every diffuse/specular model, the hemisphere/flat mixes and a per-material override fed by
 *     StyleAovs.materialId, from the G-buffer readbacks (max abs error < 1e-4 away from the culling boundary).
 *  3. Ramp/band properties are rejected.
 */

import { EnvMap, LightType, Scene, createPass, float2, float3, float4, type Device } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { kSize, log, norm3, readAll, runGraph, type V3 } from "./stylize-helpers.js";

const kLightDirW: V3 = [0.3, -1, 0.2];
const kLi = 3;
const kSky: V3 = [0.3, 0.35, 0.45];
const kMinCos = 1e-6;

function sphere(r: number, c: V3, nLon: number, nLat: number, materialID: number) {
    const vertices = [];
    for (let a = 0; a <= nLat; a++) {
        const th = (Math.PI * a) / nLat;
        for (let b = 0; b <= nLon; b++) {
            const ph = (2 * Math.PI * b) / nLon;
            const n: V3 = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
            vertices.push({ position: new float3(c[0] + n[0] * r, c[1] + n[1] * r, c[2] + n[2] * r), normal: new float3(...n), tangent: new float4(-Math.sin(ph), 0, Math.cos(ph), 1), texCrd: new float2(b / nLon, a / nLat) });
        }
    }
    const idx: number[] = [];
    for (let a = 0; a < nLat; a++) {
        for (let b = 0; b < nLon; b++) {
            const i0 = a * (nLon + 1) + b, i1 = i0 + 1, i2 = i0 + nLon + 1, i3 = i2 + 1;
            idx.push(i0, i1, i2, i1, i3, i2);
        }
    }
    return { vertices, indices: new Uint32Array(idx), materialID };
}

const kScenes = new WeakMap<Device, Scene>();

/** One scene per device (see stylize-helpers: a second identical Scene can render an empty G-buffer). */
function shadeScene(device: Device): Scene {
    let scene = kScenes.get(device);
    if (!scene) {
        const f = 5, uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
        const floor = {
            vertices: ([[-f, 0, f], [f, 0, f], [f, 0, -f], [-f, 0, -f]] as V3[]).map((p, i) => ({ position: new float3(...p), normal: new float3(0, 1, 0), tangent: new float4(1, 0, 0, 1), texCrd: uv[i]! })),
            indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
            materialID: 0,
        };
        const mats = [
            { basic: { baseColor: new float4(0.7, 0.7, 0.7, 1), specular: new float4(0, 0.6, 0, 0) } },
            { basic: { baseColor: new float4(0.2, 0.5, 0.8, 1), specular: new float4(0, 0.35, 0, 0) } },
        ];
        scene = new Scene(device, [floor, sphere(0.7, [0.2, 0.7, -0.2], 96, 48, 1)], mats, [{ type: LightType.Directional, dirW: new float3(...kLightDirW), intensity: new float3(kLi, kLi, kLi) }]);
        scene.setEnvMap(new EnvMap(device, { width: 1, height: 1, data: new Float32Array([...kSky, 1]) } as never));
        kScenes.set(device, scene);
    }
    scene.camera.setPosition(new float3(2.0, 1.7, 2.3));
    scene.camera.setTarget(new float3(0.2, 0.5, -0.2));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
    return scene;
}

// ---------------------------------------------------------------------------
// 1. Identity vs DeferredLighting

const kDlEdges: [string, string][] = [["G.vbuffer", "L.vbuffer"], ["G.viewW", "L.viewW"], ["G.vbuffer", "S.vbuffer"], ["G.viewW", "S.viewW"]];

async function identityCase(device: Device, scene: Scene, shadows: boolean, ssProps: Record<string, unknown>) {
    const passes: [string, string, Record<string, unknown>][] = [];
    const edges = [...kDlEdges];
    if (shadows) {
        passes.push(["CascadedShadowMap", "CSM", { cascadeCount: 2, mapSize: 1024 }]);
        edges.push(["CSM.shadowMap", "L.shadowMap"], ["L.visibility", "S.visibility"]);
    }
    passes.push(["DeferredLighting", "L", { brdf: "LambertGGX", shadowMode: shadows ? "ShadowMap" : "None" }], ["StyleShade", "S", ssProps]);
    const outs = ["L.color", "S.color", "L.diffuse", "S.diffuse", "L.specular", "S.specular", "L.visibility"];
    const g = await runGraph(device, scene, passes, edges, outs);
    const [lc, sc, ld, sdf, ls, ss, vis] = await readAll(device, g, outs);
    let shadowed = 0;
    for (let i = 0; i < kSize * kSize; i++) if (vis![i]! < 0.5) shadowed++;
    // Max abs difference, number of differing values, p99 of the per-pixel max-over-RGB difference on geometry pixels.
    const diff = (a: Float32Array, b: Float32Array) => {
        let max = 0, n = 0;
        const px: number[] = [];
        for (let i = 0; i < kSize * kSize; i++) {
            let m = 0;
            for (let c = 0; c < 4; c++) if (a[i * 4 + c] !== b[i * 4 + c]) { n++; m = Math.max(m, Math.abs(a[i * 4 + c]! - b[i * 4 + c]!)); }
            max = Math.max(max, m);
            px.push(m);
        }
        px.sort((x, y) => x - y);
        return { bitwise: n === 0, max, n, p99: px[Math.floor(0.99 * (px.length - 1))]! };
    };
    const color = diff(lc!, sc!), diffuse = diff(ld!, sdf!), specular = diff(ls!, ss!);
    const f = (d: ReturnType<typeof diff>) => `${d.bitwise ? "bitwise" : `${d.n} values differ, max ${d.max.toExponential(2)}, p99 ${d.p99.toExponential(2)}`}`;
    log(`  identity case ${shadows ? "shadow map" : "no shadows"} ${JSON.stringify(ssProps)}: color ${f(color)}; diffuse ${f(diffuse)}; specular ${f(specular)}`);
    return { color, diffuse, specular, shadowed };
}

gpuTest("StyleShade.identityVsDeferredLighting", async ({ device }) => {
    const scene = shadeScene(device);
    const a = await identityCase(device, scene, true, { styleMix: 0, diffuseModel: "wrap", wrap: 0.8, specModel: "hardSpec" });
    const b = await identityCase(device, scene, true, {});
    const c = await identityCase(device, scene, false, {});
    const d = await identityCase(device, scene, true, { specModel: "hardSpec", shininess: 64 });
    // The two shaders evaluate the same expressions but are compiled separately, so the GPU compiler may contract or
    // reorder differently: differences at the fp32 rounding level are accepted (identity rule: far below tau2 = 0.13 p99
    // and below one 8-bit step), anything larger fails.
    const kUlpTol = 1e-6;
    log(`StyleShade identity: styleMix 0 max ${a.color.max.toExponential(2)}; default record max ${b.color.max.toExponential(2)} (shadowed px ${b.shadowed}); no shadows max ${c.color.max.toExponential(2)}; hardSpec diffuse max ${d.diffuse.max.toExponential(2)}, colour change max ${d.color.max.toExponential(2)}`);
    expectEq(b.shadowed > 200, true, "the shadow map shadows part of the floor");
    expectEq(a.color.max <= kUlpTol, true, "styleMix 0 = DeferredLighting LambertGGX (fp32 rounding level)");
    expectEq(b.color.max <= kUlpTol && b.diffuse.max <= kUlpTol && b.specular.max <= kUlpTol, true, "default record = DeferredLighting LambertGGX (shadow map)");
    expectEq(c.color.max <= kUlpTol, true, "default record = DeferredLighting LambertGGX (no shadows)");
    expectEq(d.diffuse.max <= kUlpTol && d.color.max > 1e-2, true, "hardSpec changes the highlight only (diffuse untouched)");
});

gpuTest("StyleShade.rejectsRampProperties", async ({ device }) => {
    const bad: Record<string, unknown>[] = [{ rampTexture: "x" }, { bandCount: 3 }, { materials: [{ materialId: 0, rampColors: [0, 1] }] }, { diffuseModel: "ramp" }];
    let rejected = 0;
    for (const p of bad) {
        try { createPass(device, "StyleShade", p); } catch { rejected++; }
    }
    log(`StyleShade: ramp/band properties rejected ${rejected}/${bad.length}`);
    expectEq(rejected, bad.length, "no ramp/band mode");
});

// ---------------------------------------------------------------------------
// 2. CPU parity

interface Rec { diffuseModel: string; wrap: number; minnaertK: number; specModel: string; skyColor: V3; groundColor: V3; hemiMix: number; flatMix: number; shininess: number; hardThreshold: number; hardSoftness: number; specScale: number }
const kDef: Rec = { diffuseModel: "lambert", wrap: 0.5, minnaertK: 1.2, specModel: "ggx", skyColor: [0.6, 0.7, 0.9], groundColor: [0.25, 0.2, 0.15], hemiMix: 0, flatMix: 0, shininess: 0, hardThreshold: 0.5, hardSoftness: 0.05, specScale: 1 };

const sstep = (e0: number, e1: number, x: number) => { const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1); return t * t * (3 - 2 * t); };
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

/** CPU StyleShade for one pixel (no env light, no shadows, one directional light). Returns rgb and a "near a threshold" flag. */
function cpuPixel(r: Rec, N: V3, V: V3, faceN: V3, albedo: V3, F0: V3, rough: number): { rgb: V3; edge: boolean } {
    const L = norm3(kLightDirW.map((x) => -x));
    const NoL = dot(N, L), NoV = dot(N, V);
    const fN = dot(faceN, V) >= 0 ? faceN : (faceN.map((x) => -x) as V3);
    const culled = dot(fN, L) <= kMinCos;
    const edge = Math.abs(NoL) < 1e-3 || Math.abs(NoV) < 1e-3 || Math.abs(dot(fN, L)) < 1e-3;
    // reference GGX
    let ggx: V3 = [0, 0, 0];
    let lam = 0;
    if (!culled && NoL > kMinCos && NoV > kMinCos) {
        lam = NoL / Math.PI;
        const H = norm3([V[0] + L[0], V[1] + L[1], V[2] + L[2]]);
        const NoH = Math.min(Math.max(dot(N, H), 0), 1), VoH = Math.min(Math.max(dot(V, H), 0), 1);
        const a = Math.max(rough * rough, 0.0064), a2 = a * a;
        const dd = NoH * NoH * (a2 - 1) + 1, D = a2 / (Math.PI * dd * dd);
        const vis = 0.5 / (NoL * Math.sqrt(NoV * NoV * (1 - a2) + a2) + NoV * Math.sqrt(NoL * NoL * (1 - a2) + a2));
        if (rough > 0) ggx = F0.map((f) => (f + (1 - f) * Math.pow(1 - VoH, 5)) * D * vis * NoL) as V3;
    }
    let dfac = lam;
    if (r.diffuseModel === "wrap") dfac = NoV > kMinCos ? Math.max((NoL + r.wrap) / (1 + r.wrap), 0) / Math.PI : 0;
    else if (r.diffuseModel === "halfLambert") dfac = NoV > kMinCos ? Math.pow(0.5 * NoL + 0.5, 2) / Math.PI : 0;
    else if (r.diffuseModel === "minnaert") dfac = NoV > kMinCos && NoL > 0 ? (Math.pow(NoL, r.minnaertK) * Math.pow(Math.max(NoV, 1e-4), r.minnaertK - 1)) / Math.PI : 0;
    let spec: V3 = ggx;
    let near = false;
    if (r.specModel !== "ggx") {
        spec = [0, 0, 0];
        if (!culled && NoL > kMinCos && NoV > kMinCos) {
            const H = norm3([V[0] + L[0], V[1] + L[1], V[2] + L[2]]);
            const NoH = Math.min(Math.max(dot(N, H), 0), 1);
            const a = Math.max(rough * rough, 0.0064);
            const n = r.shininess > 0 ? r.shininess : Math.max(2 / (a * a) - 2, 1);
            const lobe = Math.pow(NoH, n);
            if (r.specModel === "blinn") spec = F0.map((f) => (f * (n + 8)) / (8 * Math.PI) * lobe * NoL) as V3;
            else {
                spec = F0.map((f) => f * sstep(r.hardThreshold - r.hardSoftness, r.hardThreshold + r.hardSoftness, lobe)) as V3;
                near = Math.abs(lobe - r.hardThreshold) < r.hardSoftness + 1e-3 && r.hardSoftness < 1e-3;
            }
        }
    }
    const t = 0.5 * N[1] + 0.5;
    const rgb = [0, 1, 2].map((c) => {
        const direct = albedo[c]! * dfac * kLi;
        const hemi = albedo[c]! * (r.groundColor[c]! + (r.skyColor[c]! - r.groundColor[c]!) * t);
        const d = direct + hemi * r.hemiMix; // ambient (env) is off here
        const dd = d + (albedo[c]! - d) * r.flatMix;
        return dd + spec[c]! * kLi * r.specScale;
    }) as V3;
    return { rgb, edge: edge || near };
}

const kGOut = ["G.normW", "G.viewW", "G.faceNormalW", "G.diffuseOpacity", "G.specRough", "G.posW"];

gpuTest("StyleShade.matchesCpuReference", async ({ device }) => {
    const scene = shadeScene(device);
    const cases: { name: string; rec: Partial<Rec>; sphere?: Partial<Rec> }[] = [
        { name: "lambert/ggx", rec: {} },
        { name: "wrap/ggx", rec: { diffuseModel: "wrap", wrap: 0.5 } },
        { name: "halfLambert/blinn", rec: { diffuseModel: "halfLambert", specModel: "blinn", shininess: 32 } },
        { name: "minnaert/hardSpec", rec: { diffuseModel: "minnaert", minnaertK: 1.5, specModel: "hardSpec", shininess: 64, hardThreshold: 0.4, hardSoftness: 0.05 } },
        { name: "blinn(roughness)/hemi/flat/specScale", rec: { specModel: "blinn", hemiMix: 0.7, flatMix: 0.3, specScale: 2 } },
        { name: "override(sphere=wrap/hardSpec via StyleAovs)", rec: { diffuseModel: "halfLambert" }, sphere: { diffuseModel: "wrap", wrap: 0.8, specModel: "hardSpec", shininess: 48, hardThreshold: 0.6, hardSoftness: 0.1, hemiMix: 0.4 } },
    ];
    let worstAll = 0;
    for (const cs of cases) {
        const props: Record<string, unknown> = { ...cs.rec, envLighting: false, envBackground: false };
        const passes: [string, string, Record<string, unknown>][] = [["StyleShade", "S", props]];
        const edges: [string, string][] = [["G.vbuffer", "S.vbuffer"], ["G.viewW", "S.viewW"]];
        const outs = ["S.color", ...kGOut];
        if (cs.sphere) {
            props["materials"] = [{ materialId: 1, ...cs.sphere }];
            passes.unshift(["StyleAovs", "A", {}]);
            edges.push(["G.vbuffer", "A.vbuffer"], ["A.materialId", "S.materialId"]);
            outs.push("A.materialId");
        }
        const g = await runGraph(device, scene, passes, edges, outs);
        const [col, nrm, view, face, dif, spr, pos, mid] = await readAll(device, g, outs);
        let maxErr = 0, n = 0, skipped = 0, sphereN = 0;
        for (let i = 0; i < kSize * kSize; i++) {
            if (pos![i * 4 + 3] === 0) continue;
            const v3 = (b: Float32Array): V3 => [b[i * 4]!, b[i * 4 + 1]!, b[i * 4 + 2]!];
            let rec: Rec = { ...kDef, ...cs.rec };
            if (cs.sphere && mid![i * 4 + 3]! > 0 && Math.round(mid![i * 4 + 1]!) === 1) { rec = { ...rec, ...cs.sphere }; sphereN++; }
            const r = cpuPixel(rec, norm3(v3(nrm!)), norm3(v3(view!)), norm3(v3(face!)), v3(dif!), v3(spr!), spr![i * 4 + 3]!);
            if (r.edge) { skipped++; continue; }
            n++;
            for (let c = 0; c < 3; c++) maxErr = Math.max(maxErr, Math.abs(col![i * 4 + c]! - r.rgb[c]!));
        }
        worstAll = Math.max(worstAll, maxErr);
        log(`StyleShade[${cs.name}]: px ${n}, skipped (threshold) ${skipped}${cs.sphere ? `, override px ${sphereN}` : ""}, max abs err ${maxErr.toExponential(2)}`);
        expectEq(n > 0.3 * kSize * kSize, true, `${cs.name}: enough pixels`);
        if (cs.sphere) expectEq(sphereN > 1000, true, "override applies to the sphere's pixels");
        expectEq(maxErr < 1e-4, true, `${cs.name}: GPU = CPU (max abs err ${maxErr})`);
    }
    log(`StyleShade CPU parity: worst max abs err ${worstAll.toExponential(2)}`);
});
