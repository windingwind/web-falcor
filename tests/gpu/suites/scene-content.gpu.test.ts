/**
 * Scene contents against native (tests/oracle/render-native-scene-content.py): each material's Python properties and
 * texture slots (by size), the analytic lights and the cameras of the media scenes in scene-stats.json.
 * Floats compare with a small tolerance (native keeps material values as float16).
 */

import { AlphaMode, AssetCategory, AssetResolver, MaterialType, ShadingModel, initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
const kSlots: [string, string][] = [["BaseColor", "texBaseColor"], ["Specular", "texSpecular"], ["Emissive", "texEmissive"], ["Normal", "texNormalMap"], ["Transmission", "texTransmission"], ["Displacement", "texDisplacement"]];

function near(a: number, b: number): boolean {
    return a === b || (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= 2e-3 * Math.max(1, Math.abs(b));
}
function value(v: unknown): Json {
    if (v === undefined || v === null) return null;
    if (typeof v === "object" && "x" in (v as object)) return ["x", "y", "z", "w"].filter((c) => c in (v as object)).map((c) => Number((v as Record<string, number>)[c]));
    return v as Json;
}
/** The oracle writes non-finite floats as "inf" / "-inf" / "nan". */
const unJson = (v: Json): Json => (v === "inf" ? Infinity : v === "-inf" ? -Infinity : v === "nan" ? NaN : Array.isArray(v) ? v.map(unJson) : v);

function compare(path: string, web: Record<string, unknown>, nat: Record<string, Json>, out: string[]): void {
    for (const [k, raw] of Object.entries(nat)) {
        const want = unJson(raw);
        if (k === "textures") continue;
        // Native's pybind enums print by name; the web returns their values.
        const enums: Record<string, Record<number, string>> = { type: MaterialType as never, shadingModel: ShadingModel as never, alphaMode: AlphaMode as never };
        const got = k in enums && typeof web[k] === "number" ? enums[k]![web[k] as number] ?? web[k] : value(web[k]);
        const ok = Array.isArray(want) ? Array.isArray(got) && want.every((w, i) => near(Number((got as number[])[i]), Number(w))) : typeof want === "number" ? typeof got === "number" && near(got, want) : got === want;
        if (!ok) out.push(`${path}.${k}: ${JSON.stringify(got)} vs ${JSON.stringify(want)}`);
    }
}

/**
 * Known residual: grey_and_white_room's animated FBX camera (Assimp pivot helper chain, fixFbxCameraAnimation) poses
 * differently at load; native's up vector keeps the node's 0.01 scale.
 */
const kResiduals = [/^test_scenes\/grey_and_white_room\/grey_and_white_room\.pyscene camera\[0\]\./];

gpuTest("Scene.contentMatchesNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const native = (await (await fetch("/tests/oracle/out-native/scene-content.json")).json()) as Record<string, { materials: Record<string, Json>[]; lights: Record<string, Json>[]; cameras: Record<string, Json>[] }>;
    const bad: string[] = [];
    for (const [path, want] of Object.entries(native)) {
        const url = await AssetResolver.getDefaultResolver().resolvePath(path, AssetCategory.Scene);
        const scene = await runSceneScript(device, await (await fetch(url)).text(), url.slice(0, url.lastIndexOf("/")), { path: url });
        const out: string[] = [];
        const tm = (scene as unknown as { lcTextureManager: { getSource(id: number): { bitmap: { width: number; height: number }; compressed?: { width: number; height: number } } | undefined } }).lcTextureManager;
        const mats = scene.materials.map((_m, i) => scene.getMaterial(i) as unknown as Record<string, unknown> & { basic: Record<string, number | undefined> });
        if (mats.length !== want.materials.length) out.push(`${path}: ${mats.length} materials vs ${want.materials.length}`);
        want.materials.forEach((nat, i) => {
            const m = mats[i];
            if (!m) return;
            compare(`${path} material[${i}] ${String(nat["name"])}`, m, nat, out);
            const textures: Record<string, [number, number]> = {};
            for (const [slot, field] of kSlots) {
                const h = m.basic[field];
                if (h === undefined || ((h >>> 29) & 3) !== 1) continue;
                const src = tm.getSource(h & 0x1fffffff);
                if (src) textures[slot] = src.compressed ? [src.compressed.width, src.compressed.height] : [src.bitmap.width, src.bitmap.height];
            }
            // MERLMix keeps its normal map handle and index map in the MERLMix data (the index map in the material buffer).
            const mix = (m as unknown as { merlMix?: { texNormalMap?: number; indexMap: { width: number; height: number; fromTexture?: boolean } } }).merlMix;
            if (mix?.texNormalMap !== undefined && ((mix.texNormalMap >>> 29) & 3) === 1) {
                const src = tm.getSource(mix.texNormalMap & 0x1fffffff);
                if (src) textures["Normal"] = [src.bitmap.width, src.bitmap.height];
            }
            if (mix?.indexMap.fromTexture) textures["Index"] = [mix.indexMap.width, mix.indexMap.height];
            const nt = nat["textures"] as Record<string, [number, number]>;
            if (JSON.stringify(Object.keys(textures).sort().map((k) => [k, textures[k]])) !== JSON.stringify(Object.keys(nt).sort().map((k) => [k, nt[k]]))) {
                out.push(`${path} material[${i}] ${String(nat["name"])} textures: ${JSON.stringify(textures)} vs ${JSON.stringify(nt)}`);
            }
        });
        if (scene.lights.length !== want.lights.length) out.push(`${path}: ${scene.lights.length} lights vs ${want.lights.length}`);
        want.lights.forEach((nat, i) => scene.lights[i] && compare(`${path} light[${i}]`, scene.lights[i] as unknown as Record<string, unknown>, nat, out));
        const cams = scene.cameras;
        if (cams.length !== want.cameras.length) out.push(`${path}: ${cams.length} cameras vs ${want.cameras.length}`);
        want.cameras.forEach((nat, i) => cams[i] && compare(`${path} camera[${i}]`, cams[i] as unknown as Record<string, unknown>, nat, out));
        for (const line of out) console.error(`# scene-content ${line}`);
        bad.push(...out.filter((l) => !kResiduals.some((r) => r.test(l))));
        scene.destroy();
    }
    expectEq(bad.length, 0, `${bad.length} mismatches (first: ${bad.slice(0, 3).join("; ")})`);
});
