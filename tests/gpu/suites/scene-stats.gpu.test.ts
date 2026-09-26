/**
 * Scene.stats / getSceneStats against native `scene.stats` (tests/oracle/render-native-scene-stats.py).
 * Counts must match; memory sizes follow the web's own layouts (48-byte vertices, 32-bit indices,
 * flattened instances, software BVH), so they are only logged, as are transformCount (no scene graph
 * here). Vertex counts are exact except the residuals below.
 */

import { AssetCategory, AssetResolver, initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const kExact = [
    "meshCount", "meshInstanceCount", "meshInstanceOpaqueCount", "uniqueTriangleCount", "uniqueVertexCount", "instancedTriangleCount", "instancedVertexCount",
    "curveCount", "curveInstanceCount", "uniqueCurveSegmentCount", "sdfGridCount", "sdfGridInstancesCount", "customPrimitiveCount",
    "materialCount", "materialOpaqueCount", "materialMemoryInBytes", "textureCount", "textureCompressedCount", "textureTexelCount",
    "activeLightCount", "totalLightCount", "pointLightCount", "directionalLightCount", "rectLightCount", "discLightCount", "sphereLightCount", "distantLightCount",
    "gridVolumeCount", "gridCount", "gridVoxelCount",
];

/**
 * Known vertex-count residuals: the spheres scenes' createSphere poles merge
 * slightly differently (browser vs glibc sin/cos last bits change which pole vertices compare equal); ZeroDay's
 * post-MikkTSpace vertex merge keeps 3 of 1.47M vertices more (tangent last bits).
 */
const kVertexResiduals: Record<string, number> = { "inv_rendering_scenes/spheres_material_init.pyscene": 1e-3, "inv_rendering_scenes/spheres_material_ref.pyscene": 1e-3, "ZeroDay_v1/ZeroDay.pyscene": 1e-5 };

/** EXTRA_QUERY=bigScenes=1: the large production scenes (Bistro, EmeraldSquare, SunTemple, ZeroDay) instead. */
const kBigScenes = new URLSearchParams(location.search).get("bigScenes") === "1";

gpuTest("Scene.statsMatchNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const native = (await (await fetch(`/tests/oracle/out-native/scene-stats${kBigScenes ? "-big" : ""}.json`)).json()) as Record<string, Record<string, number>>;
    const bad: string[] = [];
    for (const [path, want] of Object.entries(native)) {
        const url = await AssetResolver.getDefaultResolver().resolvePath(path, AssetCategory.Scene);
        const scene = await runSceneScript(device, await (await fetch(url)).text(), url.slice(0, url.lastIndexOf("/")), { path: url });
        const got = scene.stats as unknown as Record<string, number>;
        for (const k of kExact) {
            if (got[k] === want[k]) continue;
            const tol = k.endsWith("VertexCount") ? kVertexResiduals[path] : undefined;
            if (tol !== undefined && Math.abs(got[k]! - want[k]!) <= tol * want[k]!) console.error(`# scene-stats ${path} ${k}: ${got[k]} vs ${want[k]} (known residual)`);
            else bad.push(`${path} ${k}: ${got[k]} vs ${want[k]}`);
        }
        const logged = Object.keys(want).filter((k) => !kExact.includes(k) && got[k] !== want[k]).map((k) => `${k} ${got[k]}/${want[k]}`);
        console.error(`# scene-stats ${path}: other fields web/native: ${logged.join(", ")}`);
        const text = scene.getSceneStatsText();
        // Native prints the last import path (a pyscene's imported asset comes after the pyscene itself).
        const last = scene.importPaths.at(-1) ?? url;
        if (!(text.startsWith(`Path: ${last}`) && text.includes(`  Mesh count: ${want["meshCount"]}`))) bad.push(`${path}: stats text ${JSON.stringify(text.split("\n").slice(0, 2))}`);
        scene.destroy();
    }
    for (const line of bad) console.error(`# scene-stats mismatch ${line}`);
    expectEq(bad.length, 0, `mismatches: ${bad.join("; ")}`);
});
