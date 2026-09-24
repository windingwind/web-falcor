/**
 * Scene.stats / getSceneStats against native `scene.stats` (tests/oracle/render-native-scene-stats.py).
 * Counts must match; memory sizes follow the web's own layouts (48-byte vertices, 32-bit indices,
 * flattened instances, software BVH), so they are only logged, as are transformCount (no scene graph
 * here) and the vertex counts: native MikkTSpace splits vertices at UV seams, the web keeps them shared.
 */

import { AssetCategory, AssetResolver, initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const kExact = [
    "meshCount", "meshInstanceCount", "meshInstanceOpaqueCount", "uniqueTriangleCount", "instancedTriangleCount",
    "curveCount", "curveInstanceCount", "uniqueCurveSegmentCount", "sdfGridCount", "sdfGridInstancesCount", "customPrimitiveCount",
    "materialCount", "materialOpaqueCount", "materialMemoryInBytes", "textureCount", "textureCompressedCount", "textureTexelCount",
    "activeLightCount", "totalLightCount", "pointLightCount", "directionalLightCount", "rectLightCount", "discLightCount", "sphereLightCount", "distantLightCount",
    "gridVolumeCount", "gridCount", "gridVoxelCount",
];

gpuTest("Scene.statsMatchNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const native = (await (await fetch("/tests/oracle/out-native/scene-stats.json")).json()) as Record<string, Record<string, number>>;
    const bad: string[] = [];
    for (const [path, want] of Object.entries(native)) {
        const url = await AssetResolver.getDefaultResolver().resolvePath(path, AssetCategory.Scene);
        const scene = await runSceneScript(device, await (await fetch(url)).text(), url.slice(0, url.lastIndexOf("/")), { path: url });
        const got = scene.stats as unknown as Record<string, number>;
        for (const k of kExact) if (got[k] !== want[k]) bad.push(`${path} ${k}: ${got[k]} vs ${want[k]}`);
        const logged = Object.keys(want).filter((k) => !kExact.includes(k) && got[k] !== want[k]).map((k) => `${k} ${got[k]}/${want[k]}`);
        console.error(`# scene-stats ${path}: other fields web/native: ${logged.join(", ")}`);
        const text = scene.getSceneStatsText();
        expectEq(text.startsWith(`Path: ${url}`) && text.includes(`  Mesh count: ${want["meshCount"]}`), true, `stats text for ${path}`);
    }
    expectEq(bad.length, 0, `mismatches: ${bad.join("; ")}`);
});
