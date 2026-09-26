/**
 * Importer plugins (native: Importer plugin classes picked by file extension): a module loaded with
 * loadPluginLibrary registers an importer through the webFalcorPlugins hook, and importScene uses it.
 */

import { getRegisteredImporter, initScripting, loadPluginLibrary, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

// A plugin for ".tri" text files (one triangle per line), written as a standalone module.
const kPluginSource = `
const { float2, float3, float4 } = await import("${location.origin}/packages/falcor/src/index.ts");
globalThis.webFalcorPlugins.registerImporter(["tri"], async (bytes) => {
    const lines = new TextDecoder().decode(bytes).split("\\n").filter((l) => l.trim() && !l.startsWith("#"));
    const vertices = [];
    for (const line of lines) {
        const v = line.trim().split(/\\s+/).map(Number);
        for (let k = 0; k < 3; k++) vertices.push({ position: new float3(v[k * 3], v[k * 3 + 1], v[k * 3 + 2]), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) });
    }
    return {
        meshes: [{ vertices, indices: Uint32Array.from(vertices.keys()), materialID: 0, tangentSpace: "keep" }],
        materials: [{ name: "TriPluginMaterial", basic: { baseColor: new float4(0.25, 0.5, 0.75, 1) } }],
        nodes: [], animations: [], lights: [], weightTracks: [],
    };
});
`;

gpuTest("ImporterPlugin.registersAndImports", async ({ device }) => {
    const url = URL.createObjectURL(new Blob([kPluginSource], { type: "text/javascript" }));
    const added = await loadPluginLibrary(url);
    URL.revokeObjectURL(url);
    expectEq(added.importers.join(","), "tri", "the plugin registered its extension");
    expectEq(getRegisteredImporter("scenes/x.TRI") !== undefined, true, "extension lookup is case-insensitive");

    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(device, 'sceneBuilder.importScene("plugin-quad.tri")', "/tests/oracle/assets");
    const stats = scene.stats as unknown as Record<string, number>;
    expectEq(stats.uniqueTriangleCount, 2, "the plugin's triangles");
    expectEq(scene.getMaterial(0).name, "TriPluginMaterial", "the plugin's material");
    scene.destroy();
});
