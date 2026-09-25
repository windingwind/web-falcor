/**
 * SceneBuilder python API in pyscenes (docs/usage/scripting.md): getLight, lights, addMaterial,
 * getGridVolume, renderSettings (incl. diffuseAlbedoMultiplier, which reaches the scene defines).
 */

import { initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Scripting.sceneBuilderPythonApi", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(
        device,
        [
            "l = PointLight('Key')",
            "l.intensity = float3(2, 2, 2)",
            "sceneBuilder.addLight(l)",
            "assert sceneBuilder.getLight('Key') is not None",
            "assert sceneBuilder.getLight('missing') is None",
            "assert len(sceneBuilder.lights) == 1",
            "assert sceneBuilder.getGridVolume('none') is None",
            "unused = StandardMaterial('Unused')",
            "unused.baseColor = float4(1, 0, 0, 1)  # distinct, or native's duplicate merge folds it into 'Quad'",
            "assert sceneBuilder.addMaterial(unused) == 0",
            "m = StandardMaterial('Quad')",
            "m.alphaMode = AlphaMode.Mask",
            "m.alphaThreshold = 0.25",
            "assert m.type == MaterialType.Standard",
            "sceneBuilder.addMeshInstance(sceneBuilder.addNode('q', Transform()), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(), m))",
            "assert (l.intensity * 2).x == 4.0  # bridge reads come back as python vectors",
            "m.roughness = 0.25",
            "assert m.roughness == 0.25 and m.baseColor.w == 1.0",
            "c = Camera('C')",
            "c.position = float3(1, 2, 3)",
            "assert c.position.y == 2.0",
            "sceneBuilder.renderSettings.useEnvLight = False",
            "sceneBuilder.renderSettings.diffuseAlbedoMultiplier = 0.5",
        ].join("\n"),
        "/Falcor/media",
    );
    expectEq(scene.getMaterialCount(), 2, "addMaterial keeps an unused material");
    expectEq(scene.getMaterial(0).name, "Unused", "added material comes first");
    expectEq(scene.renderSettings.useEnvLight, false, "renderSettings applied");
    expectEq([scene.getMaterial(1).header?.alphaMode, scene.getMaterial(1).header?.alphaThreshold].join(), "1,0.25", "alphaMode / alphaThreshold");
    expectEq(scene.getSceneDefines().get("SCENE_DIFFUSE_ALBEDO_MULTIPLIER"), "0.500000", "diffuseAlbedoMultiplier define");
});

gpuTest("Scripting.materialTextureTransform", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(
        device,
        [
            "plain = StandardMaterial('Plain')",
            "moved = StandardMaterial('Moved')",
            "moved.textureTransform.scaling = float3(2, 4, 1)  # in place, as native returns a reference",
            "moved.textureTransform.translation = float3(0.5, 0, 0)",
            "assert moved.textureTransform.scaling.y == 4.0",
            "swapped = StandardMaterial('Swapped')",
            "t = Transform()",
            "t.scaling = float3(0.5, 0.5, 1)",
            "swapped.textureTransform = t",
            "for i, mat in enumerate([plain, moved, swapped]):",
            "    sceneBuilder.addMeshInstance(sceneBuilder.addNode(f'q{i}', Transform()), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(), mat))",
        ].join("\n"),
        "/Falcor/media",
    );
    const blob = await (scene as unknown as { buffers: Record<string, { getBlob(): Promise<Uint8Array> }> }).buffers["vertices"]!.getBlob();
    const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
    const n = blob.byteLength / 48 / 3;
    const uv = (mesh: number, v: number) => [dv.getFloat32((mesh * n + v) * 48 + 32, true), dv.getFloat32((mesh * n + v) * 48 + 36, true)];
    expectEq(Number.isInteger(n) && n >= 4, true, `three equal quads (${n} vertices each)`);
    for (let v = 0; v < n; v++) {
        const [u0, v0] = uv(0, v);
        // The inverse transform: uv' = ((u - 0.5) / 2, v / 4) and uv' = 2 uv.
        expectEq(uv(1, v).join(), [Math.fround((u0! - 0.5) / 2), Math.fround(v0! / 4)].join(), `moved texcoord ${v}`);
        expectEq(uv(2, v).join(), [u0! * 2, v0! * 2].join(), `swapped texcoord ${v}`);
    }
});
