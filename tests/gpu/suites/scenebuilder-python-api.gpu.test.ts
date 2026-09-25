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

gpuTest("Scripting.pysceneObjectProperties", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(
        device,
        [
            "c = Camera('Cam')",
            "c.name = 'Renamed'",
            "c.nearPlane = 0.5",
            "c.farPlane = 50",
            "c.frameHeight = 36",
            "c.animated = False",
            "sceneBuilder.addCamera(c)",
            "l = PointLight('L')",
            "l.active = False",
            "l.animated = False",
            "sceneBuilder.addLight(l)",
            "mat = StandardMaterial('M')",
            "mat.name = 'M2'",
            "mat.emissiveColor = float3(1, 0, 0)",
            "assert mat.emissive",
            "mat.loadTexture(MaterialTextureSlot.BaseColor, 'missing.png')",
            "mat.clearTexture(MaterialTextureSlot.BaseColor)",
            "sceneBuilder.addMeshInstance(sceneBuilder.addNode('q', Transform()), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(), mat))",
            "vol = GridVolume('V')",
            "assert vol.gridFrameCount == 0",
            "vol.gridFrame = 0",
            "b = AABB(min_point=float3(-1, -2, -3), max_point=float3(1, 2, 3))",
            "assert b.max_point.y == 2 and b.min_point.z == -3",
        ].join("\n"),
        "/Falcor/media",
    );
    const cam = scene.cameras.find((x) => x.name === "Renamed");
    expectEq(cam !== undefined, true, "camera renamed");
    expectEq([cam!.getNearPlane(), cam!.getFarPlane(), cam!.getFrameHeight(), cam!.animated].join(), "0.5,50,36,false", "camera properties");
    const light = scene.getLight("L") as { active: boolean; animated: boolean };
    expectEq([light.active, light.animated, scene.activeLights.length].join(), "false,false,0", "light active/animated");
    expectEq(scene.getMaterial(0).name, "M2", "material renamed");
    expectEq(scene.getMaterial(0).basic.texBaseColor, undefined, "cleared texture never loads");
});

gpuTest("Scripting.triangleMeshFrontFaceCW", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(
        device,
        [
            "mat = StandardMaterial('M')",
            "plain = TriangleMesh.createQuad()",
            "mirrored = TriangleMesh.createQuad(float2(-1, 1))",
            "assert mirrored.frontFaceCW and not plain.frontFaceCW",
            "custom = TriangleMesh()",
            "custom.name = 'Custom'",
            "for p in [float3(0, 0, 0), float3(1, 0, 0), float3(0, 1, 0)]: custom.addVertex(p, float3(0, 0, 1), float2(0, 0))",
            "custom.addTriangle(0, 1, 2)",
            "custom.frontFaceCW = True",
            "for i, mesh in enumerate([plain, mirrored, custom]):",
            "    sceneBuilder.addMeshInstance(sceneBuilder.addNode(f'n{i}', Transform()), sceneBuilder.addTriangleMesh(mesh, mat))",
        ].join("\n"),
        "/Falcor/media",
    );
    const bufs = (scene as unknown as { buffers: Record<string, { getBlob(): Promise<Uint8Array> }> }).buffers;
    const ib = await bufs["indices"]!.getBlob();
    const vb = await bufs["vertices"]!.getBlob();
    const idx = new Uint32Array(ib.buffer, ib.byteOffset, 15);
    const pos = new DataView(vb.buffer, vb.byteOffset, vb.byteLength);
    expectEq(vb.byteLength / 48, 11, "4 + 4 + 3 vertices");
    // Sign of each triangle's geometric normal along the authored normal (+Y quads, +Z custom).
    const facing = (first: number, base: number, count: number, axis: number) =>
        Array.from({ length: count }, (_, t) => {
            const p = [0, 1, 2].map((k) => [0, 1, 2].map((c) => pos.getFloat32((base + idx[first + 3 * t + k]!) * 48 + 4 * c, true)));
            const [e1, e2] = [p[1]!.map((v, c) => v - p[0]![c]!), p[2]!.map((v, c) => v - p[0]![c]!)];
            const n = [e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!];
            return Math.sign(n[axis]!);
        }).join();
    // SceneBuilder::unifyTriangleWinding makes every mesh counter-clockwise around its front face.
    expectEq(facing(0, 0, 2, 1), "1,1", "counter-clockwise quad");
    expectEq(facing(6, 4, 2, 1), "1,1", "mirrored quad flipped back to counter-clockwise");
    expectEq(facing(12, 8, 1, 2), "-1", "frontFaceCW mesh: its front is the clockwise side");
});

gpuTest("Scripting.pysceneVectorTypes", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    await runSceneScript(
        device,
        [
            "a = float3(1, 2, 3)",
            "assert repr(a + float3([1, 1, 1])) == 'float3(2.000000, 3.000000, 4.000000)'  # native repr",
            "assert str(float2(0.5)) == '[0.500000, 0.500000]'",
            "b = float4(x=1, y=2, z=3, w=4)",
            "b -= float4(1)",
            "b *= 2",
            "assert (b.x, b.w) == (0.0, 6.0)",
            "assert (float2(4, 6) / 2).y == 3.0 and (-int2(1, 2)).y == -2",
            "assert repr(uint3(7) / 2) == 'uint3(3, 3, 3)' and repr(bool2(1, 0)) == 'bool2(1, 0)'",
            "assert list(a) == [1.0, 2.0, 3.0] and a[2] == 3.0",
            "sceneBuilder.addMeshInstance(sceneBuilder.addNode('q', Transform(translation=a * 0.5)), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(float2(2) - 1), StandardMaterial('M')))",
        ].join("\n"),
        "/Falcor/media",
    );
});
