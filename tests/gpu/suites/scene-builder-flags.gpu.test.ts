/**
 * SceneBuilder::Flags on the web: AssumeLinearSpaceTextures changes the
 * texture colour space, DontUseDisplacement demotes displaced meshes to plain
 * triangles (scene defines), UseCache/RebuildCache drive the OPFS cache, and
 * python scripts can combine SceneBuilderFlags like the upstream image tests.
 */

import { GeometryType, RenderGraph, SceneBuilderFlags, clearSceneCache, createPass, initScripting, runGraphScript, runSceneScript, wasSceneLoadedFromCache } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("SceneBuilderFlags.honouredFlags", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    await clearSceneCache();

    // AssumeLinearSpaceTextures: colour textures stop being sRGB (tutorial.pyscene has PNG material textures).
    const textured = await (await fetch("/Falcor/media/test_scenes/tutorial.pyscene")).text();
    const srgbOf = (scene: unknown) => {
        const tm = (scene as { lcTextureManager: { count: number; getSource(id: number): { srgb: boolean } | undefined } }).lcTextureManager;
        const flags: boolean[] = [];
        for (let i = 0; i < tm.count; i++) flags.push(tm.getSource(i)!.srgb);
        return flags;
    };
    const plain = await runSceneScript(device, textured, "/Falcor/media/test_scenes");
    const linear = await runSceneScript(device, textured, "/Falcor/media/test_scenes", { flags: SceneBuilderFlags.AssumeLinearSpaceTextures });
    expectEq(srgbOf(plain).some((s) => s), true, "default load has sRGB colour textures");
    expectEq(srgbOf(linear).every((s) => !s), true, "AssumeLinearSpaceTextures loads every texture linear");

    // DontUseDisplacement: displaced meshes become plain triangle meshes.
    const displaced = await (await fetch("/Falcor/media/test_scenes/cornell_box_displaced.pyscene")).text();
    const withDisp = await runSceneScript(device, displaced, "/Falcor/media/test_scenes");
    const noDisp = await runSceneScript(device, displaced, "/Falcor/media/test_scenes", { flags: SceneBuilderFlags.DontUseDisplacement });
    // SCENE_GEOMETRY_TYPES is a mask of 1 << GeometryType (TriangleMesh = 1, DisplacedTriangleMesh = 2).
    const types = (scene: Awaited<ReturnType<typeof runSceneScript>>) => Number(scene.getSceneDefines().get("SCENE_GEOMETRY_TYPES"));
    const kTri = 1 << GeometryType.TriangleMesh;
    const kDisplaced = 1 << GeometryType.DisplacedTriangleMesh;
    expectEq((types(withDisp) & kDisplaced) !== 0, true, `displaced geometry type present by default (types=${types(withDisp)})`);
    expectEq(types(noDisp), kTri, `DontUseDisplacement leaves triangle meshes only (types=${types(noDisp)})`);

    // UseCache / RebuildCache.
    const cornell = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    await runSceneScript(device, cornell, "/Falcor/media/test_scenes", { flags: SceneBuilderFlags.UseCache });
    expectEq(wasSceneLoadedFromCache(), false, "UseCache: first load imports and stores");
    await runSceneScript(device, cornell, "/Falcor/media/test_scenes", { flags: SceneBuilderFlags.UseCache });
    expectEq(wasSceneLoadedFromCache(), true, "UseCache: second load hits the cache");
    await runSceneScript(device, cornell, "/Falcor/media/test_scenes", { flags: SceneBuilderFlags.UseCache | SceneBuilderFlags.RebuildCache });
    expectEq(wasSceneLoadedFromCache(), false, "RebuildCache bypasses the stored entry");
    await runSceneScript(device, cornell, "/Falcor/media/test_scenes", { flags: SceneBuilderFlags.UseCache | SceneBuilderFlags.DontUseDisplacement });
    expectEq(wasSceneLoadedFromCache(), false, "different build flags key a different cache entry");
    await clearSceneCache();

    // Python surface (upstream test_*.py idiom).
    const [graph] = await runGraphScript(
        device,
        "from falcor import *\nflags = SceneBuilderFlags.NonIndexedVertices | SceneBuilderFlags.Force32BitIndices\nassert flags == 0xC0, flags\nassert SceneBuilderFlags.UseCache == 0x10000000\ng = RenderGraph('F')\ng.addPass(createPass('ImageLoader', {'filename': 'test_images/smoke_puff.png'}), 'I')\ng.markOutput('I.dst')\nm.addGraph(g)\n",
    );
    expectEq(graph !== undefined, true, "SceneBuilderFlags usable from python");
});

gpuTest("SceneBuilderFlags.nonIndexedVerticesRenderTheSameScene", async ({ device }) => {
    // NonIndexedVertices gives every triangle its own vertices; the triangles,
    // their order and their attributes are unchanged, so the G-buffer must be
    // byte-identical to the indexed build.
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const size = 64;
    const stats: { vertices: number; triangles: number }[] = [];
    const render = async (flags: number) => {
        const scene = await runSceneScript(device, source, "/Falcor/media/test_scenes", { flags });
        scene.camera.setAspectRatio(1);
        stats.push(scene.stats);
        const graph = new RenderGraph(device, "NonIndexed");
        graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBuffer");
        graph.markOutput("GBuffer.posW");
        graph.markOutput("GBuffer.normW");
        graph.markOutput("GBuffer.texC");
        graph.onResize(size, size);
        graph.setScene(scene);
        await graph.init();
        graph.execute(device.renderContext);
        const out: Uint8Array[] = [];
        for (const name of ["GBuffer.posW", "GBuffer.normW", "GBuffer.texC"]) out.push(await device.renderContext.readTextureSubresource(graph.getOutput(name)!));
        return out;
    };
    const indexed = await render(SceneBuilderFlags.None);
    const nonIndexed = await render(SceneBuilderFlags.NonIndexedVertices);
    let diffs = 0;
    indexed.forEach((a, k) => a.forEach((byte, i) => (diffs += byte === nonIndexed[k]![i] ? 0 : 1)));
    console.error(`# NonIndexedVertices: ${diffs} differing bytes across posW/normW/texC`);
    expectEq(diffs, 0, "de-indexed geometry renders byte-identically");
    // The flag did take effect: three vertices per triangle, none shared.
    console.error(`# vertices: indexed ${stats[0]!.vertices}, non-indexed ${stats[1]!.vertices} for ${stats[1]!.triangles} triangles`);
    expectEq(stats[1]!.triangles, stats[0]!.triangles, "same triangles");
    expectEq(stats[1]!.vertices, 3 * stats[1]!.triangles, "every triangle owns its vertices");
    expectEq(stats[0]!.vertices < stats[1]!.vertices, true, "the indexed build shares vertices");
});
