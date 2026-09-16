/**
 * AssetResolver over the dev server: HEAD-probe resolution through the
 * default media search path, the pyscene-directory push/pop, the python
 * binding (AssetResolver.default_resolver.add_search_path) and ImageLoader
 * resolving through search paths.
 */

import { AssetCategory, AssetResolver, SearchPathPriority, initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("AssetResolver.resolvesThroughMediaSearchPath", async () => {
    const r = AssetResolver.getDefaultResolver();
    expectEq(await r.resolvePath("test_scenes/cornell_box.pyscene", AssetCategory.Scene), "/Falcor/media/test_scenes/cornell_box.pyscene", "relative path found under /Falcor/media");
    expectEq(await r.resolvePath("/tests/oracle/assets/checker.png"), "/tests/oracle/assets/checker.png", "absolute existing URL");
    expectEq(await r.resolvePath("/tests/oracle/assets/does-not-exist.png"), "", "absolute missing URL");
    expectEq(await r.resolvePath("nope/missing.pyscene"), "", "unresolvable relative path");
    // Dev-server SPA fallback must not count as an existing file.
    expectEq(await r.resolvePath("/tests/oracle/assets/no-such-dir/thing"), "", "extension-less miss (HTML fallback) is not a hit");
});

gpuTest("AssetResolver.sceneScriptDirectoryIsPushedAndRestored", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const before = [...AssetResolver.getDefaultResolver().getSearchPaths()];
    // Builder-geometry scene (no importScene), so only the push/pop is exercised.
    const source = await (await fetch("/tests/oracle/assets/oracle-mirror.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    expectEq(scene !== null, true, "scene loaded");
    expectEq([...AssetResolver.getDefaultResolver().getSearchPaths()], before, "default resolver restored after the script");
});

gpuTest("AssetResolver.pythonBindingAndImageLoaderSearchPath", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const saved = AssetResolver.getDefaultResolver().clone();
    try {
        // checker.png only exists under /tests/oracle/assets — the script registers that directory.
        const [graph] = await runGraphScript(
            device,
            [
                "from falcor import *",
                "AssetResolver.default_resolver.add_search_path('/tests/oracle/assets', SearchPathPriority.First)",
                "g = RenderGraph('R')",
                "g.addPass(createPass('ImageLoader', {'filename': 'checker.png', 'srgb': False}), 'Img')",
                "g.markOutput('Img.dst')",
                "m.addGraph(g)",
            ].join("\n"),
        );
        expectEq(AssetResolver.getDefaultResolver().getSearchPaths()[0], "/tests/oracle/assets", "python add_search_path(First) took effect");
        await graph!.init();
        graph!.onResize(16, 16);
        graph!.execute(device.renderContext);
        const px = await device.renderContext.readTextureSubresource(graph!.getOutput("Img.dst")!);
        let nonZero = 0;
        for (let i = 0; i < px.length; i++) if (px[i]! !== 0) nonZero++;
        expectEq(nonZero > 0, true, "ImageLoader found checker.png via the search path");

        // settings searchpath:media flows into the resolver (SampleApp analog).
        await runGraphScript(device, "from falcor import *\nm.settings.addOptions({'searchpath': {'media': ['/tests/oracle']}})\ng = RenderGraph('S')\ng.addPass(createPass('ImageLoader', {'filename': 'assets/checker.png'}), 'I')\nm.addGraph(g)\n");
        expectEq(AssetResolver.getDefaultResolver().getSearchPaths().includes("/tests/oracle"), true, "searchpath:media appended to the default resolver");
        expectEq(await AssetResolver.getDefaultResolver().resolvePath("assets/checker.png"), "/tests/oracle/assets/checker.png", "resolves through the settings search path");
        expectEq(SearchPathPriority.Last, 1, "enum sanity");
    } finally {
        AssetResolver.setDefaultResolver(saved);
    }
});
