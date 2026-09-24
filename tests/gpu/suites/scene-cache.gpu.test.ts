/**
 * SceneCache phase 1: importing cornell_box with cache enabled stores the
 * scene description in OPFS; the next load skips script + import entirely
 * and must render BYTE-IDENTICALLY through the MinimalPathTracer graph.
 */

import { clearSceneCache, initScripting, runGraphScript, runSceneScript, wasSceneLoadedFromCache } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 128;

gpuTest("SceneCache.cachedSceneRendersIdentically", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    await clearSceneCache();
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();
    const ctx = device.renderContext;

    const render = async (scene: Awaited<ReturnType<typeof runSceneScript>>) => {
        const [graph] = await runGraphScript(device, graphSource);
        scene.camera.setAspectRatio(1.0);
        graph!.onResize(size, size);
        graph!.setScene(scene);
        graph!.execute(ctx);
        return new Uint8Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    };

    const scene1 = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes", { cache: true });
    expectEq(wasSceneLoadedFromCache(), false, "first load imports");
    const img1 = await render(scene1);

    const t0 = performance.now();
    const scene2 = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes", { cache: true });
    const cachedMs = performance.now() - t0;
    expectEq(wasSceneLoadedFromCache(), true, "second load hits the cache");
    const img2 = await render(scene2);

    let diff = 0;
    for (let i = 0; i < img1.length; i++) if (img1[i] !== img2[i]) diff++;
    console.error(`# scene-cache: cached load ${cachedMs.toFixed(1)}ms, ${diff} differing bytes of ${img1.length}`);
    expectEq(diff, 0, "cached scene renders byte-identically");

    // Textured scene (phase 2): OBJ mesh + PNG material textures.
    const texturedSource = await (await fetch("/Falcor/media/test_scenes/tutorial.pyscene")).text();
    const tex1 = await runSceneScript(device, texturedSource, "/Falcor/media/test_scenes", { cache: true });
    expectEq(wasSceneLoadedFromCache(), false, "textured first load imports");
    const texImg1 = await render(tex1);
    const tex2 = await runSceneScript(device, texturedSource, "/Falcor/media/test_scenes", { cache: true });
    expectEq(wasSceneLoadedFromCache(), true, "textured second load hits the cache");

    // Probe: locate any GPU-input divergence between import and cache paths.
    const bufs = (sc: unknown) => (sc as { buffers: Record<string, { getBlob(): Promise<Uint8Array> } | undefined> }).buffers;
    for (const name of ["vertices", "indices", "meshes", "materialData", "lights", "instances", "worldMatrices", "bvhNodes"]) {
        const a = bufs(tex1)[name];
        const b = bufs(tex2)[name];
        if (!a || !b) { console.error(`# probe ${name}: missing ${!a ? "import" : "cache"}`); continue; }
        const [da, db] = [await a.getBlob(), await b.getBlob()];
        let d = 0;
        const detail: string[] = [];
        for (let i = 0; i < Math.max(da.length, db.length); i++) {
            if (da[i] !== db[i]) {
                d++;
                if (name === "vertices" && detail.length < 8) {
                    const fi = Math.floor(i / 4);
                    const fa = new Float32Array(da.buffer, da.byteOffset)[fi];
                    const fb = new Float32Array(db.buffer, db.byteOffset)[fi];
                    detail.push(`f[${fi}] (field ${fi % 12}): ${fa} vs ${fb}`);
                }
            }
        }
        console.error(`# probe ${name}: ${d} diff bytes (${da.length}/${db.length}) ${detail.join(" | ")}`);
    }
    const texArr = (sc: unknown) => (sc as { textureBuckets: import("@web-falcor/falcor").Texture[] }).textureBuckets[0]!;
    for (let layer = 0; layer < Math.min(3, texArr(tex1).gpuTexture.depthOrArrayLayers); layer++) {
        const [ta, tb] = [await ctx.readTextureSubresource(texArr(tex1), 0, layer), await ctx.readTextureSubresource(texArr(tex2), 0, layer)];
        let d = 0;
        for (let i = 0; i < ta.length; i++) if (ta[i] !== tb[i]) d++;
        console.error(`# probe texLayer${layer}: ${d} diff bytes of ${ta.length}`);
    }
    const texImg2 = await render(tex2);
    let texDiff = 0;
    for (let i = 0; i < texImg1.length; i++) if (texImg1[i] !== texImg2[i]) texDiff++;
    console.error(`# scene-cache textured: ${texDiff} differing bytes of ${texImg1.length}`);
    expectEq(texDiff, 0, "cached textured scene renders byte-identically");

    await clearSceneCache();
});

gpuTest("SceneCache.curvesAndEnvMapClasses", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    await clearSceneCache();
    const ctx = device.renderContext;
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();

    const render = async (scene: Awaited<ReturnType<typeof runSceneScript>>) => {
        const [graph] = await runGraphScript(device, graphSource);
        scene.camera.setAspectRatio(1.0);
        graph!.onResize(size, size);
        graph!.setScene(scene);
        graph!.execute(ctx);
        return new Uint8Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    };

    // Phase 3 classes: curve geometry (two_curves) and env map (sphere_array).
    for (const [label, dir, file] of [
        ["curves", "/Falcor/media/test_scenes/curves", "two_curves.pyscene"],
        ["envmap", "/Falcor/media/test_scenes", "sphere_array.pyscene"],
    ] as const) {
        const source = await (await fetch(`${dir}/${file}`)).text();
        const s1 = await runSceneScript(device, source, dir, { cache: true });
        expectEq(wasSceneLoadedFromCache(), false, `${label} first load imports`);
        const img1 = await render(s1);
        const t0 = performance.now();
        const s2 = await runSceneScript(device, source, dir, { cache: true });
        const cachedMs = performance.now() - t0;
        expectEq(wasSceneLoadedFromCache(), true, `${label} second load hits the cache`);
        const img2 = await render(s2);
        let diff = 0;
        let nonzero = 0;
        for (let i = 0; i < img1.length; i++) {
            if (img1[i] !== img2[i]) diff++;
            if (img1[i] !== 0) nonzero++;
        }
        console.error(`# scene-cache ${label}: cached load ${cachedMs.toFixed(1)}ms, ${diff} differing bytes of ${img1.length} (${nonzero} nonzero)`);
        expectEq(nonzero > 1000, true, `${label} renders content (${nonzero} nonzero bytes)`);
        expectEq(diff, 0, `${label} cached scene renders byte-identically`);
    }

    await clearSceneCache();
});

gpuTest("SceneCache.animationSdfVolumeClasses", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    await clearSceneCache();
    const ctx = device.renderContext;
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).text();

    const render = async (scene: Awaited<ReturnType<typeof runSceneScript>>, time?: number) => {
        if (time !== undefined) scene.animate(time);
        const [graph] = await runGraphScript(device, graphSource);
        scene.camera.setAspectRatio(1.0);
        graph!.onResize(size, size);
        graph!.setScene(scene);
        graph!.execute(ctx);
        return new Uint8Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    };

    // v4 classes: FBX node animation (animated_cubes at t=3), procedural NDSDF grid (recipe), .vdb grid volume + env map (smoke).
    for (const [label, dir, file, time, check] of [
        ["animation", "/Falcor/media/test_scenes/animated_cubes", "animated_cubes.pyscene", 3.0, (s: Awaited<ReturnType<typeof runSceneScript>>) => s.isAnimated()],
        ["sdf", "/tests/oracle/assets", "ndsdf-mesh.pyscene", undefined, (s: Awaited<ReturnType<typeof runSceneScript>>) => s.sdfGrids.length === 1],
        ["volume", "/Falcor/media/test_scenes", "smoke.pyscene", undefined, (s: Awaited<ReturnType<typeof runSceneScript>>) => s.gridVolumes.length === 1 && s.gridVolumes[0]!.densityGrid!.voxelCount > 0 && s.useEnvLight],
    ] as const) {
        const source = await (await fetch(`${dir}/${file}`)).text();
        const s1 = await runSceneScript(device, source, dir, { cache: true });
        expectEq(wasSceneLoadedFromCache(), false, `${label} first load imports`);
        const img1 = await render(s1, time);
        const t0 = performance.now();
        const s2 = await runSceneScript(device, source, dir, { cache: true });
        const cachedMs = performance.now() - t0;
        expectEq(wasSceneLoadedFromCache(), true, `${label} second load hits the cache`);
        expectEq(check(s2), true, `${label} class restored from the cache`);
        const img2 = await render(s2, time);
        let diff = 0;
        let nonzero = 0;
        for (let i = 0; i < img1.length; i++) {
            if (img1[i] !== img2[i]) diff++;
            if (img1[i] !== 0) nonzero++;
        }
        console.error(`# scene-cache ${label}: cached load ${cachedMs.toFixed(1)}ms, ${diff} differing bytes of ${img1.length} (${nonzero} nonzero)`);
        expectEq(nonzero > 1000, true, `${label} renders content (${nonzero} nonzero bytes)`);
        expectEq(diff, 0, `${label} cached scene renders byte-identically`);
    }

    // Animated scenes keep animating from the cache (a later time renders differently).
    const source = await (await fetch("/Falcor/media/test_scenes/animated_cubes/animated_cubes.pyscene")).text();
    const cached = await runSceneScript(device, source, "/Falcor/media/test_scenes/animated_cubes", { cache: true });
    expectEq(wasSceneLoadedFromCache(), true, "animated scene served from cache");
    const a = await render(cached, 3.0);
    const b = await render(cached, 8.0);
    let moved = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) moved++;
    expectEq(moved > 100, true, `cached animation advances between t=3 and t=8 (${moved} bytes changed)`);

    await clearSceneCache();
});
