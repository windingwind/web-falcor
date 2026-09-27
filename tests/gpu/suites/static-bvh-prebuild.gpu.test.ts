/**
 * Scene.create prebuilds an animated scene's static-geometry BVH on the worker pool; it must match the tree
 * the first animate() would build serially from its triangle objects, byte for byte.
 */

import { initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

type Tree = { nodes: Float32Array; tris: Float32Array; order: Uint32Array; buildArea: number };

gpuTest("StaticBvhPrebuild.matchesSerialBuild", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const src = await (await fetch("/Falcor/media/test_scenes/animated_cubes/animated_cubes.pyscene")).text();
    const scene = await runSceneScript(device, src, "/Falcor/media/test_scenes/animated_cubes");
    const sc = scene as unknown as { staticBvh: Tree | null; invalidateAnimation(): void; animate(t: number): boolean };
    const pre = sc.staticBvh;
    expectEq(pre !== null && pre.order.length > 0, true, "the static tree is prebuilt");
    // Drop animate()'s caches so it rebuilds the static tree serially.
    Object.assign(sc, { staticBvh: null, staticWorldMats: null, staticLcInputs: null, dynamicBvh: null, animatedBvh: null, uploadedBvh: null, dynamicVerts: null, dynamicTriVerts: null });
    sc.invalidateAnimation();
    sc.animate(0);
    const post = sc.staticBvh!;
    const words = (a: Float32Array | Uint32Array) => Array.from(new Uint32Array(a.buffer, a.byteOffset, a.length));
    expectEq(words(pre!.nodes).join(), words(post.nodes).join(), "nodes");
    expectEq(words(pre!.tris).join(), words(post.tris).join(), "triangles");
    expectEq(Array.from(pre!.order).join(), Array.from(post.order).join(), "order");
    expectEq(pre!.buildArea, post.buildArea, "build area");
    scene.destroy();
});
