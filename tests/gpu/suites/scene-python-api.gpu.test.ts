/**
 * Native python properties on the live scene objects (docs/usage/scripting.md), driven through
 * the Mogwai console: lights (intensity/position/active), materials (roughness/baseColor/
 * alphaMode/volumeAnisotropy), camera (position/nearPlane/animated), scene.bounds and
 * loopAnimations. Edits must reach the GPU-side data (light buffer, material blob, camera).
 */

import { Profiler, SceneLight, initScripting, runConsoleCommand, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Scripting.liveScenePythonProperties", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const base = "/tests/oracle/assets";
    const scene = await runSceneScript(device, await (await fetch(`${base}/oracle-pt.pyscene`)).text(), base);
    const profiler = new Profiler(device);
    const run = (src: string) => runConsoleCommand(device, src, { scene, graph: null, profiler });

    // Lights.
    expectEq(run("len(m.scene.lights)"), "1", "scene.lights");
    run("l = m.scene.getLight('oracleLight')\nl.intensity = float3(1, 2, 3)\nl.position = float3(0, 1, 2)");
    const light = scene.getLight(0) as SceneLight;
    expectEq([light.intensity.x, light.intensity.y, light.intensity.z, light.posW!.z].join(), "1,2,3,2", "light edits");
    expectEq(run("m.scene.lights[0].intensity.y"), "2", "light.intensity reads back");
    run("m.scene.lights[0].active = False");
    expectEq(scene.activeLights.length, 0, "inactive light leaves the light buffer");
    run("m.scene.lights[0].active = True");
    expectEq(scene.activeLights.length, 1, "reactivated");

    // Materials.
    run("mat = m.scene.materials[0]\nmat.roughness = 0.25\nmat.baseColor = float4(0.5, 0.25, 0.125, 1)\nmat.volumeAnisotropy = 0.5\nmat.alphaThreshold = 0.3");
    const desc = scene.getMaterial(0);
    expectEq([desc.basic.specular!.y, desc.basic.baseColor!.y, desc.basic.volumeAnisotropy, desc.header?.alphaThreshold].join(), "0.25,0.25,0.5,0.3", "material edits");
    expectEq(run("m.scene.materials[0].roughness"), "0.25", "roughness reads back");

    // Camera.
    run("m.scene.camera.position = float3(1, 2, 3)\nm.scene.camera.nearPlane = 0.5");
    expectEq([scene.camera.getPosition().y, scene.camera.getNearPlane()].join(), "2,0.5", "camera edits");
    expectEq(run("m.scene.camera.position.z"), "3", "camera.position reads back");

    // Bounds and animation settings.
    expectEq(run("m.scene.bounds.valid"), "True", "scene.bounds");
    expectEq(Number(run("m.scene.bounds.radius")) > 0, true, "bounds radius");
    run("m.scene.loopAnimations = False");
    expectEq(scene.isLooped(), false, "loopAnimations");

    // Viewpoints: the default one exists; a saved pose comes back on select.
    run("m.scene.addViewpoint()\nm.scene.camera.position = float3(9, 9, 9)\nm.scene.selectViewpoint(1)");
    expectEq(scene.camera.getPosition().y, 2, "selectViewpoint restores the saved pose");
    expectEq(scene.getViewpointCount(), 2, "viewpoint count");
    const script = scene.getViewpointsScript(30).trim().split("\n");
    expectEq(script.length === 3 && script[2]!.startsWith("30, Transform(position = float3("), true, `viewpoints script ${script.join(" | ")}`);
    run("m.scene.removeViewpoint()");
    expectEq(scene.getViewpointCount(), 1, "removeViewpoint");

    // m.profiler.event(name) is a context manager (native ProfilerEvent).
    run("with m.profiler.event('scriptEvent'):\n    pass");
    expectEq(profiler.findEvent("/scriptEvent") !== undefined, true, "profiler event recorded");

    // updateCallback(scene, time) runs at each frame's scene update.
    run("def _update(scene, time):\n    scene.camera.position = float3(time, 0, 0)\nm.scene.updateCallback = _update");
    scene.runUpdateCallback(2.5);
    expectEq(scene.camera.getPosition().x, 2.5, "updateCallback ran with the frame time");
});

gpuTest("Scripting.liveSceneQueries", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(
        device,
        [
            "a = StandardMaterial('A')",
            "b = StandardMaterial('B')",
            "b.baseColor = float4(1, 0, 0, 1)",
            "quad = sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(), a)",
            "cube = sceneBuilder.addTriangleMesh(TriangleMesh.createCube(), b)",
            "for i in range(2): sceneBuilder.addMeshInstance(sceneBuilder.addNode(f'q{i}', Transform()), quad)",
            "sceneBuilder.addMeshInstance(sceneBuilder.addNode('c', Transform()), cube)",
        ].join("\n"),
        "/Falcor/media",
    );
    const run = (src: string) => runConsoleCommand(device, src, { scene, graph: null, profiler: new Profiler(device) });
    expectEq(run("list(m.scene.getGeometryIDsForMaterial(m.scene.get_material('B')))"), "[0]", "geometry IDs of B (native mesh groups: the non-instanced cube comes first)");
    expectEq(run("list(m.scene.getGeometryIDsForMaterial(m.scene.materials[0]))"), "[1]", "geometry IDs of A (the instanced quad)");
    expectEq(run("(m.scene.get_mesh(0).vertex_count, m.scene.get_mesh(0).triangle_count)"), "(24, 12)", "get_mesh counts (cube)");
    expectEq(Number(run("m.scene.memory_usage")) > 0, true, "memory_usage");
    run("m.scene.setCameraBounds(float3(-1, -1, -1), float3(1, 1, 1))\nm.scene.cameraSpeed = 2.5");
    expectEq([scene.cameraBounds!.maxPoint.y, scene.cameraSpeed].join(), "1,2.5", "camera bounds and speed");
    expectEq(run("m.scene.setEnvMap('/tests/oracle/assets/gradient.hdr')"), "True", "setEnvMap(path)");
    await scene.pendingEnvMap;
    expectEq(scene.getEnvMap() !== null, true, "env map loaded");
});

gpuTest("Scene.geometryUVTilesMatchNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const base = "/tests/oracle/assets";
    const scene = await runSceneScript(device, await (await fetch(`${base}/uv-tiles.pyscene`)).text(), base);
    const native = (await (await fetch("/tests/oracle/out-native/uv-tiles.json")).json()) as number[][][];
    const web = [0, 1].map((id) => scene.getGeometryUVTiles(id).map((t) => [t.minPoint.x, t.minPoint.y, t.maxPoint.x, t.maxPoint.y]));
    expectEq(JSON.stringify(web), JSON.stringify(native), "UV tiles per mesh, in native order");
    expectEq(runConsoleCommand(device, "len(m.scene.getGeometryUVTiles(0))", { scene, graph: null }), "2", "python binding");
});
