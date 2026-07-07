/**
 * GridVolume scene loading: the UNMODIFIED upstream smoke.pyscene (GridVolume
 * + loadGrid from the real OpenVDB smoke.vdb) loads in the browser through
 * the Pyodide bridge; grid stats and point samples match the validated
 * ground truth (which native Mogwai confirmed via its own NanoVDB loader).
 */

import { initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("SmokeScene.gridVolumeLoads", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/smoke.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");

    expectEq(scene.gridVolumes.length, 1, "gridVolumes count");
    const vol = scene.gridVolumes[0]!;
    expectEq(vol.name, "smoke", "volume name");
    expectEq(vol.densityScale, 0.5, "densityScale");
    expectEq(Math.abs(vol.albedo.x - 0.5) < 1e-6, true, "albedo");

    const grid = vol.densityGrid!;
    const ref = await (await fetch("/tests/oracle/assets/smoke-vdb-samples.json")).json();
    expectEq(grid.voxelCount, ref.activeVoxels, "voxelCount");
    expectEq(grid.minIndex.join(","), "0,0,0", "minIndex");
    expectEq(grid.maxIndex.join(","), "112,224,112", "maxIndex");
    console.error(`# smoke grid: voxels=${grid.voxelCount} min=${grid.minValue.toExponential(3)} max=${grid.maxValue} bounds=${JSON.stringify(grid.worldBounds)}`);
    expectEq(Math.abs(grid.maxValue - 5.71484375) < 1e-6, true, "maxValue");
    // Env light also present in smoke.pyscene.
    expectEq(scene.useEnvLight, true, "env light");
});
