/**
 * GridVolume scene loading: the UNMODIFIED upstream smoke.pyscene (GridVolume
 * + loadGrid from the real OpenVDB smoke.vdb) loads in the browser through
 * the Pyodide bridge; grid stats and point samples match the validated
 * ground truth (which native Mogwai confirmed via its own NanoVDB loader).
 */

import { Buffer, ComputePass, MemoryType, ResourceBindFlags, initScripting, runSceneScript } from "@web-falcor/falcor";
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

    // GPU probe: PNanoVDB traversal of gScene.grid0 at the 500 ground-truth
    // coordinates (validates the WGSL read-accessor chain over the buffer).
    const pass = ComputePass.create(device, { path: "WebFalcor/GridLookupTest.cs.slang", defines: scene.getSceneDefines() });
    const n = ref.samples.length;
    const coords = new Int32Array(n * 4);
    ref.samples.forEach((s: number[], i: number) => {
        coords[i * 4] = s[0]!;
        coords[i * 4 + 1] = s[1]!;
        coords[i * 4 + 2] = s[2]!;
    });
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const coordBuf = new Buffer(device, { size: n * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "probe::coords" });
    coordBuf.setBlob(new Uint8Array(coords.buffer));
    const resultBuf = new Buffer(device, { size: n * 4, structSize: 4, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "probe::results" });
    const ctx = device.renderContext;
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    (root["CB"] as Record<string, unknown>)["gCount"] = n;
    root["gCoords"] = coordBuf;
    root["gResults"] = resultBuf;
    pass.execute(ctx, n, 1);
    const results = new Float32Array((await ctx.readBuffer(resultBuf)).buffer);
    let bad = 0;
    for (let i = 0; i < n; i++) {
        if (Math.abs(results[i]! - ref.samples[i][3]) > 1e-7) bad++;
    }
    console.error(`# grid GPU lookup: ${bad}/${n} mismatches`);
    expectEq(bad, 0, "GPU PNanoVDB lookups");

    // Volume state probe: what the GPU actually sees in gScene.
    const probe = ComputePass.create(device, { path: "WebFalcor/GridVolumeProbe.cs.slang", defines: scene.getSceneDefines() });
    const probeOut = new Buffer(device, { size: 128, structSize: 4, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "probe::vol" });
    const proot = probe.getRootVar();
    scene.bindShaderData(proot);
    proot["gOut"] = probeOut;
    probe.execute(ctx, 1, 1);
    const vals = new Float32Array((await ctx.readBuffer(probeOut)).buffer);
    console.error(`# volume state: count=${vals[0]} densityGrid=${vals[1]} scale=${vals[2]} bmin=${vals[3]},${vals[4]},${vals[5]} bmax=${vals[6]},${vals[7]},${vals[8]} albedo=${vals[9]} hasDensity=${vals[10]} lookup(56,112,56)=${vals[11]} nearFar=${vals[12]},${vals[13]} centerTr=${vals[14]} ipos=${vals[15]},${vals[16]},${vals[17]} inv00=${vals[18]} inv03=${vals[19]} inv30=${vals[20]} inv13=${vals[21]} midDensity=${vals[22]}`);
});
