/**
 * Blosc-compressed OpenVDB volumes end to end: the browser parses openvdb.org's
 * sample `.vdb` (blosc + LZ4 + byte shuffle, half-float leaves, UniformScaleMap),
 * builds the NanoVDB buffer and traverses it on the GPU.
 *
 * Native Falcor's own OpenVDB reader is broken on this host (DESIGN §6.3), so
 * the ground truth is analytic: a level-set sphere stores the signed distance to
 * its surface, so every GPU lookup must satisfy |p| - value == radius.
 *
 * Fetch the volume with: npm run download:assets -- openvdb
 */

import { Buffer, ComputePass, MemoryType, ResourceBindFlags, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose, SkipError } from "../harness/registry.js";

gpuTest("OpenVDBBlosc.levelSetSphereTraversesOnGPU", async ({ device }) => {
    const probe = await fetch("/Falcor/media/openvdb/sphere.vdb", { method: "HEAD" });
    if (!probe.ok) throw new SkipError("Falcor/media/openvdb/sphere.vdb missing (npm run download:assets -- openvdb)");

    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/tests/oracle/assets/openvdb-sphere.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/tests/oracle/assets");

    expectEq(scene.gridVolumes.length, 1, "gridVolumes count");
    const grid = scene.gridVolumes[0]!.densityGrid!;
    // The file's own metadata says 270638 active voxels; the parser must agree.
    expectEq(grid.voxelCount, 270638, "active voxel count");
    console.error(`# sphere grid: voxels=${grid.voxelCount} min=${grid.minValue.toFixed(4)} max=${grid.maxValue.toFixed(4)} bounds=${JSON.stringify(grid.worldBounds)}`);
    // Narrow band: values span roughly ±background (0.15) and never explode.
    expectEq(grid.minValue < 0 && grid.minValue > -0.2, true, `minValue ${grid.minValue}`);
    expectEq(grid.maxValue > 0 && grid.maxValue < 0.2, true, `maxValue ${grid.maxValue}`);

    // GPU probe: PNanoVDB traversal at index coordinates on the narrow band.
    // The band is a thin shell around radius 3, so sample directions on a
    // spherical Fibonacci lattice at a few radii rather than the whole bbox.
    const coordList: [number, number, number][] = [];
    const voxelSize = 0.05; // the file's UniformScaleMap
    const shellIndex = 3 / voxelSize; // radius 3 in index space
    const points = 1024;
    for (let i = 0; i < points; i++) {
        const y = 1 - (2 * i + 1) / points;
        const r = Math.sqrt(Math.max(0, 1 - y * y));
        const phi = i * Math.PI * (3 - Math.sqrt(5)); // golden angle
        for (const dr of [-1, 0, 1]) {
            const radius = shellIndex + dr;
            coordList.push([Math.round(Math.cos(phi) * r * radius), Math.round(y * radius), Math.round(Math.sin(phi) * r * radius)]);
        }
    }
    const n = coordList.length;
    const coords = new Int32Array(n * 4);
    coordList.forEach((c, i) => {
        coords[i * 4] = c[0];
        coords[i * 4 + 1] = c[1];
        coords[i * 4 + 2] = c[2];
    });

    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const coordBuf = new Buffer(device, { size: n * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "vdb::coords" });
    coordBuf.setBlob(new Uint8Array(coords.buffer));
    const resultBuf = new Buffer(device, { size: n * 4, structSize: 4, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "vdb::results" });

    const pass = ComputePass.create(device, { path: "WebFalcor/GridLookupTest.cs.slang", defines: scene.getSceneDefines() });
    const ctx = device.renderContext;
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    (root["CB"] as Record<string, unknown>)["gCount"] = n;
    root["gCoords"] = coordBuf;
    root["gResults"] = resultBuf;
    pass.execute(ctx, n, 1);
    const results = new Float32Array((await ctx.readBuffer(resultBuf)).buffer);

    // Every voxel inside the narrow band must reproduce the analytic SDF; the
    // rest read back the background (outside the band) and are skipped.
    let inBand = 0;
    let worst = 0;
    for (let i = 0; i < n; i++) {
        const value = results[i]!;
        if (Math.abs(Math.abs(value) - 0.1500244140625) < 1e-6) continue; // ±background: outside the band
        inBand++;
        const [x, y, z] = coordList[i]!;
        const radius = Math.hypot(x * voxelSize, y * voxelSize, z * voxelSize) - value;
        worst = Math.max(worst, Math.abs(radius - 3));
    }
    console.error(`# GPU SDF probe: ${inBand}/${n} inside the narrow band, worst radius error ${worst.toExponential(2)}`);
    expectEq(inBand > 250, true, `enough in-band samples (${inBand})`); // rounding to integer voxels puts most probes just outside the shell
    expectClose(worst, 0, 1e-3, "GPU lookups reproduce the analytic sphere radius");
});
