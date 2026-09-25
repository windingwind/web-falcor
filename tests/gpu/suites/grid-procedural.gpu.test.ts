/**
 * Grid.createSphere / createBox vs native (NanoVDB fog volumes, tests/oracle/assets/
 * oracle-grid-stats.pyscene, whose printed stats are stored in out-native/grid-stats.json):
 * voxel counts, index bounds, value range and probed voxel values, including the thin-band sphere
 * whose interior native's signed flood fill leaves partly empty. The volume's density grid binds.
 */

import { Buffer, ComputePass, MemoryType, ResourceBindFlags, initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Grid.proceduralMatchesNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const native = (await (await fetch("/tests/oracle/out-native/grid-stats.json")).json()) as Record<string, unknown>;
    const source = (await (await fetch("/tests/oracle/assets/oracle-grid-stats.pyscene")).text()).replace("print('GRIDSTATS ' + json.dumps(out))", "_web_stats = json.dumps(out)\nimport js\njs.globalThis.__gridStats = _web_stats");
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    const web = JSON.parse((globalThis as unknown as { __gridStats: string }).__gridStats) as Record<string, unknown>;
    for (const name of Object.keys(native)) {
        const [a, b] = [JSON.stringify(web[name]), JSON.stringify(native[name])];
        if (a !== b) console.error(`# grid ${name}: web ${a} vs native ${b}`);
        expectEq(a, b, `${name} stats`);
    }
    expectEq(scene.gridVolumes[0]?.getGrid("density")?.voxelCount, 4169, "the procedural density grid is bound");
});

// GPU PNanoVDB lookups on the web-built buffer at the native probes, negative octants included
// (root keys once sign-extended negative origins, so every lookup below zero missed).
gpuTest("Grid.gpuLookupsMatchNativeProbes", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const native = (await (await fetch("/tests/oracle/out-native/grid-stats.json")).json()) as Record<string, { probes: [number[], number][] }>;
    const scene = await runSceneScript(device, "v = GridVolume('s')\nv.densityGrid = Grid.createSphere(1.0, 0.01)\nsceneBuilder.addGridVolume(v)", "");
    const probes = native["sphere100"]!.probes;
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const coordBuf = new Buffer(device, { size: probes.length * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal });
    coordBuf.setBlob(new Uint8Array(new Int32Array(probes.flatMap(([c]) => [c[0]!, c[1]!, c[2]!, 0])).buffer));
    const resultBuf = new Buffer(device, { size: probes.length * 4, structSize: 4, bindFlags: storage, memoryType: MemoryType.DeviceLocal });
    const pass = ComputePass.create(device, { path: "WebFalcor/GridLookupTest.cs.slang", defines: scene.getSceneDefines() });
    const root = pass.getRootVar();
    scene.bindShaderData(root);
    (root["CB"] as Record<string, unknown>)["gCount"] = probes.length;
    root["gCoords"] = coordBuf;
    root["gResults"] = resultBuf;
    pass.execute(device.renderContext, probes.length, 1);
    const r = new Float32Array((await device.renderContext.readBuffer(resultBuf)).buffer);
    const bad = probes.filter(([c, v], i) => r[i] !== Math.fround(v) && console.error(`# probe ${c}: web ${r[i]} native ${v}`) === undefined);
    expectEq(probes.some(([c]) => c.some((x) => x < 0)), true, "probes include negative indices");
    expectEq(bad.length, 0, "GPU lookups equal native getValue at every probe");
});

