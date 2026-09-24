/**
 * Grid.createSphere / createBox vs native (NanoVDB fog volumes, tests/oracle/assets/
 * oracle-grid-stats.pyscene, whose printed stats are stored in out-native/grid-stats.json):
 * voxel counts, index bounds, value range and probed voxel values, including the thin-band sphere
 * whose interior native's signed flood fill leaves partly empty. The volume's density grid binds.
 */

import { initScripting, runSceneScript } from "@web-falcor/falcor";
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
