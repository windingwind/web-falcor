/**
 * Grid's BrickedGrid (NanoVDBConverterBC4, Scene/Volume/GridConverter.ts): brick lookups (Grid::lookupIndexTex) on
 * the GPU return the uploaded bricks as a CPU decode reads them (hardware BC4 interpolation weights are implementation-
 * defined, e.g. 144/257 for 4/7 here, so within 3% of each block's endpoint span), stay within BC4 error of the NanoVDB value,
 * and every mip's majorant bounds the value. two_volumes has two grids stacked in the shared brick textures.
 */

import { Buffer, ComputePass, MemoryType, ResourceBindFlags, decodeBC4Volume, float16ToFloat32, initScripting, runSceneScript, type Grid, type Device } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

/** The brick value at `rel` (index relative to minIndex) from the converter's data, and the decode tolerance there. */
function cpuBrickValue(grid: Grid, atlas: Float32Array, rel: number[]): [number, number] {
    const b = grid.brickedGrid;
    const [lx, ly] = b.leafDim;
    const i = ((rel[2]! >> 3) * ly + (rel[1]! >> 3)) * lx + (rel[0]! >> 3);
    const range = b.rangeMips[0]![i]!;
    const [maj, min] = [float16ToFloat32(range & 0xffff), float16ToFloat32(range >>> 16)];
    const ptr = b.indirection[i]!;
    const [ax, ay] = b.atlasSize;
    const [px, py, pz] = [((ptr & 0xff) << 3) + (rel[0]! & 7), (((ptr >> 8) & 0xff) << 3) + (rel[1]! & 7), (((ptr >> 16) & 0xff) << 3) + (rel[2]! & 7)];
    // Hardware BC4 interpolation weights deviate from k/7 by up to about 2.2% of the block's endpoint span.
    const blk = ((pz * (ay / 4) + (py >> 2)) * (ax / 4) + (px >> 2)) * 8;
    const endpoints = Math.abs(b.atlas[blk]! - b.atlas[blk + 1]!);
    return [Math.fround(Math.fround(atlas[(pz * ay + py) * ax + px]! * Math.fround(maj - min)) + min), ((0.03 * endpoints + 1) * (maj - min)) / 255];
}

async function checkGrids(device: Device, scenePath: string, dir: string, expectedGrids: number): Promise<void> {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(device, await (await fetch(scenePath)).text(), dir);
    const grids = [...new Set(scene.gridVolumes.flatMap((v) => [v.densityGrid, v.emissionGrid]).filter((g): g is Grid => !!g))];
    expectEq(grids.length, expectedGrids, "grids");
    const pass = ComputePass.create(device, { path: "WebFalcor/GridBrickTest.cs.slang", defines: scene.getSceneDefines() });
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    let seed = 1;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (const [gi, grid] of grids.entries()) {
        const [lo, hi] = [grid.minIndex, grid.maxIndex];
        const b = grid.brickedGrid;
        const atlas = decodeBC4Volume(b.atlas, ...b.atlasSize);
        const n = 20000;
        const coords = new Int32Array(n * 4);
        for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) coords[i * 4 + k] = lo[k]! + Math.floor(rnd() * (hi[k]! - lo[k]!));
        const coordBuf = new Buffer(device, { size: n * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal });
        coordBuf.setBlob(new Uint8Array(coords.buffer));
        const resultBuf = new Buffer(device, { size: n * 16, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal });
        const root = pass.getRootVar();
        scene.bindShaderData(root);
        const cb = root["CB"] as Record<string, unknown>;
        [cb["gCount"], cb["gGridIndex"]] = [n, gi]; // finalizeGridVolumes numbers grids in this order
        [root["gCoords"], root["gResults"]] = [coordBuf, resultBuf];
        pass.execute(device.renderContext, n, 1);
        const r = new Float32Array((await device.renderContext.readBuffer(resultBuf)).buffer);
        let [decodeBad, nvdbBad, majorantBad, maxErr, maxDev] = [0, 0, 0, 0, 0];
        const span = grid.maxValue - grid.minValue;
        for (let i = 0; i < n; i++) {
            const rel = [0, 1, 2].map((k) => coords[i * 4 + k]! - lo[k]!);
            const [v, brick, slack] = [r[i * 4]!, r[i * 4 + 1]!, r[i * 4 + 2]!];
            const [expected, tol] = cpuBrickValue(grid, atlas, rel);
            const dev = Math.abs(brick - expected);
            maxDev = Math.max(maxDev, dev / Math.max(tol, 1e-12));
            if (dev > tol + 1e-6) decodeBad++;
            const err = Math.abs(brick - v);
            maxErr = Math.max(maxErr, err);
            // BC4: 8-bit endpoints and 7 steps over each 4x4 tile's range.
            if (err > span * (1 / 14 + 2 / 255) + 1e-6) nvdbBad++;
            if (!(slack >= 0)) majorantBad++;
        }
        console.error(`# grid ${gi}: bricks ${b.leafDim.join("x")} atlas ${b.atlasSize.join("x")} non-empty ${b.nonEmptyCount}, decode deviation ${maxDev.toFixed(2)} of tolerance, max |brick - nanovdb| ${maxErr.toExponential(2)} (range ${span.toFixed(3)})`);
        expectEq(decodeBad, 0, `grid ${gi}: GPU brick lookups match the uploaded bricks`);
        expectEq(nvdbBad, 0, `grid ${gi}: brick values within BC4 error of NanoVDB`);
        expectEq(majorantBad, 0, `grid ${gi}: every mip's majorant bounds the value`);
        coordBuf.destroy();
        resultBuf.destroy();
    }
    scene.destroy();
}

gpuTest("GridBricks.smokeMatchesNanoVDB", ({ device }) => checkGrids(device, "/Falcor/media/test_scenes/smoke.pyscene", "/Falcor/media/test_scenes", 1));
gpuTest("GridBricks.twoStackedGrids", ({ device }) => checkGrids(device, "/Falcor/media/test_scenes/two_volumes.pyscene", "/Falcor/media/test_scenes", 2));
