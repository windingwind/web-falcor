/**
 * PixelStats override regression: _PIXEL_STATS_ENABLED must propagate into
 * the imported Rendering.Utils.PixelStats module, the packed Atomic<uint>
 * stats buffer must bind, and region indexing must land counters per pixel.
 */

import { Buffer, ComputePass, MemoryType, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("PixelStats.overrideCounts", async ({ device }) => {
    const dim = 8;
    const n = dim * dim;
    const pass = ComputePass.create(device, {
        path: "WebFalcor/PixelStatsProbe.cs.slang",
        defines: { _PIXEL_STATS_ENABLED: 1 },
    });
    const stats = new Buffer(device, {
        size: n * 5 * 4,
        structSize: 4,
        bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
        memoryType: MemoryType.DeviceLocal,
        name: "probe::stats",
    });
    const root = pass.getRootVar();
    root["ProbeCB"]!["gDim"] = [dim, dim];
    root["PixelStatsCB"]!["gPixelStatsDim"] = [dim, dim];
    root["gStatsBuffer"] = stats;

    const ctx = device.renderContext;
    ctx.clearBuffer(stats);
    pass.execute(ctx, dim, dim);
    const data = new Uint32Array((await ctx.readBuffer(stats)).buffer);

    let visOk = true;
    let hitOk = true;
    let lenOk = true;
    let vtxOk = true;
    for (let y = 0; y < dim; y++) {
        for (let x = 0; x < dim; x++) {
            const i = y * dim + x;
            if (data[i]! !== 1) visOk = false;
            if (data[n + i]! !== (x & 1)) hitOk = false;
            if (data[2 * n + i]! !== x + y) lenOk = false;
            if (data[3 * n + i]! !== 1) vtxOk = false;
        }
    }
    console.error(`# pixelstats probe: vis=${visOk} hit=${hitOk} len=${lenOk} vtx=${vtxOk} [0..3]=${data[0]},${data[1]},${data[2]},${data[3]}`);
    expectEq(visOk, true, "visibility region");
    expectEq(hitOk, true, "closestHit region");
    expectEq(lenOk, true, "pathLength region");
    expectEq(vtxOk, true, "pathVertex region");
});
