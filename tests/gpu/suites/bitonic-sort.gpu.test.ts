/**
 * Portable BitonicSort override vs CPU reference: each chunk sorts
 * ascending in-place, tails pad as UINT_MAX (out-of-range elements
 * untouched). Exact equality across chunk/group/size combinations,
 * including the warp-crossing (>32) and non-multiple-total paths.
 */

import { BitonicSort, Buffer, MemoryType, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("BitonicSort.matchesCpuChunkSort", async ({ device }) => {
    const sorter = new BitonicSort(device);
    const ctx = device.renderContext;

    const cases: { total: number; chunk: number; group: number }[] = [
        { total: 4096, chunk: 16, group: 256 },   // shared-only minor steps
        { total: 4096, chunk: 256, group: 256 },  // chunk == group
        { total: 5000, chunk: 128, group: 512 },  // total not a multiple of group
        { total: 100000, chunk: 1024, group: 1024 }, // max sizes, 2D dispatch
    ];

    for (const { total, chunk, group } of cases) {
        // Deterministic LCG data (Math.random is fine here but keep it reproducible).
        let seed = 0x12345678;
        const lcg = () => (seed = (seed * 1664525 + 1013904223) >>> 0);
        const data = new Uint32Array(total);
        for (let i = 0; i < total; i++) data[i] = lcg();

        const buf = new Buffer(device, {
            size: total * 4,
            structSize: 4,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name: "BitonicSortTest::data",
        });
        buf.setBlob(data);
        sorter.execute(ctx, buf, total, chunk, group);
        const gpu = new Uint32Array((await buf.getBlob()).buffer);

        // CPU reference: sort each chunk's in-range elements ascending.
        const ref = new Uint32Array(data);
        for (let base = 0; base < total; base += chunk) {
            const end = Math.min(base + chunk, total);
            const slice = [...ref.subarray(base, end)].sort((a, b) => a - b);
            ref.set(slice, base);
        }

        let mismatches = 0;
        for (let i = 0; i < total; i++) if (gpu[i] !== ref[i]) mismatches++;
        console.error(`# bitonic total=${total} chunk=${chunk} group=${group}: mismatches=${mismatches}`);
        expectEq(mismatches, 0, `total=${total} chunk=${chunk} group=${group}`);
        buf.destroy();
    }
});
