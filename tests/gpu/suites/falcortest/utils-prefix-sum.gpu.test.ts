/**
 * Transplanted FalcorTest Utils/PrefixSumTests: in-place exclusive scan at native's sizes (up to
 * 13.9M elements, sums near 2^32), the returned total and its copy in a separate buffer.
 */

import { MemoryType, Mt19937, PrefixSum, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { Expect } from "../../harness/expect.js";

/** Native prefixSumRef: exclusive scan in place (uint32 wrap), returns the total. */
function prefixSumRef(elems: Uint32Array): number {
    let sum = 0;
    for (let i = 0; i < elems.length; i++) {
        const tmp = elems[i]!;
        elems[i] = sum;
        sum = (sum + tmp) >>> 0;
    }
    return sum;
}

gpuTest("FalcorTest.PrefixSum", async ({ device }) => {
    const x = Uint32Array.from([5, 17, 2, 9, 23]);
    const e = new Expect();
    e.check(prefixSumRef(x) === 56 && [...x].join() === "0,5,22,24,33", () => "prefixSumRef");

    const prefixSum = new PrefixSum(device);
    for (const numElems of [1, 27, 64, 2049, 10201, 231917, 1088921, 13912615]) {
        // Random data whose total fits in 32 bits.
        const maxVal = Math.floor(0xffffffff / numElems);
        const r = new Mt19937();
        const testData = Uint32Array.from({ length: numElems }, () => r.next() % maxVal);
        const data = device.createBuffer(numElems * 4, ResourceBindFlags.UnorderedAccess, MemoryType.DeviceLocal, testData);
        const sumBuffer = device.createBuffer(4, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, new Uint32Array(1));

        const sum = await prefixSum.execute(device.renderContext, data, numElems, true, sumBuffer, 0);
        const refSum = prefixSumRef(testData);
        e.check(sum === refSum, () => `n = ${numElems}: sum ${sum} vs ${refSum}`);
        const resultSum = new Uint32Array((await device.renderContext.readBuffer(sumBuffer, 0, 4)).buffer, 0, 1)[0];
        e.check(resultSum === refSum, () => `n = ${numElems}: sum buffer ${resultSum} vs ${refSum}`);
        const result = new Uint32Array((await device.renderContext.readBuffer(data, 0, numElems * 4)).buffer, 0, numElems);
        let bad = 0;
        for (let i = 0; i < numElems; i++) if (result[i] !== testData[i]) bad++;
        e.check(bad === 0, () => `n = ${numElems}: ${bad} elements differ`);
        data.destroy();
        sumBuffer.destroy();
    }
    e.done("PrefixSum");
});
