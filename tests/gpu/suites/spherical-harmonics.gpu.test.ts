/**
 * Utils/Math/SphericalHarmonics.slang on WGSL: the 16 real SH basis functions
 * (l <= 3) evaluated on the GPU match a CPU transcription of the same table.
 */

import { Buffer, ComputePass, MemoryType, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const M_1_SQRTPI = 0.5641895835477563;
function evalSH(idx: number, x: number, y: number, z: number): number {
    const k = 1 / (2 * M_1_SQRTPI);
    switch (idx) {
        case 0: return k;
        case 1: return y * Math.sqrt(3) * k;
        case 2: return z * Math.sqrt(3) * k;
        case 3: return x * Math.sqrt(3) * k;
        case 4: return x * y * Math.sqrt(15) * k;
        case 5: return y * z * Math.sqrt(15) * k;
        case 6: return (3 * z * z - 1) * Math.sqrt(5) / (4 * M_1_SQRTPI);
        case 7: return x * z * Math.sqrt(15) * k;
        case 8: return (x * x - y * y) * Math.sqrt(15) / (4 * M_1_SQRTPI);
        case 9: return y * (3 * x * x - y * y) * Math.sqrt(70) / (8 * M_1_SQRTPI);
        case 10: return x * y * z * Math.sqrt(105) * k;
        case 11: return y * (5 * z * z - 1) * Math.sqrt(42) / (8 * M_1_SQRTPI);
        case 12: return z * (5 * z * z - 3) * Math.sqrt(7) / (4 * M_1_SQRTPI);
        case 13: return x * (5 * z * z - 1) * Math.sqrt(42) / (8 * M_1_SQRTPI);
        case 14: return z * (x * x - y * y) * Math.sqrt(105) / (4 * M_1_SQRTPI);
        case 15: return x * (x * x - 3 * y * y) * Math.sqrt(70) / (8 * M_1_SQRTPI);
        default: return 0;
    }
}

gpuTest("SphericalHarmonics.basisMatchesCpu", async ({ device }) => {
    const ctx = device.renderContext;
    // Unit directions incl. axes and a few generic ones.
    const dirs: [number, number, number][] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]];
    for (let i = 0; i < 12; i++) {
        const theta = Math.acos(1 - (2 * (i + 0.5)) / 12);
        const phi = i * 2.399963;
        dirs.push([Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)]);
    }
    const n = dirs.length;
    const dirData = new Float32Array(n * 4);
    dirs.forEach((d, i) => dirData.set([...d, 0], i * 4));
    const dirBuf = new Buffer(device, { size: n * 16, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "sh::dirs" });
    dirBuf.setBlob(dirData);
    const result = new Buffer(device, { size: n * 16 * 4, structSize: 4, bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "sh::result" });

    const pass = ComputePass.create(device, { path: "WebFalcor/SHTest.cs.slang" });
    const root = pass.getRootVar();
    root["gDirs"] = dirBuf;
    root["gResult"] = result;
    pass.execute(ctx, n);
    ctx.submit();
    const gpu = new Float32Array((await result.getBlob()).buffer);

    let worst = 0;
    for (let i = 0; i < n; i++) {
        for (let idx = 0; idx < 16; idx++) {
            const ref = evalSH(idx, ...dirs[i]!);
            worst = Math.max(worst, Math.abs(gpu[i * 16 + idx]! - ref));
        }
    }
    console.error(`# spherical-harmonics: ${n} dirs x 16 basis, worst |gpu-cpu| = ${worst.toExponential(2)}`);
    expectEq(worst < 2e-6, true, `SH basis matches CPU (worst ${worst})`);
    // Native normalizes with 1/(2*M_1_SQRTPI) = sqrt(pi)/2 (its own convention, not 1/(2 sqrt(pi))).
    expectEq(Math.abs(gpu[0]! - Math.sqrt(Math.PI) / 2) < 1e-6, true, "Y00 = sqrt(pi)/2 (native table)");
    expectEq(Math.abs(gpu[2 * 16 + 2]! - (Math.sqrt(3) * Math.sqrt(Math.PI)) / 2) < 1e-6, true, "Y10 at +z = sqrt(3) sqrt(pi)/2");
});
