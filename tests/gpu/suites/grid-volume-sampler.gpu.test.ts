/**
 * GridVolumeSampler (Rendering/Volumes/GridVolumeSampler.slang, unmodified): every transmittance estimator, with the
 * NanoVDB grid or the BrickedGrid (local-majorant DDA), estimates the same transmittance through smoke.pyscene's
 * volume, and both distance samplers scatter with probability 1 - T.
 */

import { Buffer, ComputePass, DefineList, DistanceSampler, GridVolumeSampler, MemoryType, ResourceBindFlags, TransmittanceEstimator, initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("GridVolumeSampler.estimatorsAgree", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(device, await (await fetch("/Falcor/media/test_scenes/smoke.pyscene")).text(), "/Falcor/media/test_scenes");
    const b = scene.gridVolumes[0]!.bounds!;
    const n = 256;
    // Rays from outside the volume's bounds towards points inside it.
    const rays = new Float32Array(n * 8);
    let seed = 3;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const center = [0, 1, 2].map((k) => (b.min[k]! + b.max[k]!) / 2);
    const extent = [0, 1, 2].map((k) => b.max[k]! - b.min[k]!);
    const radius = Math.hypot(...extent);
    for (let i = 0; i < n; i++) {
        const target = [0, 1, 2].map((k) => b.min[k]! + extent[k]! * (0.2 + 0.6 * rnd()));
        const [u, v] = [rnd() * 2 - 1, rnd() * 2 * Math.PI];
        const origin = [center[0]! + radius * Math.sqrt(1 - u * u) * Math.cos(v), center[1]! + radius * u, center[2]! + radius * Math.sqrt(1 - u * u) * Math.sin(v)];
        const dir = [0, 1, 2].map((k) => target[k]! - origin[k]!);
        const len = Math.hypot(...dir);
        rays.set([...origin, 0, ...dir.map((d) => d / len), 0], i * 8);
    }
    const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
    const rayBuf = new Buffer(device, { size: rays.byteLength, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal });
    rayBuf.setBlob(rays);
    const run = async (options: ConstructorParameters<typeof GridVolumeSampler>[1], samples: number) => {
        const sampler = new GridVolumeSampler(scene, options);
        const pass = ComputePass.create(device, { path: "WebFalcor/GridVolumeSamplerTest.cs.slang", defines: new DefineList().addAll(scene.getSceneDefines()).addAll(sampler.getDefines()) });
        const out = new Buffer(device, { size: n * 8, structSize: 8, bindFlags: storage, memoryType: MemoryType.DeviceLocal });
        const root = pass.getRootVar();
        scene.bindShaderData(root);
        const cb = root["CB"] as Record<string, unknown>;
        [cb["gRayCount"], cb["gSamples"]] = [n, samples];
        [root["gRays"], root["gResults"]] = [rayBuf, out];
        pass.execute(device.renderContext, n, 1);
        const r = new Float32Array((await device.renderContext.readBuffer(out)).buffer);
        out.destroy();
        return r;
    };
    // Reference: ratio tracking over the NanoVDB grid.
    const ref = await run({ useBrickedGrid: false, transmittanceEstimator: TransmittanceEstimator.RatioTracking, distanceSampler: DistanceSampler.DeltaTracking }, 8192);
    const meanRefT = Array.from({ length: n }, (_v, i) => ref[i * 2]!).reduce((a, x) => a + x, 0) / n;
    expectEq(meanRefT > 0.1 && meanRefT < 0.95, true, `rays cross the smoke (mean T ${meanRefT.toFixed(3)})`);
    const configs: [string, ConstructorParameters<typeof GridVolumeSampler>[1]][] = [
        ["DeltaTracking/NanoVDB", { useBrickedGrid: false, transmittanceEstimator: TransmittanceEstimator.DeltaTracking, distanceSampler: DistanceSampler.DeltaTracking }],
        ["RatioTracking/bricks", { useBrickedGrid: true, transmittanceEstimator: TransmittanceEstimator.RatioTracking, distanceSampler: DistanceSampler.DeltaTracking }],
        ["RatioTrackingLocalMajorant/bricks", { useBrickedGrid: true, transmittanceEstimator: TransmittanceEstimator.RatioTrackingLocalMajorant, distanceSampler: DistanceSampler.DeltaTrackingLocalMajorant }],
    ];
    for (const [name, options] of configs) {
        const r = await run(options, 2048);
        let [dT, dScatter, bias] = [0, 0, 0];
        for (let i = 0; i < n; i++) {
            dT += Math.abs(r[i * 2]! - ref[i * 2]!) / n;
            bias += (r[i * 2]! - ref[i * 2]!) / n;
            // A distance is sampled inside the volume with probability 1 - T.
            dScatter += Math.abs(r[i * 2 + 1]! - (1 - ref[i * 2]!)) / n;
        }
        console.error(`# ${name}: mean |T - ref| ${dT.toFixed(4)}, bias ${bias.toFixed(4)}, mean |P(scatter) - (1 - T)| ${dScatter.toFixed(4)}`);
        expectEq(dT < 0.02 && Math.abs(bias) < 0.01, true, `${name} transmittance matches the reference`);
        expectEq(dScatter < 0.02, true, `${name} scattering probability is 1 - T`);
    }
    rayBuf.destroy();
    scene.destroy();
});
