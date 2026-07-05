/**
 * M2 program-system GPU tests: ComputePass + ParameterBlock/ShaderVar driving
 * an unmodified-Falcor-shader-library kernel compiled at runtime by slang-wasm.
 */

import { ComputePass, MemoryType, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("ComputePass.falcorShaderRoundtrip", async ({ device }) => {
    // SanityCompute.cs.slang imports Utils.Math.MathHelpers + TinyUniformSampleGenerator
    // from the upstream Falcor tree; compiled in-browser by slang-wasm.
    const pass = ComputePass.create(device, { path: "SanityCompute.cs.slang" });
    const elementCount = 256;
    const output = device.createStructuredBuffer(16, elementCount, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);

    const root = pass.getRootVar();
    root["gOutput"] = output;
    root["PerFrame"]["gFrameIndex"] = 7;
    root["PerFrame"]["gElementCount"] = elementCount;

    const ctx = device.renderContext;
    pass.execute(ctx, elementCount);
    const data = new Float32Array((await output.getBlob()).buffer);

    // Every element: unit-length direction sampled on the sphere, w == 1.
    let checked = 0;
    for (let i = 0; i < elementCount; i++) {
        const [x, y, z, w] = [data[i * 4]!, data[i * 4 + 1]!, data[i * 4 + 2]!, data[i * 4 + 3]!];
        const len = Math.hypot(x, y, z);
        expectClose(len, 1.0, 1e-3, `|dir[${i}]|`);
        expectEq(w, 1, `w[${i}]`);
        checked++;
    }
    expectEq(checked, elementCount, "all elements checked");
    output.destroy();
});

gpuTest("ComputePass.threadGroupSizeFromReflection", ({ device }) => {
    const pass = ComputePass.create(device, { path: "SanityCompute.cs.slang" });
    expectEq(pass.getThreadGroupSize(), [64, 1, 1], "workgroup size from Slang reflection");
});

gpuTest("ComputePass.defineVariants", async ({ device }) => {
    // Same source, two define-sets -> distinct kernels (ProgramVersion semantics).
    const passScaled = ComputePass.create(device, { path: "DefineTest.cs.slang", defines: { SCALE_MODE: 2 } });
    const passPlain = ComputePass.create(device, { path: "DefineTest.cs.slang" });

    const buf1 = device.createStructuredBuffer(4, 4, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    const buf2 = device.createStructuredBuffer(4, 4, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    passScaled.getRootVar()["gData"] = buf1;
    passPlain.getRootVar()["gData"] = buf2;

    const ctx = device.renderContext;
    passScaled.execute(ctx, 4);
    passPlain.execute(ctx, 4);

    const scaled = new Uint32Array((await buf1.getBlob()).buffer);
    const plain = new Uint32Array((await buf2.getBlob()).buffer);
    expectEq(scaled[3], 300, "SCALE_MODE=2 → x*100");
    expectEq(plain[3], 3, "default → x*1");
    buf1.destroy();
    buf2.destroy();
});
