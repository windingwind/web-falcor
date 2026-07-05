/**
 * M2 program-system GPU tests: ComputePass + ParameterBlock/ShaderVar driving
 * an unmodified-Falcor-shader-library kernel compiled at runtime by slang-wasm.
 */

import { ComputePass, MemoryType, ResourceBindFlags, ResourceFormat, float4x4, mulMatVec, float4 } from "@web-falcor/falcor";
import { gpuTest, expectEq, expectClose, expectArrayClose } from "../harness/registry.js";

gpuTest("ParameterBlock.nestedBlockPaths", async ({ device }) => {
    // The gScene binding pattern: ParameterBlock with uniforms (incl. matrix),
    // nested struct containing a buffer, texture + sampler — set via ShaderVar paths.
    const pass = ComputePass.create(device, { path: "NestedBlockTest.cs.slang" });
    const count = 4;
    const verts = new Float32Array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]);
    const vertexBuf = device.createBuffer(verts.byteLength, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, MemoryType.DeviceLocal, verts);
    const out = device.createStructuredBuffer(16, count, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    // 1x1 texture with value (0.25, 0.5, 0.75, 1).
    const tex = device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array([0.25, 0.5, 0.75, 1]));

    const m = float4x4.fromRows([[2, 0, 0, 10], [0, 3, 0, 20], [0, 0, 4, 30], [0, 0, 0, 1]]);
    const root = pass.getRootVar();
    root["gBlock"]["transform"] = m;
    root["gBlock"]["offset"] = [100, 200, 300];
    root["gBlock"]["count"] = count;
    root["gBlock"]["vertices"]["data0"] = vertexBuf;
    root["gBlock"]["colorTex"] = tex;
    root["gBlock"]["sampler"] = device.createSampler();
    root["gOut"] = out;

    pass.execute(device.renderContext, count);
    const gpu = new Float32Array((await out.getBlob()).buffer);
    for (let i = 0; i < count; i++) {
        const v = new float4(verts[i * 4]!, verts[i * 4 + 1]!, verts[i * 4 + 2]!, verts[i * 4 + 3]!);
        const t = mulMatVec(m, v);
        const expected = [t.x + 100 + 0.25, t.y + 200 + 0.5, t.z + 300 + 0.75, t.w + 0 + 1];
        expectArrayClose(gpu.subarray(i * 4, i * 4 + 4), expected, 1e-4, `out[${i}]`);
    }
    vertexBuf.destroy();
    out.destroy();
    tex.destroy();
});

gpuTest("ParameterBlock.matrixCbufferLayout", async ({ device }) => {
    // Non-diagonal matrix through Slang's mul(): GPU result must match the
    // CPU math library (verifies the ColMajor cbuffer transposition).
    const pass = ComputePass.create(device, { path: "MatrixTest.cs.slang" });
    const out = device.createStructuredBuffer(16, 1, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);

    const m = float4x4.fromRows([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [0, 0, 0, 1],
    ]);
    const p = new float4(0.5, -1.5, 2.5, 1);

    const root = pass.getRootVar();
    root["CB"]["gWorld"] = m;
    root["CB"]["gPoint"] = p.toArray();
    root["gOut"] = out;
    pass.execute(device.renderContext, 1);

    const gpu = new Float32Array((await out.getBlob()).buffer, 0, 4);
    const cpu = mulMatVec(m, p);
    expectArrayClose(gpu, cpu.toArray(), 1e-5, "mul(M, v) GPU vs CPU");
    out.destroy();
});

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
