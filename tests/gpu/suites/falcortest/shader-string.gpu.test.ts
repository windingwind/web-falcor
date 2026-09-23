/**
 * Transplanted FalcorTest GPU tests: Slang/ShaderString — shader modules built
 * from files and code strings (ProgramDesc::addShaderModule().addString()).
 * ShaderStringImportDuplicate stays disabled as natively.
 */

import { MemoryType, Mt19937, ResourceBindFlags } from "@web-falcor/falcor";
import { gpuTest } from "../../harness/registry.js";
import { GPUUnitTestContext } from "../../harness/unit-test-context.js";
import { Expect } from "../../harness/expect.js";

const kShaderModuleA = "struct A\n{\n    ByteAddressBuffer buf;\n    uint c;\n    uint f(uint i)\n    {\n        return c * buf.Load(i * 4);\n    }\n}\n";
const kShaderModuleC = "import Tests.Slang.ShaderStringUtil;\nuint f(uint i)\n{\n    return test(i);\n}\n";
const kShaderModuleD = "uint f(uint i)\n{\n    return i * 997;\n}\n";
const kSize = 32;

async function expectResult(ctx: GPUUnitTestContext, name: string, want: (i: number) => number): Promise<void> {
    ctx.runProgram(kSize, 1, 1);
    const r = await ctx.readBuffer("result", Uint32Array);
    const e = new Expect();
    for (let i = 0; i < kSize; i++) e.check(r[i] === want(i) >>> 0, () => `i = ${i}: ${r[i]} != ${want(i) >>> 0}`);
    e.done(name);
}

gpuTest("FalcorTest.ShaderStringInline", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgramFromModules([{ sources: [{ file: "Tests/Slang/ShaderStringInline.cs.slang" }, { string: kShaderModuleA }] }]);
    ctx.allocateStructuredBuffer("result", kSize);
    const rng = new Mt19937();
    const values = Uint32Array.from({ length: kSize }, () => rng.next());
    const buf = device.createBuffer(values.byteLength, ResourceBindFlags.ShaderResource, MemoryType.DeviceLocal, values);
    ctx.vars()["gTest"]["moduleA"]["buf"] = buf;
    ctx.vars()["gTest"]["moduleA"]["c"] = 991;
    await expectResult(ctx, "ShaderStringInline", (i) => Math.imul(values[i]!, 991));
});

gpuTest("FalcorTest.ShaderStringModule", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgramFromModules([
        { name: "GeneratedModule", sources: [{ string: kShaderModuleD, path: "Tests/Slang/GeneratedModule.slang" }] },
        { sources: [{ file: "Tests/Slang/ShaderStringModule.cs.slang" }] },
    ]);
    ctx.allocateStructuredBuffer("result", kSize);
    await expectResult(ctx, "ShaderStringModule", (i) => i * 997);
});

gpuTest("FalcorTest.ShaderStringImport", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgramFromModules([{ sources: [{ file: "Tests/Slang/ShaderStringImport.cs.slang" }, { string: kShaderModuleC }] }]);
    ctx.allocateStructuredBuffer("result", kSize);
    await expectResult(ctx, "ShaderStringImport", (i) => i * 993);
});

gpuTest("FalcorTest.ShaderStringImported", async ({ device }) => {
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgramFromModules([
        { name: "GeneratedModule", sources: [{ string: kShaderModuleD, path: "Tests/Slang/GeneratedModule.slang" }] },
        { sources: [{ file: "Tests/Slang/ShaderStringImported.cs.slang" }] },
    ]);
    ctx.allocateStructuredBuffer("result", kSize);
    await expectResult(ctx, "ShaderStringImported", (i) => i * 997);
});

gpuTest("FalcorTest.ShaderStringDynamicObject", async ({ device }) => {
    const typeID = 55;
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgramFromModules(
        [{ name: "GeneratedModule", sources: [{ string: kShaderModuleD }] }, { sources: [{ file: "Tests/Slang/ShaderStringDynamic.cs.slang" }] }],
        "main",
        {},
        [{ typeName: "DynamicType", interfaceName: "IDynamicType", id: typeID }],
    );
    ctx.allocateStructuredBuffer("result", kSize);
    ctx.vars()["CB"]["type"] = typeID;
    await expectResult(ctx, "ShaderStringDynamicObject", (i) => i * 997);
});
