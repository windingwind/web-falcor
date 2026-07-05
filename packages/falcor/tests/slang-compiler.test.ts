/**
 * SlangCompiler unit tests (Node): WGSL emission, reflection, and Falcor's
 * translation-unit define semantics (defines must reach imported modules).
 */

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { DefineList } from "../src/Core/Program/DefineList.js";
import { initSlang, SlangCompiler, ShaderType } from "../src/Core/Program/SlangCompiler.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const falcorShaderRoot = resolve(repoRoot, "Falcor/Source/Falcor");

const kTestModule = `
// Module whose behavior depends on a program-level define.
#ifndef WIDGET_SCALE
#define WIDGET_SCALE 1
#endif
public uint scaleWidget(uint x) {
#if WIDGET_MODE == 2
    return x * uint(WIDGET_SCALE) * 100u;
#else
    return x * uint(WIDGET_SCALE);
#endif
}
`;

const kEntry = `
import TestModule;
import Utils.Math.HashUtils;

RWStructuredBuffer<uint> gOut;

[numthreads(64,1,1)]
void csMain(uint3 tid : SV_DispatchThreadID)
{
    gOut[tid.x] = scaleWidget(jenkinsHash(tid.x));
}
`;

const kSources = new Map<string, string>([
    ["TestModule.slang", kTestModule],
    ["TestEntry.cs.slang", kEntry],
    ["Utils/Math/HashUtils.slang", readFileSync(resolve(falcorShaderRoot, "Utils/Math/HashUtils.slang"), "utf8")],
]);

function makeCompiler(): SlangCompiler {
    return new SlangCompiler((path) => kSources.get(path), [...kSources.keys()]);
}

describe("SlangCompiler", () => {
    beforeAll(async () => {
        await initSlang(resolve(repoRoot, "tools/slang-wasm/slang-wasm.js"));
    });

    it("compiles a compute entry importing Falcor modules to WGSL", () => {
        const compiler = makeCompiler();
        const result = compiler.compile("TestEntry.cs.slang", [{ name: "csMain", type: ShaderType.Compute }]);
        expect(result.entryPointCode).toHaveLength(1);
        const wgsl = result.entryPointCode[0]!;
        expect(wgsl).toContain("@compute");
        expect(wgsl).toContain("fn csMain");
        expect(wgsl).toContain("jenkinsHash");
    });

    it("exposes reflection parameters", () => {
        const compiler = makeCompiler();
        const result = compiler.compile("TestEntry.cs.slang", [{ name: "csMain", type: ShaderType.Compute }]);
        const params = result.reflection.parameters ?? [];
        expect(params.map((p) => p.name)).toContain("gOut");
    });

    it("injects program defines into imported modules (translation-unit semantics)", () => {
        const compiler = makeCompiler();
        const defines = new DefineList().add("WIDGET_MODE", 2).add("WIDGET_SCALE", 7);
        const result = compiler.compile("TestEntry.cs.slang", [{ name: "csMain", type: ShaderType.Compute }], defines);
        const wgsl = result.entryPointCode[0]!;
        // WIDGET_MODE == 2 branch multiplies by 100 -> constant must appear.
        expect(wgsl).toMatch(/100/);
        expect(wgsl).toMatch(/7/);
    });

    it("caches sessions per define-set and separates their outputs", () => {
        const compiler = makeCompiler();
        const a = compiler.compile("TestEntry.cs.slang", [{ name: "csMain", type: ShaderType.Compute }], new DefineList().add("WIDGET_MODE", 1));
        const b = compiler.compile("TestEntry.cs.slang", [{ name: "csMain", type: ShaderType.Compute }], new DefineList().add("WIDGET_MODE", 2));
        expect(a.entryPointCode[0]).not.toEqual(b.entryPointCode[0]);
    });
});
