#!/usr/bin/env node
/**
 * Builds packages/falcor/wasm/nrd.{mjs,wasm}: the NRD 3.1.0 library (the version Falcor's
 * NRDPass links) compiled with Emscripten without embedded shader bytecode, wrapped by
 * scripts/nrd/nrd_wasm.cpp. NRDPass compiles the HLSL itself (tools/nrd-3.1.0, fetched by
 * scripts/setup-web.mjs). The settings field tables come from NRDSettings.h. The outputs are
 * committed; rerun this only to rebuild them. Needs the emsdk of build-assimp-wasm.mjs and
 * Python >= 3.10 (EMSDK_PYTHON).
 *
 *   node scripts/setup-web.mjs && node scripts/build-nrd-wasm.mjs
 */
import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const EMSDK_VERSION = "3.1.74";
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const tools = join(repoRoot, "tools");
const emsdk = join(tools, "emsdk");
const nrd = join(tools, "nrd-3.1.0");
const build = join(tools, "nrd-build");
const out = join(repoRoot, "packages/falcor/wasm");
const run = (cmd, cwd = tools) => execSync(cmd, { cwd, stdio: "inherit", shell: "/bin/bash" });

if (!existsSync(join(nrd, "Source/DenoiserImpl.cpp"))) throw new Error("tools/nrd-3.1.0 missing: run node scripts/setup-web.mjs first");
if (!existsSync(emsdk)) run("git clone --depth 1 https://github.com/emscripten-core/emsdk.git");
run(`./emsdk install ${EMSDK_VERSION} && ./emsdk activate ${EMSDK_VERSION}`, emsdk);
mkdirSync(join(build, "shim"), { recursive: true });

// Field tables (name, offset, type, count) of the settings structs JS edits in place.
const header = readFileSync(join(nrd, "Include/NRDSettings.h"), "utf8");
const structs = new Map();
for (const m of header.matchAll(/struct (\w+)\s*\{([\s\S]*?)\n {4}\};/g)) {
    const body = m[2].replace(/\/\/.*$/gm, "");
    structs.set(m[1], [...body.matchAll(/^\s*([\w:]+)\s+(\w+)(\[(\d+)\])?\s*(=[^;]*)?;/gm)].map((f) => ({ type: f[1], name: f[2], count: Number(f[4] ?? 1) })));
}
const scalar = { float: "f32", uint32_t: "u32", bool: "bool", CheckerboardMode: "u8", AccumulationMode: "u8", PrePassMode: "u32" };
const lines = [];
for (const root of ["CommonSettings", "RelaxDiffuseSpecularSettings", "RelaxDiffuseSettings", "ReblurSettings", "SpecularReflectionMvSettings", "SpecularDeltaMvSettings"]) {
    lines.push(`#define FIELDS_${root} \\`);
    const emit = (fields, prefix) => {
        for (const f of fields) {
            if (structs.has(f.type)) emit(structs.get(f.type), `${prefix}${f.name}.`);
            else if (scalar[f.type]) lines.push(`    F(${root}, ${prefix}${f.name}, "${scalar[f.type]}", ${f.count}) \\`);
            else throw new Error(`unknown field type ${f.type} in ${root}`);
        }
    };
    emit(structs.get(root), "");
    lines.push("");
}
writeFileSync(join(build, "fields.inc"), lines.join("\n") + "\n");
// MathLib includes <intrin.h> off x86; Emscripten emulates SSE4.1 over wasm SIMD.
writeFileSync(join(build, "shim/intrin.h"), "#pragma once\n#include <smmintrin.h>\n");

const sources = ["Source/Wrapper.cpp", "Source/DenoiserImpl.cpp", "Source/Timer.cpp"].map((s) => join(nrd, s)).join(" ");
run(
    `source ${join(emsdk, "emsdk_env.sh")} >/dev/null && em++ -O2 -std=c++17 -D__linux__=1 -I${build}/shim -I${build} -I${nrd}/Include -I${nrd}/Source -I${nrd}/External ` +
        `-msimd128 -msse4.1 ${sources} ${join(repoRoot, "scripts/nrd/nrd_wasm.cpp")} -sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH=1 ` +
        `-sEXPORTED_FUNCTIONS=_malloc,_free -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToNewUTF8,HEAPU8 -o ${join(build, "nrd.mjs")}`,
    build,
);
mkdirSync(out, { recursive: true });
for (const f of ["nrd.mjs", "nrd.wasm"]) copyFileSync(join(build, f), join(out, f));
console.log(`NRD wasm written to ${out}`);
