#!/usr/bin/env node
/**
 * Builds packages/falcor/wasm/assimp.{mjs,wasm}: Assimp 5.2.5 (the version Falcor's
 * AssimpImporter links) compiled with Emscripten, wrapped by scripts/assimp/import.cpp so
 * callers pass native's post-process flags and get the scene as assjson. The heap may
 * grow to 4 GB (BistroExterior needs more than 2). The outputs are committed; rerun this
 * only to rebuild them. Needs git, curl, tar, cmake and Python >= 3.10 (EMSDK_PYTHON).
 *
 *   node scripts/build-assimp-wasm.mjs
 */
import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const EMSDK_VERSION = "3.1.74";
const ASSIMP_TAG = "v5.2.5";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const tools = join(repoRoot, "tools");
const emsdk = join(tools, "emsdk");
const src = join(tools, `assimp-${ASSIMP_TAG.slice(1)}`);
const build = join(tools, "assimp-build");
const out = join(repoRoot, "packages/falcor/wasm");
const env = `source ${join(emsdk, "emsdk_env.sh")} >/dev/null &&`;
const run = (cmd, cwd = tools) => execSync(cmd, { cwd, stdio: "inherit", shell: "/bin/bash" });

mkdirSync(tools, { recursive: true });
if (!existsSync(emsdk)) run("git clone --depth 1 https://github.com/emscripten-core/emsdk.git");
run(`./emsdk install ${EMSDK_VERSION} && ./emsdk activate ${EMSDK_VERSION}`, emsdk);
if (!existsSync(src)) run(`curl -sL https://github.com/assimp/assimp/archive/refs/tags/${ASSIMP_TAG}.tar.gz | tar xz`);

mkdirSync(build, { recursive: true });
const cmakeOptions = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DASSIMP_BUILD_TESTS=OFF",
    "-DASSIMP_BUILD_ASSIMP_TOOLS=OFF",
    "-DASSIMP_BUILD_SAMPLES=OFF",
    // Every importer; only the assjson exporter (the wrapper's output format).
    "-DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT=OFF",
    "-DASSIMP_BUILD_ASSJSON_EXPORTER=ON",
    "-DASSIMP_BUILD_ZLIB=ON",
    "-DASSIMP_BUILD_DRACO=OFF",
    "-DASSIMP_WARNINGS_AS_ERRORS=OFF",
    "-DASSIMP_INSTALL=OFF",
];
run(`${env} emcmake cmake ${src} ${cmakeOptions.join(" ")}`, build);
run(`${env} emmake make -j8 assimp`, build);
mkdirSync(join(build, "out"), { recursive: true });

const exported = ["clear_files", "add_file", "import", "result", "result_size", "error", "free_result"].map((f) => `_ai_${f}`);
run(
    [
        env,
        "em++ -O2 -std=c++17",
        `-I${join(src, "include")} -I${join(build, "include")}`,
        join(repoRoot, "scripts/assimp/import.cpp"),
        join(build, "lib/libassimp.a"),
        join(build, "contrib/zlib/libzlibstatic.a"),
        `-o ${join(build, "out/assimp.mjs")}`,
        "-sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sENVIRONMENT=web,worker,node",
        `-sEXPORTED_FUNCTIONS=${[...exported, "_malloc", "_free"].join(",")}`,
        "-sEXPORTED_RUNTIME_METHODS=HEAPU8,UTF8ToString,stringToNewUTF8",
    ].join(" "),
    build,
);
mkdirSync(out, { recursive: true });
for (const f of ["assimp.mjs", "assimp.wasm"]) copyFileSync(join(build, "out", f), join(out, f));
copyFileSync(join(src, "LICENSE"), join(out, "assimp-LICENSE.txt"));
console.log(`Assimp ${ASSIMP_TAG} wasm written to packages/falcor/wasm/`);
