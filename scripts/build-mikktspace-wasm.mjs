#!/usr/bin/env node
/**
 * Builds packages/falcor/wasm/mikktspace.wasm: Falcor's bundled MikkTSpace
 * (Falcor/external/mikktspace/mikktspace.c) behind scripts/mikktspace/mikk_wasm.c, as a
 * standalone wasm module with no imports (Scene/MikkTSpace.ts instantiates it directly).
 * The output is committed; rerun this only to rebuild it. Needs the emsdk of build-assimp-wasm.mjs and
 * Python >= 3.10 (EMSDK_PYTHON).
 *
 *   node scripts/build-mikktspace-wasm.mjs
 */
import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const EMSDK_VERSION = "3.1.74";
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const tools = join(repoRoot, "tools");
const emsdk = join(tools, "emsdk");
const mikk = join(repoRoot, "Falcor/external/mikktspace");
const build = join(tools, "mikktspace-build");
const out = join(repoRoot, "packages/falcor/wasm");
const run = (cmd, cwd = tools) => execSync(cmd, { cwd, stdio: "inherit", shell: "/bin/bash" });

if (!existsSync(join(mikk, "mikktspace.c"))) throw new Error("Falcor/external/mikktspace missing (the Falcor submodule)");
if (!existsSync(emsdk)) run("git clone --depth 1 https://github.com/emscripten-core/emsdk.git");
run(`./emsdk install ${EMSDK_VERSION} && ./emsdk activate ${EMSDK_VERSION}`, emsdk);
mkdirSync(build, { recursive: true });
// No FMA contraction: native computes these in plain f32 SSE arithmetic.
run(
    `source ${join(emsdk, "emsdk_env.sh")} >/dev/null && emcc -O2 -DNDEBUG -ffp-contract=off -I${mikk} ${join(mikk, "mikktspace.c")} ${join(repoRoot, "scripts/mikktspace/mikk_wasm.c")} ` +
        `--no-entry -sSTANDALONE_WASM=1 -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=0 -sEXPORTED_FUNCTIONS=_mikk_generate,_mikk_alloc,_mikk_free -o ${join(build, "mikktspace.wasm")}`,
    build,
);
mkdirSync(out, { recursive: true });
copyFileSync(join(build, "mikktspace.wasm"), join(out, "mikktspace.wasm"));
// The license is the header comment of mikktspace.h.
const header = readFileSync(join(mikk, "mikktspace.h"), "utf8");
const start = header.lastIndexOf("/**", header.indexOf("Copyright"));
writeFileSync(join(out, "mikktspace-LICENSE.txt"), header.slice(start, header.indexOf("*/", start) + 2) + "\n");
console.log(`MikkTSpace wasm written to ${out}`);
