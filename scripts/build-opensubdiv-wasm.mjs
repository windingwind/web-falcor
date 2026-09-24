#!/usr/bin/env node
/**
 * Builds packages/falcor/wasm/opensubdiv.{mjs,wasm}: OpenSubdiv's Bfr tessellation
 * (the version Falcor's USD importer links, v3_5_0) compiled with Emscripten, wrapped by
 * scripts/opensubdiv/tessellate.cpp. The outputs are committed; rerun this only to
 * rebuild them. Needs git, curl, tar and Python >= 3.10 (EMSDK_PYTHON) for emsdk.
 *
 *   node scripts/build-opensubdiv-wasm.mjs
 */
import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const EMSDK_VERSION = "3.1.74";
const OSD_TAG = "v3_5_0";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const tools = join(repoRoot, "tools");
const emsdk = join(tools, "emsdk");
const osd = join(tools, `OpenSubdiv-${OSD_TAG.slice(1)}`);
const out = join(repoRoot, "packages/falcor/wasm");
const run = (cmd, cwd = tools) => execSync(cmd, { cwd, stdio: "inherit", shell: "/bin/bash" });

mkdirSync(tools, { recursive: true });
if (!existsSync(emsdk)) run("git clone --depth 1 https://github.com/emscripten-core/emsdk.git");
run(`./emsdk install ${EMSDK_VERSION} && ./emsdk activate ${EMSDK_VERSION}`, emsdk);
if (!existsSync(osd)) run(`curl -sL https://github.com/PixarAnimationStudios/OpenSubdiv/archive/refs/tags/${OSD_TAG}.tar.gz | tar xz`);

const sources = ["sdc", "vtr", "far", "bfr"].flatMap((dir) =>
    readdirSync(join(osd, "opensubdiv", dir))
        .filter((f) => f.endsWith(".cpp"))
        .map((f) => join(osd, "opensubdiv", dir, f)),
);
const exported = ["tessellate", "positions", "position_count", "normals", "uvs", "uv_count", "indices", "coarse_faces"].map((f) => `_osd_${f}`);
const build = join(tools, "osd-build");
mkdirSync(build, { recursive: true });
run(
    [
        `source ${join(emsdk, "emsdk_env.sh")} >/dev/null &&`,
        "em++ -O2 -std=c++14",
        `-I${osd}`,
        ...sources,
        join(repoRoot, "scripts/opensubdiv/tessellate.cpp"),
        `-o ${join(build, "opensubdiv.mjs")}`,
        "-sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=web,worker,node",
        `-sEXPORTED_FUNCTIONS=${[...exported, "_malloc", "_free"].join(",")}`,
        "-sEXPORTED_RUNTIME_METHODS=HEAPF32,HEAP32",
    ].join(" "),
);
mkdirSync(out, { recursive: true });
for (const f of ["opensubdiv.mjs", "opensubdiv.wasm"]) copyFileSync(join(build, f), join(out, f));
copyFileSync(join(osd, "LICENSE.txt"), join(out, "OpenSubdiv-LICENSE.txt"));
copyFileSync(join(osd, "NOTICE.txt"), join(out, "OpenSubdiv-NOTICE.txt"));
console.log(`OpenSubdiv ${OSD_TAG} wasm written to packages/falcor/wasm/`);
