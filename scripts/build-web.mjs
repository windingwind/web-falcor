#!/usr/bin/env node
/**
 * Builds the static Mogwai site (GitHub Pages demo, @web-falcor/mogwai): the Vite build plus the runtime assets,
 * laid out like the dev server. NRD and RTXDI sources are not copied (their licence forbids it); the site fetches
 * them from their repos at runtime. Needs `npm run setup:web` (and `download:scenes -- test_scenes` for media).
 *
 *   node scripts/build-web.mjs [--base /web-falcor/] [--out <dir>] [--no-media]   (--no-media: only the default Cornell box)
 */

import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { kMathLibUpstream, kNRDUpstream } from "../packages/render-passes/src/NRDPass/NRDShaderPatch.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const argv = process.argv.slice(2);
const option = (name, fallback) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : fallback;
};
const base = option("--base", "/");
const outDir = resolve(repoRoot, option("--out", "packages/mogwai/dist"));
const withMedia = !argv.includes("--no-media");

// test_scenes content that is BSD-3 like the rest of Falcor's media; the rest carries its own licence.
const kExcludedMediaDirs = ["envmaps", "cesium_man", "grey_and_white_room", "mori_knob", "materials"];
const kExcludedMediaRe = new RegExp(`\\b(${kExcludedMediaDirs.join("|")})[/\\\\]`);

const src = (p) => join(repoRoot, p);
const dst = (p) => join(outDir, p);
function copy(from, to = from) {
    if (!existsSync(src(from))) throw new Error(`missing ${from} (run npm run setup:web)`);
    mkdirSync(dirname(dst(to)), { recursive: true });
    cpSync(src(from), dst(to), { recursive: true, dereference: true });
}
const writeJson = (p, value) => {
    mkdirSync(dirname(dst(p)), { recursive: true });
    writeFileSync(dst(p), JSON.stringify(value) + "\n");
};

// 1. The app, with its entry moved to the site root (asset URLs are absolute under `base`).
console.log(`vite build (base ${base}) -> ${outDir}`);
execFileSync(process.execPath, [src("node_modules/vite/bin/vite.js"), "build", "--base", base, "--outDir", outDir, "--emptyOutDir"], { cwd: src("packages/mogwai"), stdio: "inherit" });
renameSync(dst("packages/mogwai/index.html"), dst("index.html"));
rmSync(dst("packages"), { recursive: true, force: true });

// 2. Shader sources: Falcor's (BSD-3) and web-falcor's; SDK headers point at their upstream repos instead.
const list = JSON.parse(readFileSync(src("packages/falcor/shaders/generated/shader-file-list.json"), "utf8"));
for (const f of list.falcorFiles) copy(`Falcor/Source/Falcor/${f}`);
for (const f of list.renderPassFiles) copy(`Falcor/Source/${f}`);
for (const f of list.localFiles) copy(`packages/falcor/shaders/${f}`);
const externalFiles = list.externalFiles.map((e) => {
    if (e.upstream) return { path: e.path, url: e.upstream };
    copy(e.url.slice(1));
    return e;
});
writeJson("packages/falcor/shaders/generated/shader-file-list.json", { falcorFiles: list.falcorFiles, renderPassFiles: list.renderPassFiles, localFiles: list.localFiles, externalFiles });

// 3. NRD: its repo at the pinned tag, patched for WGSL by NRDPass when it loads.
const nrdFiles = JSON.parse(readFileSync(src("tools/nrd-3.1.0/shader-files.json"), "utf8")).map(({ path, url }) => {
    const rel = url.slice("/tools/nrd-3.1.0/".length);
    if (rel === "External/MathLib/STL.hlsli") return { path, url: `${kMathLibUpstream}/STL.hlsli` };
    return { path, url: `${kNRDUpstream}/${rel}`, patch: rel };
});
writeJson("tools/nrd-3.1.0/shader-files.json", nrdFiles);

// 4. Compilers, Python, wasm decoders, fonts and Mogwai's graph scripts.
for (const dir of ["tools/slang-wasm", "tools/slang-wasm-2026.5.2"]) for (const f of ["slang-wasm.js", "slang-wasm.wasm"]) copy(`${dir}/${f}`);
copy("tools/pyodide-packages");
for (const f of ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]) copy(`node_modules/pyodide/${f}`);
for (const f of ["tinyusdz.js", "tinyusdz.wasm", "LICENSE"]) copy(`node_modules/tinyusdz/${f}`);
for (const f of ["draco_decoder_nodejs.js", "draco_decoder.wasm"]) copy(`node_modules/draco3d/${f}`);
copy("Falcor/data/framework/fonts/dejavu-sans-mono-14.bin");
copy("Falcor/data/framework/fonts/dejavu-sans-mono-14.dds");
for (const f of readdirSync(src("Falcor/scripts")).filter((f) => f.endsWith(".py"))) copy(`Falcor/scripts/${f}`);

// 5. Demo scenes: the BSD test scenes that reference none of the excluded folders.
if (withMedia) {
    const root = "Falcor/media/test_scenes";
    if (!existsSync(src(root))) throw new Error(`missing ${root} (run npm run download:scenes -- test_scenes, or pass --no-media)`);
    const scenes = [];
    const walk = (rel) => {
        for (const name of readdirSync(src(rel))) {
            const p = `${rel}/${name}`;
            if (name === "__pycache__" || kExcludedMediaDirs.includes(name)) continue;
            if (statSync(src(p)).isDirectory()) walk(p);
            else if (/\.(pyscene|py|usda)$/.test(name) && kExcludedMediaRe.test(readFileSync(src(p), "utf8"))) continue;
            else {
                copy(p);
                if (name.endsWith(".pyscene")) scenes.push(p.slice("Falcor/media/".length));
            }
        }
    };
    walk(root);
    copy("Falcor/media/LICENSE.md");
    writeJson("demo-scenes.json", scenes.sort());
    console.log(`demo scenes: ${scenes.join(", ")}`);
} else {
    // The viewer's default scene (procedural, no assets).
    copy("Falcor/media/test_scenes/cornell_box.pyscene");
    copy("Falcor/media/LICENSE.md");
    writeJson("demo-scenes.json", ["test_scenes/cornell_box.pyscene"]);
}

// 6. Licences, and no Jekyll on GitHub Pages.
for (const f of ["LICENSE", "THIRD-PARTY-NOTICES.md", "licenses"]) copy(f);
writeFileSync(dst(".nojekyll"), "");

const size = execFileSync("du", ["-sh", outDir], { encoding: "utf8" }).split("\t")[0];
console.log(`\nstatic site ready: ${outDir} (${size})`);
