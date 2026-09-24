#!/usr/bin/env node
/**
 * Web setup — provisions everything needed to *use* web-falcor in a browser
 * WITHOUT cloning or building upstream Falcor.
 *
 * The runtime compiles Slang -> WGSL in the browser (shaders are specialized
 * per-scene, so a build-time-only pipeline can't cover them), which needs three
 * things a fresh checkout doesn't have (all .gitignored — see README):
 *
 *   1. The upstream Falcor shader *sources* (~340 .slang/.slangh text files).
 *      Fetched from GitHub at the pinned commit into Falcor/Source/** — the
 *      same paths the dev server serves, so no code changes are needed. This is
 *      just the text shaders; it does NOT clone the repo or run the native
 *      CMake build (that is only for the test oracles — see scripts below).
 *   2. The slang-wasm compiler (the official per-release build) into
 *      tools/slang-wasm/, plus the pinned autodiff build into tools/slang-wasm-2026.5.2/.
 *   3. The SDK shader headers those sources include — nanovdb/PNanoVDB.h (pulled
 *      in by Scene.slang, i.e. by EVERY scene-bound pass) and the RTXDI SDK
 *      headers — from their public repos at Falcor's pinned versions, into the
 *      packman link paths the dev server serves (manifest `externalFiles`).
 *   4. The NRD 3.1.0 SDK shaders NRDPass compiles (the version Falcor pins), cloned into
 *      tools/nrd-3.1.0/ with the MathLib submodule; their HLSL register bindings are dropped
 *      (WGSL numbers bindings per group; the pass binds by reflected name), `unorm` texture
 *      element types are stripped (no WGSL form), read-modify-write outputs are split into a
 *      write target and a read copy (WGSL read_write storage is r32-only), and the file list the pass loads is written next
 *      to them.
 *   5. The Pyodide packages Falcor's Python scripts import (numpy, pillow), pinned by the
 *      pyodide-lock.json of the installed pyodide and checked against its sha256,
 *      into tools/pyodide-packages/ (Scripting loads packages from there).
 *
 * What this does NOT fetch: media/test scenes (see scripts/download-scenes.mjs).
 *
 * Usage: node scripts/setup-web.mjs [--skip-slang] [--skip-shaders] [--skip-nrd] [--skip-pyodide-packages]
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync, readdirSync, statSync, copyFileSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// Pinned upstream versions (keep in sync with README / docs).
const FALCOR_COMMIT = "eb540f6748774680ce0039aaf3ac9279266ec521";
const SLANG_VERSION = "2026.18.2";
// Backward-mode autodiff kernels (WARDiffPathTracer) compile with the last release before Slang's
// autodiff refactor (#9808, in 2026.7), which crashes transposing them (docs/module-mapping.md).
const SLANG_AUTODIFF_VERSION = "2026.5.2";
const slangWasmUrl = (version) => `https://github.com/shader-slang/slang/releases/download/v${version}/slang-${version}-wasm.zip`;
const FALCOR_RAW = `https://raw.githubusercontent.com/NVIDIAGameWorks/Falcor/${FALCOR_COMMIT}`;

const args = new Set(process.argv.slice(2));
const CONCURRENCY = 24;

/** Runs `items` through `fn` with a bounded concurrency pool. */
async function pool(items, fn) {
    let i = 0;
    let done = 0;
    const total = items.length;
    async function worker() {
        while (i < total) {
            const idx = i++;
            await fn(items[idx], idx);
            done++;
            if (done % 40 === 0 || done === total) process.stdout.write(`\r  ${done}/${total}`);
        }
    }
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, total) }, worker));
    process.stdout.write("\n");
}

async function fetchWithRetry(url, read, attempts = 4) {
    for (let a = 1; ; a++) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await read(res);
        } catch (err) {
            if (a >= attempts) throw new Error(`${err.message} for ${url}`);
            // Backoff on transient failures (429 / network blips) — matters at
            // 340 shader files + the slang-wasm download on CI.
            await new Promise((r) => setTimeout(r, 400 * a * a));
        }
    }
}
const fetchText = (url) => fetchWithRetry(url, (res) => res.text());
const fetchBuffer = (url) => fetchWithRetry(url, async (res) => Buffer.from(await res.arrayBuffer()));

const manifestPath = join(repoRoot, "packages/falcor/shaders/generated/shader-file-list.json");
function readManifest() {
    if (!existsSync(manifestPath)) throw new Error(`missing shader manifest: ${manifestPath}`);
    return JSON.parse(readFileSync(manifestPath, "utf8"));
}

async function fetchShaders() {
    const manifest = readManifest();

    // (repo-relative dest, upstream URL) for every fetchable shader source.
    const jobs = [
        // falcorFiles live under Source/Falcor/**.
        ...manifest.falcorFiles.map((f) => ({ dest: join(repoRoot, "Falcor/Source/Falcor", f), url: `${FALCOR_RAW}/Source/Falcor/${f}` })),
        // renderPassFiles already carry the "RenderPasses/" prefix, under Source/**.
        ...manifest.renderPassFiles.map((f) => ({ dest: join(repoRoot, "Falcor/Source", f), url: `${FALCOR_RAW}/Source/${f}` })),
    ];
    console.log(`Fetching ${jobs.length} Falcor shader sources @ ${FALCOR_COMMIT.slice(0, 8)} (no clone, no native build)`);

    const failures = [];
    await pool(jobs, async ({ dest, url }) => {
        try {
            const text = await fetchText(url);
            mkdirSync(dirname(dest), { recursive: true });
            writeFileSync(dest, text);
        } catch (err) {
            failures.push(`${url}: ${err.message}`);
        }
    });
    if (failures.length > 0) {
        console.error(`\n${failures.length} shader(s) failed to fetch (first 5):`);
        for (const f of failures.slice(0, 5)) console.error(`  ${f}`);
        throw new Error("shader fetch incomplete — see errors above");
    }
    console.log("  Falcor shader sources ready under Falcor/Source/");
}

/** SDK headers the shaders include (manifest externalFiles), fetched from their public
 *  upstream copies into the packman link paths; paths a full setup already links are kept. */
async function fetchExternalHeaders() {
    const entries = (readManifest().externalFiles ?? []).filter((e) => e.upstream);
    const jobs = entries.filter((e) => !existsSync(join(repoRoot, e.url)));
    if (jobs.length === 0) {
        console.log("SDK shader headers (NanoVDB, RTXDI) already present — skipping");
        return;
    }
    console.log(`Fetching ${jobs.length} SDK shader headers (NanoVDB: MPL-2.0; RTXDI SDK: NVIDIA RTX SDKs license)`);
    const failures = [];
    await pool(jobs, async ({ url, upstream }) => {
        try {
            const text = await fetchText(upstream);
            const dest = join(repoRoot, url);
            mkdirSync(dirname(dest), { recursive: true });
            writeFileSync(dest, text);
        } catch (err) {
            failures.push(`${upstream}: ${err.message}`);
        }
    });
    if (failures.length > 0) {
        console.error(`\n${failures.length} SDK header(s) failed to fetch:`);
        for (const f of failures) console.error(`  ${f}`);
        throw new Error("SDK header fetch incomplete — see errors above");
    }
    console.log("  SDK shader headers ready under Falcor/external/packman/");
}

async function fetchSlangWasm(version, dir) {
    const outDir = join(repoRoot, dir);
    if (existsSync(join(outDir, "slang-wasm.wasm")) && existsSync(join(outDir, "slang-wasm.js"))) {
        console.log(`slang-wasm ${version} already present — skipping (delete ${dir} to refetch)`);
        return;
    }
    console.log(`Downloading slang-wasm ${version} (~25 MB)…`);
    const zipBytes = await fetchBuffer(slangWasmUrl(version));
    const scratch = join(tmpdir(), `slang-wasm-${process.pid}`);
    rmSync(scratch, { recursive: true, force: true });
    mkdirSync(scratch, { recursive: true });
    const zipPath = join(scratch, "slang.zip");
    writeFileSync(zipPath, zipBytes);
    // System unzip (present on CI ubuntu + dev machines); no npm zip dependency.
    execFileSync("unzip", ["-o", "-q", zipPath, "-d", scratch]);

    // Copy the runtime artifacts (js + wasm + optional .d.ts) wherever they land.
    mkdirSync(outDir, { recursive: true });
    const wanted = new Set(["slang-wasm.js", "slang-wasm.wasm"]);
    let copied = 0;
    const walk = (dir) => {
        for (const name of readdirSync(dir)) {
            const p = join(dir, name);
            if (statSync(p).isDirectory()) walk(p);
            else if (wanted.has(name) || name.endsWith(".d.ts")) {
                copyFileSync(p, join(outDir, name));
                copied++;
            }
        }
    };
    walk(scratch);
    rmSync(scratch, { recursive: true, force: true });
    if (!existsSync(join(outDir, "slang-wasm.wasm"))) throw new Error("slang-wasm.wasm not found in the release zip");
    console.log(`  slang-wasm ready under ${dir}/ (${copied} files)`);
}

const NRD_TAG = "v3.1.0";

/** Fetches the NRD shader sources NRDPass compiles in the browser (see header, item 4). */
function fetchNRD() {
    const outDir = join(repoRoot, "tools/nrd-3.1.0");
    if (existsSync(join(outDir, "shader-files.json"))) {
        console.log("NRD shaders already present — skipping (delete tools/nrd-3.1.0 to refetch)");
        return;
    }
    console.log(`Cloning NRD ${NRD_TAG} (shaders + MathLib)…`);
    rmSync(outDir, { recursive: true, force: true });
    execFileSync("git", ["clone", "-q", "--depth", "1", "--branch", NRD_TAG, "--recurse-submodules", "--shallow-submodules", "https://github.com/NVIDIAGameWorks/RayTracingDenoiser", outDir], { stdio: "inherit" });
    // FXC/DXC register bindings would collide in WGSL (t0/u0/s0 all map to binding 0).
    const hlsli = join(outDir, "Shaders/Include/NRD.hlsli");
    const src = readFileSync(hlsli, "utf8");
    const start = src.indexOf("#elif( defined NRD_COMPILER_FXC || defined NRD_COMPILER_DXC )");
    const end = src.indexOf("#elif( defined NRD_COMPILER_PSSLC )");
    if (start < 0 || end < 0) throw new Error("NRD.hlsli: DXC macro block not found");
    const block = src.slice(start, end).replace("cbuffer globalConstants : register( b0 ) {", "cbuffer globalConstants {").replaceAll(": register( regName ## bindingIndex );", ";");
    writeFileSync(hlsli, src.slice(0, start) + block + src.slice(end));
    // `RWTexture2D<unorm float4>` has no WGSL form; the bound unorm format clamps on store anyway.
    const resources = join(outDir, "Shaders/Resources");
    for (const name of readdirSync(resources)) {
        const p = join(resources, name);
        const text = readFileSync(p, "utf8");
        let patched = text.replaceAll("<unorm ", "<");
        // Read-modify-write outputs (gInOut_*): WGSL read_write storage takes only r32 formats, so
        // each becomes a write target plus a read copy NRDPass fills before the dispatch, behind
        // a view with the original name (the __out keeps its register position).
        patched = patched.replace(/NRD_OUTPUT_TEXTURE\( RWTexture2D<([\w ]+)>, (gInOut_\w+), u, (\d+) \)/g, (_m, type, name, reg) =>
            [
                `NRD_OUTPUT_TEXTURE( RWTexture2D<${type}>, ${name}__out, u, ${reg} )`,
                `        Texture2D<${type}> ${name}__in; // WebFalcor: read copy of ${name}`,
                `        struct WebFalcorRW_${name} {`,
                `            __subscript(uint2 p) -> ${type} { get { return ${name}__in[p]; } [nonmutating] set { ${name}__out[p] = newValue; } }`,
                `            __subscript(int2 p) -> ${type} { get { return ${name}__in[p]; } [nonmutating] set { ${name}__out[p] = newValue; } }`,
                `        };`,
                `        static WebFalcorRW_${name} ${name};`,
            ].join("\n"),
        );
        // Output blocks with more than 8 storage textures (REBLUR's MipGen, which writes mips 1-3 of three
        // textures): WebGPU allows 8 per stage, so the mip outputs (_x2/_x4/_x8) become storage buffers
        // behind views with the original names; NRDPass copies each into its mip after the dispatch.
        patched = patched.replace(/NRD_OUTPUT_TEXTURE_START([\s\S]*?)NRD_OUTPUT_TEXTURE_END/g, (block) => {
            if ((block.match(/NRD_OUTPUT_TEXTURE\(/g) ?? []).length <= 8) return block;
            return block.replace(/NRD_OUTPUT_TEXTURE\( RWTexture2D<(\w+)>, (\w+_x\d), u, (\d+) \)/g, (_m, type, name) => {
                const get = type === "float" ? ".x" : type === "float2" ? ".xy" : "";
                const set = type === "float" ? "float4(newValue, 0, 0, 0)" : type === "float2" ? "float4(newValue, 0, 0)" : "newValue";
                return [
                    `RWStructuredBuffer<float4> ${name}__buf; // WebFalcor: buffer-backed ${name}`,
                    `        struct WebFalcorBuf_${name} {`,
                    `            __subscript(uint2 p) -> ${type} { get { return ${name}__buf[p.y * gWebFalcorBufStride + p.x]${get}; } [nonmutating] set { ${name}__buf[p.y * gWebFalcorBufStride + p.x] = ${set}; } }`,
                    `            __subscript(int2 p) -> ${type} { get { return ${name}__buf[p.y * gWebFalcorBufStride + p.x]${get}; } [nonmutating] set { ${name}__buf[p.y * gWebFalcorBufStride + p.x] = ${set}; } }`,
                    `        };`,
                    `        static WebFalcorBuf_${name} ${name};`,
                ].join("\n");
            });
        });
        if (patched.includes("gWebFalcorBufStride")) patched = `uniform uint gWebFalcorBufStride; // WebFalcor: row stride of the buffer-backed outputs\n${patched}`;
        // REBLUR HistoryFix reads its float4 outputs' previous values in ReconstructHistory: same
        // read copy, passed as an extra argument (see REBLUR_Common.hlsli below).
        if (name.includes("HistoryFix")) patched = patched.replace(/NRD_OUTPUT_TEXTURE\( RWTexture2D<float4>, (gOut_\w+), u, (\d+) \)/g, (m, out) => `${m}\n        Texture2D<float4> ${out}__in; // WebFalcor: read copy of ${out}`);
        if (patched !== text) writeFileSync(p, patched);
    }
    const common = join(outDir, "Shaders/Include/REBLUR/REBLUR_Common.hlsli");
    const commonSrc = readFileSync(common, "utf8");
    const commonPatched = commonSrc
        .replace("RWTexture2D<float4> texOut, Texture2D<float4> texIn, Texture2D<float> texScaledViewZ )", "RWTexture2D<float4> texOut, Texture2D<float4> texIn, Texture2D<float> texScaledViewZ, Texture2D<float4> texOutPrev )")
        .replace("    float4 c0 = texOut[ pixelPos ];", "    float4 c0 = texOutPrev[ pixelPos ];");
    if (commonPatched === commonSrc) throw new Error("REBLUR_Common.hlsli: ReconstructHistory not found");
    writeFileSync(common, commonPatched);
    const includes = join(outDir, "Shaders/Include/REBLUR");
    for (const name of readdirSync(includes).filter((n) => n.includes("HistoryFix"))) {
        const p = join(includes, name);
        const text = readFileSync(p, "utf8");
        const patched = text.replace(/(gOut_\w+), (gIn_\w+), gIn_ScaledViewZ \)/g, "$1, $2, gIn_ScaledViewZ, $1__in )");
        if (patched !== text) writeFileSync(p, patched);
    }
    // Registry keys: NRD's own tree under nrd/ (as Falcor's packman link), MathLib's STL.hlsli at the root.
    const files = [];
    const walk = (dir, rel) => {
        for (const name of readdirSync(dir)) {
            const p = join(dir, name);
            if (statSync(p).isDirectory()) walk(p, `${rel}/${name}`);
            else if (/\.(hlsl|hlsli)$/.test(name)) files.push({ path: `nrd${rel}/${name}`, url: `/tools/nrd-3.1.0${rel}/${name}` });
        }
    };
    walk(join(outDir, "Shaders"), "/Shaders");
    files.push({ path: "STL.hlsli", url: "/tools/nrd-3.1.0/External/MathLib/STL.hlsli" });
    writeFileSync(join(outDir, "shader-files.json"), JSON.stringify(files, null, 1) + "\n");
    console.log(`  NRD shaders ready under tools/nrd-3.1.0/ (${files.length} files)`);
}

/** Pyodide packages used by Falcor's Python scripts (and their dependencies from the lock). */
const PYODIDE_PACKAGES = ["numpy", "pillow"];

async function fetchPyodidePackages() {
    const pyodideDir = join(repoRoot, "node_modules/pyodide");
    const lock = JSON.parse(readFileSync(join(pyodideDir, "pyodide-lock.json"), "utf8"));
    const version = JSON.parse(readFileSync(join(pyodideDir, "package.json"), "utf8")).version;
    const outDir = join(repoRoot, "tools/pyodide-packages");
    mkdirSync(outDir, { recursive: true });
    const wanted = new Set();
    const visit = (name) => {
        if (wanted.has(name)) return;
        const pkg = lock.packages[name];
        if (!pkg) throw new Error(`pyodide-lock.json has no package '${name}'`);
        wanted.add(name);
        for (const d of pkg.depends ?? []) visit(d);
    };
    PYODIDE_PACKAGES.forEach(visit);
    for (const name of wanted) {
        const { file_name: file, sha256 } = lock.packages[name];
        const dst = join(outDir, file);
        const digest = (buf) => createHash("sha256").update(buf).digest("hex");
        if (existsSync(dst) && digest(readFileSync(dst)) === sha256) {
            console.log(`pyodide package ${file} already present — skipping`);
            continue;
        }
        const buf = await fetchBuffer(`https://cdn.jsdelivr.net/pyodide/v${version}/full/${file}`);
        if (digest(buf) !== sha256) throw new Error(`${file}: sha256 mismatch with pyodide-lock.json`);
        writeFileSync(dst, buf);
        console.log(`  ${file} (${(buf.length / 1e6).toFixed(1)} MB) -> tools/pyodide-packages/`);
    }
}

const t0 = Date.now();
if (!args.has("--skip-shaders")) {
    await fetchShaders();
    await fetchExternalHeaders();
}
if (!args.has("--skip-slang")) {
    await fetchSlangWasm(SLANG_VERSION, "tools/slang-wasm");
    await fetchSlangWasm(SLANG_AUTODIFF_VERSION, `tools/slang-wasm-${SLANG_AUTODIFF_VERSION}`);
}
if (!args.has("--skip-nrd")) fetchNRD();
if (!args.has("--skip-pyodide-packages")) await fetchPyodidePackages();
console.log(`\nWeb setup complete in ${((Date.now() - t0) / 1000).toFixed(1)}s. Next: npm run typecheck && npm run dev`);
