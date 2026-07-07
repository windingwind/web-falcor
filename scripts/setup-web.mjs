#!/usr/bin/env node
/**
 * Web setup — provisions everything needed to *use* web-falcor in a browser
 * WITHOUT cloning or building upstream Falcor.
 *
 * The runtime compiles Slang -> WGSL in the browser (shaders are specialized
 * per-scene, so a build-time-only pipeline can't cover them), which needs two
 * things a fresh checkout doesn't have (both are .gitignored — see README):
 *
 *   1. The upstream Falcor shader *sources* (~340 .slang/.slangh text files).
 *      Fetched from GitHub at the pinned commit into Falcor/Source/** — the
 *      same paths the dev server serves, so no code changes are needed. This is
 *      just the text shaders; it does NOT clone the repo or run the native
 *      CMake build (that is only for the test oracles — see scripts below).
 *   2. The slang-wasm compiler (the official per-release build) into
 *      tools/slang-wasm/.
 *
 * What this does NOT fetch: media/test scenes and the RTXDI/NanoVDB SDK headers
 * (obtained via Falcor's `setup.sh` for the full dev/test setup). The common
 * passes (path tracer, tone mapper, accumulate, scene debugger, ...) build and
 * run without them; RTXDI and GridVolume passes need the full setup.
 *
 * Usage: node scripts/setup-web.mjs [--skip-slang] [--skip-shaders]
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync, readdirSync, statSync, copyFileSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// Pinned upstream versions (keep in sync with README / DESIGN.md).
const FALCOR_COMMIT = "eb540f6748774680ce0039aaf3ac9279266ec521";
const SLANG_VERSION = "2026.12.2";
const SLANG_WASM_URL = `https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-wasm.zip`;
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

async function fetchShaders() {
    const manifestPath = join(repoRoot, "packages/falcor/shaders/generated/shader-file-list.json");
    if (!existsSync(manifestPath)) throw new Error(`missing shader manifest: ${manifestPath}`);
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

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
    console.log("  Note: RTXDI + GridVolume passes also need the RTXDI/NanoVDB headers (full Falcor setup.sh).");
}

async function fetchSlangWasm() {
    const outDir = join(repoRoot, "tools/slang-wasm");
    if (existsSync(join(outDir, "slang-wasm.wasm")) && existsSync(join(outDir, "slang-wasm.js"))) {
        console.log("slang-wasm already present — skipping (delete tools/slang-wasm to refetch)");
        return;
    }
    console.log(`Downloading slang-wasm ${SLANG_VERSION} (~25 MB)…`);
    const zipBytes = await fetchBuffer(SLANG_WASM_URL);
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
    console.log(`  slang-wasm ready under tools/slang-wasm/ (${copied} files)`);
}

const t0 = Date.now();
if (!args.has("--skip-shaders")) await fetchShaders();
if (!args.has("--skip-slang")) await fetchSlangWasm();
console.log(`\nWeb setup complete in ${((Date.now() - t0) / 1000).toFixed(1)}s. Next: npm run typecheck && npm run dev`);
